import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path


def get_device():
    """
    Automatically detect and return the best available device.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class SimpleTokenizer:
    """A simple character-level tokenizer."""
    
    def __init__(self, vocab=None):
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>']
        
        if vocab is None:
            # Default vocab: common characters
            chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-\n")
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
            
            # Add special tokens if not present
            for token in self.special_tokens:
                if token not in self.char_to_idx:
                    idx = len(self.char_to_idx)
                    self.char_to_idx[token] = idx
                    self.idx_to_char[idx] = token
        else:
            self.char_to_idx = vocab['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
            self.special_tokens = vocab.get('special_tokens', self.special_tokens)
    
    @property
    def vocab_size(self):
        return len(self.char_to_idx)
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, self.char_to_idx['<UNK>']) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in indices])
    
    def add_text(self, text):
        """Expand vocabulary with characters from the given text."""
        added = False
        for ch in text:
            if ch not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[ch] = idx
                self.idx_to_char[idx] = ch
                added = True
        return added
    
    def fit_on_texts(self, texts):
        """Expand vocabulary using a collection of texts."""
        updated = False
        for text in texts:
            if self.add_text(text):
                updated = True
        return updated
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
                'special_tokens': self.special_tokens
            }, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            vocab = json.load(f)
        return cls(vocab)


class TransformerChatModel(nn.Module):
    """
    A simple transformer-based chat model.
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input token indices [batch_size, seq_length]
            attention_mask: Padding mask [batch_size, seq_length]
        
        Returns:
            logits: [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = x.shape
        
        # Embedding + positional encoding
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:, :seq_length, :]
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        # Transform
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # Output projection
        logits = self.output_layer(x)
        
        return logits
    
    def generate(self, tokenizer, prompt, max_length=200, temperature=1.0, top_k=50, device=None):
        """
        Generate text given a prompt.
        
        Args:
            tokenizer: Tokenizer instance
            prompt: Input text string
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            device: Device to run on
        
        Returns:
            Generated text string
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Encode prompt
        input_ids = [tokenizer.char_to_idx['<BOS>']]
        input_ids.extend(tokenizer.encode(prompt))
        if '<SEP>' in tokenizer.char_to_idx:
            input_ids.append(tokenizer.char_to_idx['<SEP>'])
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                if input_ids.shape[1] >= self.max_seq_length:
                    break
                
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.char_to_idx['<EOS>']:
                    break
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode
        generated_ids = input_ids[0].tolist()[1:]  # Skip BOS token
        generated_text = tokenizer.decode(generated_ids)
        
        # Remove prompt portion
        if '<SEP>' in generated_text:
            generated_text = generated_text.split('<SEP>', 1)[-1]
        
        for token in ['<PAD>', '<BOS>', '<EOS>']:
            generated_text = generated_text.replace(token, '')
        
        return generated_text.strip()
    
    def save_checkpoint(self, path, tokenizer=None, optimizer=None, epoch=None, loss=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'max_seq_length': self.max_seq_length,
                'dropout': self.dropout_rate,
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        
        torch.save(checkpoint, path)
        
        # Save tokenizer separately
        if tokenizer is not None:
            tokenizer_path = Path(path).parent / 'tokenizer.json'
            tokenizer.save(tokenizer_path)
    
    @classmethod
    def load_checkpoint(cls, path, vocab_size=None, device=None):
        """Load model from checkpoint."""
        if device is None:
            device = get_device()
        
        checkpoint = torch.load(path, map_location=device)
        
        # Get vocab size from the model state dict if not provided
        if vocab_size is None:
            vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
        
        # Create model
        config = checkpoint.get('model_config', {})
        model = cls(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 1024),
            max_seq_length=config.get('max_seq_length', 512),
            dropout=config.get('dropout', 0.1),
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, checkpoint
