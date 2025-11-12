import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """Add positional encoding to input embeddings."""
        return x + self.pe[:, :x.size(1), :]


class TransformerChatModel(nn.Module):
    """Simple transformer-based chat model."""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        # Store hyperparameters as instance attributes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        # Model components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, src):
        """Forward pass through the model."""
        seq_len = src.size(1)
        
        # Create causal mask
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through transformer with causal mask
        output = self.transformer(src, mask=mask, is_causal=True)
        
        # Project to vocabulary
        logits = self.fc_out(output)
        
        return logits
    
    def generate(self, tokenizer, prompt, max_length=100, temperature=1.0, device='cpu'):
        """Generate text given a prompt."""
        self.eval()
        
        # Encode prompt
        bos_id = tokenizer.char_to_idx['<BOS>']
        eos_id = tokenizer.char_to_idx['<EOS>']
        sep_id = tokenizer.char_to_idx['<SEP>']
        
        token_ids = [bos_id]
        token_ids.extend(tokenizer.encode(prompt))
        token_ids.append(sep_id)
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
                
                # Get predictions
                logits = self(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if EOS
                if next_token == eos_id:
                    break
                
                token_ids.append(next_token)
        
        # Decode response (everything after SEP token)
        sep_idx = token_ids.index(sep_id)
        response_ids = token_ids[sep_idx + 1:]
        response = tokenizer.decode(response_ids)
        
        return response
    
    def save_checkpoint(self, path, tokenizer=None, optimizer=None, epoch=0, loss=0.0, iteration=None):
        """Save model checkpoint with all training state."""
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration if iteration is not None else 0,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'max_seq_length': self.max_seq_length,
                'dropout': self.dropout,
            }
        }
        
        if tokenizer is not None:
            checkpoint['tokenizer'] = {
                'char_to_idx': tokenizer.char_to_idx,
                'idx_to_char': tokenizer.idx_to_char
            }
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path, vocab_size=None, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        # Get model config
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = cls(**config)
        elif vocab_size is not None:
            model = cls(vocab_size=vocab_size)
        else:
            raise ValueError("Either checkpoint must contain model_config or vocab_size must be provided")
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint


class SimpleTokenizer:
    """Character-level tokenizer with special tokens."""
    
    def __init__(self):
        self.char_to_idx = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<SEP>': 3,
            '<UNK>': 4,
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.next_idx = 5
    
    def fit_on_texts(self, texts):
        """Add characters from texts to vocabulary."""
        for text in texts:
            for char in text:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = self.next_idx
                    self.idx_to_char[self.next_idx] = char
                    self.next_idx += 1
    
    def encode(self, text):
        """Encode text to list of token IDs."""
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
    
    def decode(self, token_ids):
        """Decode list of token IDs to text."""
        chars = [self.idx_to_char.get(idx, '<UNK>') for idx in token_ids]
        # Filter out special tokens
        chars = [c for c in chars if c not in ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '<UNK>']]
        return ''.join(chars)
    
    @property
    def vocab_size(self):
        """Return vocabulary size."""
        return len(self.char_to_idx)
    
    def save(self, path):
        """Save tokenizer to JSON file."""
        import json
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()},
            'next_idx': self.next_idx
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from JSON file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        
        # Handle legacy tokenizers without next_idx
        if 'next_idx' in data:
            tokenizer.next_idx = data['next_idx']
        else:
            # Reconstruct next_idx from existing vocabulary
            tokenizer.next_idx = max(tokenizer.idx_to_char.keys()) + 1
        
        return tokenizer

        
        return tokenizer


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
