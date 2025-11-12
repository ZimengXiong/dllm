import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import json

from model import TransformerChatModel, SimpleTokenizer, get_device


class ConversationDataset(Dataset):
    """Dataset for tab-separated prompt/response conversations."""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        data_path = Path(data_path)
        files = []
        if data_path.is_file():
            files = [data_path]
        elif data_path.is_dir():
            files = sorted(data_path.glob("*.txt"))
            if not files:
                raise ValueError(f"No .txt files found in directory {data_path}")
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" not in line:
                        raise ValueError(
                            f"Expected tab-separated prompt and response in {file_path} on line {line_number}."
                        )
                    prompt, response = [part.strip() for part in line.split("\t", 1)]
                    if not prompt or not response:
                        continue
                    self.examples.append((prompt, response))
                    self.tokenizer.fit_on_texts([prompt, response])

        if not self.examples:
            raise ValueError(f"No valid prompt/response pairs found in {data_path}")

        self.bos_id = self.tokenizer.char_to_idx['<BOS>']
        self.eos_id = self.tokenizer.char_to_idx['<EOS>']
        self.sep_id = self.tokenizer.char_to_idx['<SEP>']
        self.pad_id = self.tokenizer.char_to_idx['<PAD>']

        print(f"Loaded {len(self.examples)} prompt/response pairs from {len(files)} file(s)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, response = self.examples[idx]

        token_ids = [self.bos_id]
        token_ids.extend(self.tokenizer.encode(prompt))
        token_ids.append(self.sep_id)
        token_ids.extend(self.tokenizer.encode(response))
        token_ids.append(self.eos_id)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.pad_id] * (self.max_length - len(token_ids)))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)

        return input_ids, target_ids


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Validating"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train a simple chat model")
    parser.add_argument("--data", type=str, default="data/", help="Path to training data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize or load tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SimpleTokenizer.load(tokenizer_path)
    else:
        print("Creating new tokenizer")
        tokenizer = SimpleTokenizer()
        tokenizer.save(tokenizer_path)
    
    # Create dataset and dataloader
    print(f"Loading data from {args.data}")
    dataset = ConversationDataset(args.data, tokenizer, max_length=args.max_seq_length)
    tokenizer.save(tokenizer_path)
    
    # Split into train and validation (90/10 split) when possible
    if len(dataset) < 2:
        train_dataset = dataset
        val_dataset = None
    else:
        train_size = max(1, int(0.9 * len(dataset)))
        if train_size == len(dataset):
            train_size = len(dataset) - 1
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model, checkpoint = TransformerChatModel.load_checkpoint(args.resume, vocab_size=tokenizer.vocab_size, device=device)
        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        print("Creating new model")
        model = TransformerChatModel(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_length=args.max_seq_length
        )
        model.to(device)
        start_epoch = 0
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")
        else:
            val_loss = train_loss
            print("No validation set (dataset too small)")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        model.save_checkpoint(checkpoint_path, tokenizer=tokenizer, optimizer=optimizer, epoch=epoch, loss=train_loss)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / "best_model.pt"
            model.save_checkpoint(best_model_path, tokenizer=tokenizer, optimizer=optimizer, epoch=epoch, loss=val_loss)
            print(f"Saved best model to {best_model_path}")
        
        # Test generation
        print("\n--- Sample Generation ---")
        test_prompt = "Hello"
        model.eval()
        generated = model.generate(tokenizer, test_prompt, max_length=100, temperature=0.8, device=device)
        print(f"Prompt: {test_prompt}")
        print(f"Generated: {generated}")
        print("-------------------------\n")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
