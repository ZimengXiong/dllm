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



def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')



def train(model, train_loader, val_loader, optimizer, criterion, device, args, tokenizer, checkpoint_dir, start_iter=0):
    """Train with unified progress bar and iteration-based checkpointing/evaluation."""
    
    total_iterations = args.total_iters
    checkpoint_interval = max(1, total_iterations // 10)
    eval_interval = max(1, total_iterations // 10)
    
    batches_per_epoch = len(train_loader)
    
    progress_bar = tqdm(initial=start_iter, total=total_iterations, desc="Training", unit="iter")
    
    global_iter = start_iter
    epoch = start_iter // batches_per_epoch
    best_val_loss = float('inf')
    
    running_loss = 0
    running_count = 0
    
    # Skip batches to resume from correct position
    batches_to_skip = start_iter % batches_per_epoch
    
    while global_iter < total_iterations:
        model.train()
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            # Skip batches if resuming mid-epoch
            if batch_idx < batches_to_skip:
                continue
            batches_to_skip = 0  # Only skip on first epoch after resume
            
            if global_iter >= total_iterations:
                break
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            running_count += 1
            global_iter += 1
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'epoch': f'{epoch+1}',
                'iter': f'{global_iter}/{total_iterations}',
                'loss': f'{loss.item():.4f}',
                'avg': f'{running_loss/running_count:.4f}'
            })
            
            # Evaluation
            # In the train() function, replace the evaluation block with:

            # Evaluation
            if global_iter % eval_interval == 0 or global_iter == total_iterations:
                if val_loader is not None:
                    val_loss = evaluate_model(model, val_loader, criterion, device)
                    progress_bar.write(f"[Iter {global_iter}] Train: {running_loss/running_count:.4f} | Val: {val_loss:.4f}")
                    
                    # Sample generation
                    model.eval()
                    sample_text = model.generate(tokenizer, "Hello", max_length=50, temperature=0.8, device=device)
                    progress_bar.write(f"[Iter {global_iter}] Sample - Hello -> {sample_text}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = checkpoint_dir / "best_model.pt"
                        model.save_checkpoint(best_path, tokenizer=tokenizer, optimizer=optimizer, 
                                            epoch=epoch, loss=val_loss, iteration=global_iter)
                        progress_bar.write(f"[Iter {global_iter}] New best model saved (val_loss: {val_loss:.4f})")
                else:
                    progress_bar.write(f"[Iter {global_iter}] Train: {running_loss/running_count:.4f}")
                    
                    # Sample generation even without validation set
                    model.eval()
                    sample_text = model.generate(tokenizer, "Hello", max_length=50, temperature=0.8, device=device)
                    progress_bar.write(f"[Iter {global_iter}] Sample - Hello -> {sample_text}")
                
                running_loss = 0
                running_count = 0
                model.train()

            
            # Checkpoint saving
            if global_iter % checkpoint_interval == 0 or global_iter == total_iterations:
                checkpoint_path = checkpoint_dir / f"checkpoint_iter_{global_iter}.pt"
                model.save_checkpoint(checkpoint_path, tokenizer=tokenizer, optimizer=optimizer, 
                                    epoch=epoch, loss=loss.item(), iteration=global_iter)
                progress_bar.write(f"[Iter {global_iter}] Checkpoint saved")
        
        epoch += 1
    
    progress_bar.close()
    return best_val_loss



def main():
    parser = argparse.ArgumentParser(description="Train a simple chat model")
    parser.add_argument("--data", type=str, default="data/", help="Path to training data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--total-iters", type=int, default=10000, help="Total training iterations")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize or load state
    start_iter = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load tokenizer from checkpoint
        if 'tokenizer' in checkpoint:
            tokenizer = SimpleTokenizer()
            tokenizer.char_to_idx = checkpoint['tokenizer']['char_to_idx']
            tokenizer.idx_to_char = checkpoint['tokenizer']['idx_to_char']
            print("Loaded tokenizer from checkpoint")
        else:
            tokenizer_path = checkpoint_dir / "tokenizer.json"
            if tokenizer_path.exists():
                tokenizer = SimpleTokenizer.load(tokenizer_path)
            else:
                raise ValueError("No tokenizer found in checkpoint or checkpoint directory")
        
        # Get iteration from checkpoint
        start_iter = checkpoint.get('iteration', 0)
        print(f"Resuming from iteration {start_iter}")
        
    else:
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        if tokenizer_path.exists():
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = SimpleTokenizer.load(tokenizer_path)
        else:
            print("Creating new tokenizer")
            tokenizer = SimpleTokenizer()
    
    # Load dataset
    print(f"Loading data from {args.data}")
    dataset = ConversationDataset(args.data, tokenizer, max_length=args.max_seq_length)
    tokenizer.save(checkpoint_dir / "tokenizer.json")
    
    # Split dataset
    if len(dataset) < 2:
        train_dataset = dataset
        val_dataset = None
    else:
        train_size = max(1, int(0.9 * len(dataset)))
        if train_size == len(dataset):
            train_size = len(dataset) - 1
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Consistent splits for resuming
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize or load model
    if args.resume:
        model, checkpoint = TransformerChatModel.load_checkpoint(args.resume, vocab_size=tokenizer.vocab_size, device=device)
        print("Model loaded from checkpoint")
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
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"Total iterations: {args.total_iters} (starting from {start_iter})")
    print(f"Checkpoint interval: {args.total_iters // 10}")
    print(f"Eval interval: {args.total_iters // 10}\n")
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Load optimizer state if resuming
    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state loaded from checkpoint\n")
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])
    
    # Train
    best_val_loss = train(model, train_loader, val_loader, optimizer, criterion, device, args, tokenizer, checkpoint_dir, start_iter)
    
    # Save final checkpoint
    final_path = checkpoint_dir / "final_model.pt"
    model.save_checkpoint(final_path, tokenizer=tokenizer, optimizer=optimizer, epoch=0, loss=best_val_loss, iteration=args.total_iters)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")



if __name__ == "__main__":
    main()
