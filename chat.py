import argparse
from pathlib import Path

import torch

from model import TransformerChatModel, SimpleTokenizer, get_device


def load_checkpoint(checkpoint_path: Path, tokenizer_path: Path, device: torch.device):
    """Load model and tokenizer."""
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    
    # Load model
    model, checkpoint = TransformerChatModel.load_checkpoint(checkpoint_path, vocab_size=tokenizer.vocab_size, device=device)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    
    return model, tokenizer


def chat(model: TransformerChatModel, tokenizer: SimpleTokenizer, device: torch.device, max_length: int = 200, temperature: float = 0.8):
    """Interactive chat session."""
    print("\nStarting chat session. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Generate response
        response = model.generate(tokenizer, user_input, max_length=max_length, temperature=temperature, device=device)
        
        # Print response
        print(f"Model: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer file")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum length of generated response")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    model, tokenizer = load_checkpoint(checkpoint_path, tokenizer_path, device)
    chat(model, tokenizer, device, max_length=args.max_length, temperature=args.temperature)


if __name__ == "__main__":
    main()
