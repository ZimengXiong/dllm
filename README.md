# Simple Chat Model

A minimal, self-contained transformer-based chat model with training and inference capabilities. Built with PyTorch and managed by `uv`.

## Features

- **Simple Structure**: Just 3 Python files and 2 directories
- **Device Support**: Automatic detection and usage of CUDA, MPS (Apple Silicon), or CPU
- **Character-level**: Simple character-level tokenizer (no external dependencies)
- **Training**: Train your own chat models from scratch
- **Inference**: Interactive chat interface

## Installation

This project uses `uv` for package management. Install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the project:

```bash
uv sync
```

## Project Structure

```
.
├── model.py          # Model architecture and utilities
├── train.py          # Training script
├── chat.py           # Interactive chat interface
├── data/             # Training data directory
│   └── sample.txt    # Example training data
├── checkpoints/      # Model checkpoints directory
└── pyproject.toml    # Project configuration
```

## Usage

### Training

Train a model on your data:

```bash
uv run python train.py --data data/ --epochs 10 --batch-size 8
```

Options:
- `--data`: Path to training data (file or directory)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--d-model`: Model dimension (default: 256)
- `--nhead`: Number of attention heads (default: 8)
- `--num-layers`: Number of transformer layers (default: 6)
- `--max-seq-length`: Maximum sequence length (default: 512)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints/)
- `--resume`: Path to checkpoint to resume from

### Chat

Start an interactive chat session with a trained model:

```bash
uv run python chat.py
```

Options:
- `--checkpoint`: Path to model checkpoint (default: checkpoints/best_model.pt)
- `--tokenizer`: Path to tokenizer file (default: checkpoints/tokenizer.json)
- `--max-length`: Maximum length of generated response (default: 200)
- `--temperature`: Temperature for sampling (default: 0.8)

Type 'exit' or 'quit' to end the chat session.

## Data Format

Training data should be plain text files with one prompt/response pair per line using a tab (`\t`) separator:

```
Hello, how are you?    I am doing great, thanks!
What's your favorite color?    I like blue.
```

Place your data files in the `data/` directory. The training script can read:
- A single `.txt` file
- Multiple `.txt` files in a directory

## Device Support

The model automatically detects and uses the best available device:
1. **CUDA** (NVIDIA GPUs) - if available
2. **MPS** (Apple Silicon) - if available
3. **CPU** - fallback

No manual configuration required!

## Model Architecture

- Transformer-based architecture
- Character-level tokenization
- Positional encoding
- Causal masking for autoregressive generation
- Configurable model size (default: 256d, 8 heads, 6 layers)

## Checkpoints

During training, checkpoints are saved to the `checkpoints/` directory:
- `checkpoint_epoch_N.pt` - saved after each epoch
- `best_model.pt` - best performing model (lowest validation loss)
- `tokenizer.json` - tokenizer vocabulary

## License

MIT License - see LICENSE file for details
