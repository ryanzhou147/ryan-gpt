# ryan-gpt

A from-scratch GPT implementation with BPE tokenization, RoPE positional embeddings, and wandb logging.

## Quick Start

### Setup

Install [uv](https://github.com/astral-sh/uv) for dependency management:
```bash
pip install uv
# or
brew install uv
```

Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Or run any command directly with:
```bash
uv run python <script.py>
```

### Download Data

```bash
mkdir -p data && cd data

# TinyStories (small, good for testing)
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText (larger, more diverse)
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_train.txt.gz owt_valid.txt.gz

cd ..
```

## Training

### 1. Tokenize Data

Train a BPE tokenizer and convert text to token IDs:

```bash
python -m cs336_basics.train tokenize \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output_dir data/tinystories \
  --vocab_size 10000
```

This creates:
- `vocab.json` - token vocabulary
- `merges.txt` - BPE merge rules
- `TinyStoriesV2-GPT4-train.npy` - tokenized data

Tokenize validation set using the same vocabulary:
```bash
python -m cs336_basics.train tokenize \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output_dir data/tinystories
```

### 2. Train Model

```bash
python -m cs336_basics.train train \
  --train_data data/tinystories/TinyStoriesV2-GPT4-train.npy \
  --val_data data/tinystories/TinyStoriesV2-GPT4-valid.npy \
  --output_dir runs/tinystories \
  --vocab_size 10000 \
  --max_steps 20000
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--vocab_size` | required | Must match tokenizer |
| `--context_length` | 256 | Sequence length |
| `--num_layers` | 4 | Transformer layers |
| `--d_model` | 512 | Hidden dimension |
| `--num_heads` | 16 | Attention heads |
| `--lr` | 6e-4 | Max learning rate |
| `--batch_size` | 64 | Batch size |
| `--max_steps` | 20000 | Training iterations |
| `--resume` | None | Checkpoint to resume from |

Training logs to [Weights & Biases](https://wandb.ai). Set `--project` to customize.

### 3. Resume Training

```bash
python -m cs336_basics.train train \
  --train_data data/tinystories/TinyStoriesV2-GPT4-train.npy \
  --output_dir runs/tinystories \
  --vocab_size 10000 \
  --resume runs/tinystories/checkpoints/ckpt_10000.pt
```

## Generate Text

```bash
python -m cs336_basics.generate \
  --preset tinystories \
  --prompt "Once upon a time" \
  --max_tokens 200 \
  --temperature 0.7
```

Options:
- `--preset`: `tinystories` or `owt`
- `--checkpoint`: Override checkpoint path
- `--temperature`: Lower = more deterministic (default: 0.7)
- `--top_p`: Nucleus sampling threshold (default: 0.9)

## Model Architecture

- **Transformer LM** with pre-norm (RMSNorm)
- **RoPE** positional embeddings
- **SwiGLU** feed-forward blocks
- **BPE** tokenization with regex pretokenization

Default config (~23M parameters):
```
vocab_size: 10000
context_length: 256
num_layers: 4
d_model: 512
num_heads: 16
d_ff: 1344
```

## Project Structure

```
cs336_basics/
├── train.py          # Training CLI
├── generate.py       # Text generation
├── logger.py         # Wandb wrapper
├── utility.py        # Core utilities
├── transformer/      # Model components
│   ├── transformer.py
│   ├── multihead_self_attention.py
│   ├── rope.py
│   ├── rmsnorm.py
│   └── swiglu.py
├── tokenizer/        # BPE tokenizer
│   ├── bpe_tokenizer.py
│   └── train_bpe.py
└── optimizer/        # AdamW implementation
    └── adamw.py
```

## Tests

```bash
uv run pytest
```

