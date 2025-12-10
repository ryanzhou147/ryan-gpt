import torch
import numpy as np
import argparse
import json
from pathlib import Path

from cs336_basics.logger import Logger
from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.optimizer.cross_entropy import CrossEntropyLoss
from cs336_basics.utility import (
    learning_rate_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
    data_loading,
)

def tokenize(input_path: str, output_dir: str, vocab_size: int = 10000, special_tokens: list[str] = None):
    """Train BPE tokenizer and tokenize a text file to .npy"""
    from cs336_basics.tokenizer.train_bpe import train_bpe
    from cs336_basics.tokenizer.bpe_tokenizer import BPEProcessor
    
    special_tokens = special_tokens or ["<|endoftext|>"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    
    # Train tokenizer if doesn't exist
    if not vocab_path.exists():
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        # Save
        with open(vocab_path, "w") as f:
            json.dump({str(k): v.hex() for k, v in vocab.items()}, f)
        with open(merges_path, "w") as f:
            for l, r in merges:
                f.write(f"{l.hex()} {r.hex()}\n")
    
    tokenizer = BPEProcessor.from_files(str(vocab_path), str(merges_path), special_tokens)
    
    # Tokenize using encode_iterable for memory-efficient streaming
    print(f"Tokenizing {input_path}...")
    out_path = output_dir / (Path(input_path).stem + ".npy")
    
    # File objects are iterables that yield lines - pass directly to encode_iterable
    import time
    start_time = time.time()
    ids = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            ids.append(token_id)
            if i > 0 and i % 10_000_000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                print(f"  {i:,} tokens ({elapsed:.1f}s, {rate:,.0f} tok/s)")
    ids = np.array(ids, dtype=np.uint16)
    total_time = time.time() - start_time
    print(f"Tokenization complete: {len(ids):,} tokens in {total_time:.1f}s")
    
    np.save(out_path, ids)
    print(f"Saved {len(ids):,} tokens to {out_path}")
    return str(out_path), len(tokenizer.vocab)


def train(args):
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    run_name = f"{Path(args.output_dir).name}"
    config = vars(args)
    logger = Logger(project="cs336", name=run_name, config=config)

    # Load data with memmap
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r') if args.val_data else None
    print(f"Train tokens: {len(train_data):,}")

    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        with_rope=True,
        rope_theta=args.rope_theta,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Resume from checkpoint
    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Training loop
    model.train()
    for iter in range(start_iter, args.max_iters + 1):
        # LR schedule
        lr = learning_rate_schedule(iter, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = CrossEntropyLoss(logits, y).forward()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging
        if iter % args.log_interval == 0:
            metrics = {"loss": loss.item(), "lr": lr}
            msg = f"iter {iter} | loss {loss.item():.4f} | lr {lr:.2e}"
            if val_data is not None and iter % args.eval_interval == 0:
                model.eval()
                val_losses = []
                for _ in range(args.eval_iters):
                    vx, vy = data_loading(val_data, args.batch_size, args.context_length, device)
                    with torch.no_grad():
                        val_losses.append(CrossEntropyLoss(model(vx), vy).forward().item())
                model.train()
                val_loss = sum(val_losses) / len(val_losses)
                metrics["val_loss"] = val_loss
                msg += f" | val_loss {val_loss:.4f}"
            logger.log(iter, metrics)
            print(msg)

        # Checkpoint
        if iter > 0 and iter % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iter, ckpt_dir / f"checkpoint_{iter}.pt")

    # Final checkpoint
    save_checkpoint(model, optimizer, args.max_iters, ckpt_dir / "checkpoint_final.pt")
    logger.finish()
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    
    # Tokenize subcommand
    tok = sub.add_parser("tokenize", help="Train tokenizer and tokenize data")
    tok.add_argument("--input", type=str, required=True, help="Input text file")
    tok.add_argument("--output_dir", type=str, required=True, help="Output directory")
    tok.add_argument("--vocab_size", type=int, default=10000)
    
    # Train subcommand
    tr = sub.add_parser("train", help="Train model")
    # Data
    tr.add_argument("--train_data", type=str, required=True)
    tr.add_argument("--val_data", type=str, default=None)
    tr.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and logs")
    # Model
    tr.add_argument("--vocab_size", type=int, required=True)
    tr.add_argument("--context_length", type=int, default=256)
    tr.add_argument("--num_layers", type=int, default=4)
    tr.add_argument("--d_model", type=int, default=512)
    tr.add_argument("--num_heads", type=int, default=16)
    tr.add_argument("--d_ff", type=int, default=1344)
    tr.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer
    tr.add_argument("--max_lr", type=float, default=6e-4)
    tr.add_argument("--min_lr", type=float, default=6e-5)
    tr.add_argument("--beta1", type=float, default=0.9)
    tr.add_argument("--beta2", type=float, default=0.98)
    tr.add_argument("--eps", type=float, default=1e-8)
    tr.add_argument("--weight_decay", type=float, default=0.1)
    tr.add_argument("--max_grad_norm", type=float, default=1.0)
    # Schedule
    tr.add_argument("--warmup_iters", type=int, default=1000)
    tr.add_argument("--cosine_cycle_iters", type=int, default=20000)
    # Training
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--max_iters", type=int, default=20000)
    tr.add_argument("--seed", type=int, default=42)
    # Logging and checkpointing
    tr.add_argument("--log_interval", type=int, default=100)
    tr.add_argument("--eval_interval", type=int, default=500)
    tr.add_argument("--eval_iters", type=int, default=20)
    tr.add_argument("--checkpoint_interval", type=int, default=2000)
    tr.add_argument("--resume_from", type=str, default=None)
    
    args = p.parse_args()
    if args.command == "tokenize":
        tokenize(args.input, args.output_dir, args.vocab_size)
    elif args.command == "train":
        train(args)
    else:
        p.print_help()
