import torch
import numpy as np
import argparse
import os

from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.optimizer.adamw import AdamW
from cs336_basics.optimizer.cross_entropy import CrossEntropyLoss
from cs336_basics.utility import (
    learning_rate_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
    data_loading,
    Logger,
)


def train(args):
    # Setup
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Logger
    logger = Logger(args.log_file) if args.log_file else Logger()

    # Load data with memmap
    train_data = torch.from_numpy(np.load(args.train_data, mmap_mode='r').astype(np.int64))
    val_data = torch.from_numpy(np.load(args.val_data, mmap_mode='r').astype(np.int64)) if args.val_data else None
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

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    model.train()
    for iter in range(start_iter, args.max_iters):
        # LR schedule
        lr = learning_rate_schedule(iter, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = CrossEntropyLoss(logits, y).forward()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Logging
        if iter % args.log_interval == 0:
            metrics = {"train_loss": loss.item(), "ppl": loss.exp().item(), "lr": lr}
            msg = f"iter {iter} | loss {loss.item():.4f} | ppl {loss.exp().item():.2f} | lr {lr:.2e}"
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
                metrics["val_ppl"] = np.exp(val_loss)
                msg += f" | val_loss {val_loss:.4f} | val_ppl {np.exp(val_loss):.2f}"
            logger.log(iter, metrics)
            print(msg)

        # Checkpoint
        if args.checkpoint_dir and iter > 0 and iter % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iter, os.path.join(args.checkpoint_dir, f"ckpt_{iter}.pt"))

    # Final checkpoint
    if args.checkpoint_dir:
        save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.checkpoint_dir, "ckpt_final.pt"))
    if args.log_file:
        logger.save(args.log_file.replace(".jsonl", ".json"))
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, default=None)
    # Model
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer
    p.add_argument("--max_lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Schedule
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--cosine_cycle_iters", type=int, default=20000)
    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_iters", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    # Logging/Checkpoints
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--checkpoint_interval", type=int, default=2000)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--log_file", type=str, default=None, help="Path to save logs (jsonl)")
    
    train(p.parse_args())
