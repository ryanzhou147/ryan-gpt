import torch
import argparse
from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.tokenizer.bpe_tokenizer import BPEProcessor
from cs336_basics.utility import decode

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "owt"])
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: best for dataset)")
parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
parser.add_argument("--max_tokens", type=int, default=300)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
args = parser.parse_args()

# Dataset-specific paths
match args.dataset:
    case "tinystories":
        vocab_path = "data/tinystories/vocab.json"
        merges_path = "data/tinystories/merges.txt"
        default_checkpoint = "data/tinystories/bs_64/checkpoints/checkpoint_final.pt"
        default_prompt = "Once upon a time"
    case "owt":
        vocab_path = "data/owt/vocab.json"
        merges_path = "data/owt/merges.txt"
        default_checkpoint = "data/owt/main_experiment/checkpoints/checkpoint_final.pt"
        default_prompt = "The scientists discovered that"

checkpoint_path = args.checkpoint or default_checkpoint
prompt = args.prompt or default_prompt

# Load tokenizer
tokenizer = BPEProcessor.from_files(vocab_path, merges_path, ["<|endoftext|>"])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerLM(
    vocab_size=10000,
    context_length=256,
    num_layers=4,
    d_model=512,
    num_heads=16,
    d_ff=1344,
    with_rope=True,
    rope_theta=10000.0,
).to(device)

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

eos_id = tokenizer.vocab.get(b"<|endoftext|>")

prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

print("=" * 70)
print(f"Dataset: {args.dataset}")
print(f"Prompt: {prompt}")
print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
print("=" * 70)

with torch.no_grad():
    generated_ids = decode(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        context_length=256,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )

generated_text = tokenizer.decode(generated_ids.tolist())
print(f"\nTotal tokens: {len(generated_ids)}")
print(f"\n{generated_text}")