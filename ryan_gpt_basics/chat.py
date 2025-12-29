#!/usr/bin/env python3
"""Interactive chat with a trained GPT model."""

import argparse
import torch

from ryan_gpt_basics.generate import (
    load_model,
    load_tokenizer,
    generate_response,
    PRESETS,
)


def main():
    parser = argparse.ArgumentParser(description="Chat with RyanGPT")
    parser.add_argument("--checkpoint", default=PRESETS["chat"]["checkpoint"])
    parser.add_argument("--vocab", default=PRESETS["chat"]["vocab"])
    parser.add_argument("--merges", default=PRESETS["chat"]["merges"])
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.vocab, args.merges)
    
    print()
    print("╔" + "═" * 48 + "╗")
    print("║   Chat with RyanGPT! (type 'quit' to exit)     ║")
    print("╚" + "═" * 48 + "╝")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = generate_response(
            model, tokenizer, user_input,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )
        
        print(f"Bot: {response}")
        print()


if __name__ == "__main__":
    main()