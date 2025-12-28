#!/usr/bin/env python3
"""
Extract and format DailyDialog dataset for conversational fine-tuning.
"""

import os
import zipfile
import urllib.request
from pathlib import Path


def extract_dailydialog(
    output_dir: str = "data/dailydialog",
    max_dialogues: int = None,
):
    """
    Download and format DailyDialog dataset from HuggingFace repo.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download from HuggingFace repo - use resolve/main for direct download
    base_url = "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main"
    
    splits = ['train', 'validation', 'test']
    
    print("Downloading DailyDialog from HuggingFace...")
    
    for split_name in splits:
        zip_path = raw_dir / f"{split_name}.zip"
        url = f"{base_url}/{split_name}.zip"
        
        if not zip_path.exists():
            print(f"  Downloading {split_name} from {url}...")
            urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        extract_dir = raw_dir / split_name
        if not extract_dir.exists():
            print(f"  Extracting {split_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(raw_dir)
    
    def process_split(split_name: str, output_path: Path, max_dialogues: int = None):
        """Process a single split and save to file."""
        
        # Find dialogues file - check multiple possible locations
        possible_paths = [
            raw_dir / split_name / "dialogues.txt",
            raw_dir / split_name / f"dialogues_{split_name}.txt",
            raw_dir / f"dialogues_{split_name}.txt",
        ]
        
        dialogues_path = None
        for p in possible_paths:
            if p.exists():
                dialogues_path = p
                break
        
        if dialogues_path is None:
            # List what's actually in the directory
            print(f"  Looking for dialogues in {raw_dir / split_name}...")
            if (raw_dir / split_name).exists():
                for f in (raw_dir / split_name).iterdir():
                    print(f"    Found: {f}")
            return 0
        
        print(f"Processing {split_name} split from {dialogues_path}...")
        
        dialogue_count = 0
        
        with open(dialogues_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                # DailyDialog format: utterances separated by __eou__
                utterances = [u.strip() for u in line.split('__eou__') if u.strip()]
                
                if len(utterances) < 2:
                    continue
                
                # Format as conversation
                formatted = ""
                for i, turn in enumerate(utterances):
                    if i % 2 == 0:
                        formatted += f"<|user|>\n{turn}\n"
                    else:
                        formatted += f"<|assistant|>\n{turn}\n"
                
                formatted += "<|endoftext|>\n\n"
                f_out.write(formatted)
                
                dialogue_count += 1
                
                if dialogue_count % 5000 == 0:
                    print(f"  Processed {dialogue_count} dialogues...")
                
                if max_dialogues and dialogue_count >= max_dialogues:
                    break
        
        print(f"  Saved {dialogue_count} dialogues to {output_path}")
        return dialogue_count
    
    # Process all splits
    results = {}
    for split_name in splits:
        output_path = output_dir / f"{split_name}.txt"
        count = process_split(split_name, output_path, max_dialogues)
        results[split_name] = count
    
    # Print summary
    print("\n" + "=" * 50)
    print("DailyDialog extraction complete!")
    print("=" * 50)
    for split_name, count in results.items():
        print(f"  {split_name.capitalize():12}: {count:,} dialogues -> {output_dir / f'{split_name}.txt'}")
    print("=" * 50)
    
    return output_dir


def preview_dialogues(file_path: str, n: int = 3):
    """Preview n dialogues from the extracted file."""
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"\nPreviewing {n} dialogues from {file_path}:")
    print("-" * 50)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    dialogues = content.split("<|endoftext|>\n\n")
    
    for i, dialogue in enumerate(dialogues[:n]):
        if dialogue.strip():
            print(f"\n[Dialogue {i + 1}]")
            print(dialogue.strip())
            print("<|endoftext|>")
            print("-" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract DailyDialog dataset")
    parser.add_argument("--output_dir", default="data/dailydialog", help="Output directory")
    parser.add_argument("--max_dialogues", type=int, default=None, help="Max dialogues per split")
    parser.add_argument("--preview", action="store_true", help="Preview extracted dialogues")
    
    args = parser.parse_args()
    
    output_dir = extract_dailydialog(
        output_dir=args.output_dir,
        max_dialogues=args.max_dialogues,
    )
    
    if args.preview:
        preview_dialogues(Path(output_dir) / "train.txt", n=3)