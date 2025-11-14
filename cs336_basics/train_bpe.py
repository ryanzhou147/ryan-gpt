from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
from time import time
import cProfile, pstats

from cs336_basics.pretokenizer import PreTokenizer
from cs336_basics.bpe_tokenizer import BPEProcessor


def train_bpe(input_path: str | Path, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: path to training text
        vocab_size: total desired vocabulary size (including initial 256 byte tokens and special tokens)
        special_tokens: list of special token strings to include in the vocabulary

    Returns:
        vocab: mapping from int id -> bytes
        merges: list of merges as tuples of bytes
    """
    input_path = Path(input_path)

    # Step 1: pretokenize using provided PreTokenizer
    pretokenizer = PreTokenizer(special_tokens)
    # pretokenize_file_parallel returns the bytes->count mapping
    pretokenizer.pretokenize_file_parallel(str(input_path))

    # Step 2: initialize BPE processor with the pretokenizer results
    bpe = BPEProcessor(pretokenizer)

    # initial vocab size includes 256 byte tokens and special tokens
    initial_vocab = 256 + len(special_tokens)
    if vocab_size <= initial_vocab:
        # no merges to perform
        return bpe.vocab, bpe.merges

    num_merges = vocab_size - initial_vocab
    bpe.run_bpe(num_merges)

    print(bpe.vocab, bpe.merges)
    return bpe.vocab, bpe.merges

if __name__ == "__main__":
    start_time = time()
    pr = cProfile.Profile()

    result = pr.runcall(train_bpe,
        input_path='data/TinyStoriesV2-GPT4-valid.txt', vocab_size=10000, special_tokens=['<|endoftext|>']
    )

    end_time = time()

    stats = pstats.Stats(pr).strip_dirs()
    stats.sort_stats("cumulative")  # sort by cumulative time
    stats.print_stats(10)           # print top 10 functions

    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    vocab, merges = result
    max_token = max(vocab.values(), key=len)
    print(max_token, len(max_token))

    top_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    print(f"{'Vocab ID':<8} {' Token':<20} {'Bytes':<5}")
    print("-" * 40)

    for vocab_id, token_bytes in top_tokens:
        try:
            token_str = token_bytes.decode("utf-8")  # decode for readability
        except UnicodeDecodeError:
            token_str = repr(token_bytes)  # fallback if not valid UTF-8
        print(f"{vocab_id:<8} {token_str:<20} {len(token_bytes):<5}")