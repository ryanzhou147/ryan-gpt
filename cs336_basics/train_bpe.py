from pathlib import Path
from typing import List, Tuple, Dict

from .pretokenizer import PreTokenizer
from .bpe_tokenizer import BPEProcessor

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
    # pretokenize the file in parallel
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
    import time
    start_time = time.perf_counter()
    # Call with a larger vocab size so merges will be performed for the small test file
    vocab, merges = train_bpe("./test.txt", 260, ["<|endoftext|>"])
    # Print a compact summary so running this script shows useful output
    print("vocab size:", len(vocab))
    # print a few vocab entries and merges for quick inspection
    print("BPE training complete.")
    end_time = time.perf_counter()
    print(f"Training took {end_time - start_time:.2f} seconds.")