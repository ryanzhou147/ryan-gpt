"""
Pre-tokenization module for text files.
Implements parallel pre-tokenization with special token handling.
"""
import os
import regex as re
from multiprocessing import Pool
from typing import BinaryIO, Dict

# Default number of worker processes used when splitting files
# Use single-process by default to ensure deterministic behavior in tests
NUM_PRETOKENIZING_PROCESSES = 4

# module-level globals for workers
_worker_pat: re.Pattern | None = None
_worker_special_tokens: list[str] | None = None
_worker_initialized = False

def _init_worker(pat_str: str, special_tokens: list[str]):
    """Initializer run once in each worker process: compile pattern and store tokens."""
    global _worker_pat, _worker_special_tokens, _worker_initialized
    if _worker_initialized:
        return
    _worker_pat = re.compile(pat_str)
    _worker_special_tokens = special_tokens
    _worker_initialized = True


def _process_chunk_worker(start: int, end: int, filepath: str) -> Dict[str, int]:
    """Module-level worker: read bytes, remove special tokens, decode and tokenize."""
    assert _worker_pat is not None and _worker_special_tokens is not None, "Worker not initialized"
    local_counts: Dict[str, int] = {}

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

        # normalize line endings in bytes, then decode
        chunk_bytes = chunk_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        chunk_data = chunk_bytes.decode("utf-8", errors="ignore")

        for m in _worker_pat.finditer(chunk_data):
            tok = m.group()
            local_counts[tok] = local_counts.get(tok, 0) + 1

    return local_counts
 
class PreTokenizer:
    def __init__(self, special_tokens: list[str]) -> None:
        self.special_tokens: list[str] = special_tokens
        self.PAT = None # Pre-tokenization pattern
        self._initialize_PAT()
        self.global_pretokenization_dict: Dict[str, int] = {} # Global frequency table
        self.pretokenization_dict_to_bytes: Dict[tuple[bytes], int] = {} # Global frequency table in bytes
        
    def _initialize_PAT(self) -> None:
        special_patterns = [re.escape(c) for c in self.special_tokens]
        special_group = "|".join(special_patterns)
        # Use the canonical GPT-2 pretokenization pattern from the assignment
        PAT = rf"{special_group}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"
        self.PAT = re.compile(PAT)
        
    def _find_EOF_boundaries(self, file: BinaryIO, desired_num_chunks: int, 
                            split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        Only reads mini_chunk_size at a time to save memory.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    # place boundary AFTER the special token so it is not split across chunks
                    chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def pretokenize_file_parallel(self, file_path_from_root_folder: str) -> None:
        """
        Pretokenize given file and return frequency table
        """
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, file_path_from_root_folder)

        with open(file_path, "rb") as f:

            boundaries = self._find_EOF_boundaries(f, NUM_PRETOKENIZING_PROCESSES, b"<|endoftext|>")

            # Run through every consecutive boundaries
            # Start pre-tokenization process for each

            chunks = [(boundaries[i], boundaries[i+1], file_path) for i in range(len(boundaries)-1)]
            
            if NUM_PRETOKENIZING_PROCESSES <= 1:
                _init_worker(self.PAT, self.special_tokens)
                results = [_process_chunk_worker(start, end, file_path) for start, end, file_path in chunks]
            else:
                with Pool(processes=NUM_PRETOKENIZING_PROCESSES,
                          initializer=_init_worker,
                          initargs=(self.PAT, self.special_tokens)) as pool:
                    # use starmap so each arg is a separate parameter
                    results = pool.starmap(_process_chunk_worker, chunks)

            for chunk_result in results:
                for token, count in chunk_result.items():
                    self.global_pretokenization_dict[token] = self.global_pretokenization_dict.get(token, 0) + count
            
            self._convert_to_bytes_dict()
            return self.pretokenization_dict_to_bytes
  
    def _convert_to_bytes_dict(self) -> None:
        """
        Convert the global pretokenization dict from str to tuple[bytes]
        """
        # Represent tokens as tuples of bytes. Single-byte tokens (regular
        # characters or punctuation) are stored as a sequence of single-byte
        # bytes objects (e.g., b'a', b'b', ...). Special tokens are stored as a
        # single-element tuple containing the entire special token bytestring
        # (e.g., (b"<|endoftext|>",)). This matches what BPEProcessor expects.

        for token, count in self.global_pretokenization_dict.items():
            if token in self.special_tokens:
                # special token -> one element: the full bytes of the token
                token_tuple = (token.encode("utf-8"),)
            else:
                token_bytes = token.encode("utf-8")
                # split into single-byte bytes objects
                token_tuple = tuple(bytes([b]) for b in token_bytes)

            # Store in new dictionary (accumulate counts)
            self.pretokenization_dict_to_bytes[token_tuple] = (
                self.pretokenization_dict_to_bytes.get(token_tuple, 0) + count
            )


