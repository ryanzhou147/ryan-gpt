"""
Pre-tokenization module for text files.
Implements parallel pre-tokenization with special token handling.
"""
import os
import regex as re
from multiprocessing import Pool
from typing import BinaryIO, Tuple, Dict

NUM_PRETOKENIZING_PROCESSES = 4
SPECIAL_TOKENS = ["<|endoftext|>", "\r"]

class PreTokenizer:
    def __init__(self, special_tokens: list[str]) -> None:
        self.special_tokens: list[str] = special_tokens
        self.vocab: dict[int, bytes] = {
            **{x: bytes([x]) for x in range(256)}, # byte values
            **{256 + i: c.encode("utf-8") for i, c in enumerate(special_tokens)} # special tokens
        }
        self.vocab_index: int = 256 + len(special_tokens) # Store current index of dictionary
        self.PAT: str = self._initialize_PAT() # Pre-tokenization pattern
        self.global_pretokenization_dict: Dict[str, int] = {} # Global frequency table
        self.pretokenization_dict_to_bytes: Dict[tuple[bytes], int] = {} # Global frequency table in bytes
        
    def _initialize_PAT(self) -> str:
        special_patterns = [re.escape(c) for c in self.special_tokens]
        special_group = "|".join(special_patterns)
        PAT = rf"""{special_group}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+ | """
        return PAT
        
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
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _process_chunk(self, args: Tuple[int, int, str]) -> Dict[str, int]:
        """
        Worker function for processing each chunk
        Strips special tokens and runs pre-tokenization
        """
        start, end, filepath = args
        local_counts: Dict[str, int] = {}

        with open(filepath, "rb") as f:
            f.seek(start)
            chunk_data = f.read(end - start).decode("utf-8", errors="ignore")
            for token in self.special_tokens:
                chunk_data = re.sub(re.escape(token), "", chunk_data)
            # run pre-tokenization and count frequencies
            for pre_token in re.finditer(self.PAT, chunk_data):
                local_counts[pre_token.group()] = local_counts.get(pre_token.group(), 0) + 1
        print(local_counts)
        return local_counts

    def pretokenize_file_parallel(self, file_path_from_root_folder: str) -> None:
        """
        Pretokenize given file and return frequency table
        """
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, file_path_from_root_folder)

        with open(file_path, "rb") as f:

            boundaries = self._find_EOF_boundaries(f, NUM_PRETOKENIZING_PROCESSES, b"<|endoftext|>")
            print(boundaries)

            # Run through every consecutive boundaries
            # Start pre-tokenization process for each

            chunks = [(boundaries[i], boundaries[i+1], file_path) for i in range(len(boundaries)-1)]
            
            with Pool(processes=NUM_PRETOKENIZING_PROCESSES) as pool:
                results = pool.map(self._process_chunk, chunks)

            for chunk_result in results:
                for token, count in chunk_result.items():
                    self.global_pretokenization_dict[token] = self.global_pretokenization_dict.get(token, 0) + count
            
            self._convert_to_bytes_dict()
  
    def _convert_to_bytes_dict(self) -> None:
        """
        Convert the global pretokenization dict from str to tuple[bytes]
        """
        for token, count in self.global_pretokenization_dict.items():
            # Encode the string to bytes
            token_bytes = token.encode("utf-8")

            # Represent each byte as a bytes object of length 1
            token_tuple = tuple(bytes([b]) for b in token_bytes)

            # Store in new dictionary
            self.pretokenization_dict_to_bytes[token_tuple] = count

Pretokenizer = PreTokenizer(SPECIAL_TOKENS)

if __name__ == "__main__":
    # Expose the pattern at module-level for convenience when running as a script
    PAT = Pretokenizer.PAT
    print(PAT)
    print(Pretokenizer.vocab)
    print(Pretokenizer.vocab_index)
    Pretokenizer.pretokenize_file_parallel("cs336_basics/test.txt")
    print(Pretokenizer.global_pretokenization_dict)
    print(Pretokenizer.pretokenization_dict_to_bytes)

