# Given a file, we want to find all the places where it has <|endofbytes|>
# Then generate a list of it and chunk it accordingly to NUM_PROCESSES

import os
import regex as re
from multiprocessing import Pool
from typing import BinaryIO, Tuple, Dict

NUM_PROCESSES = 4

special_tokens = ["<|endoftext|>"]
special_patterns = [re.escape(c) for c in special_tokens]
special_group = "|".join(special_patterns)
PAT = rf"""{special_group}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+ | """

def find_EOF_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
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

# Worker function for pre-tokenization
def process_chunk(args: Tuple[int, int, str]) -> Dict[str, int]:
    start, end, filepath = args
    local_counts: Dict[str, int] = {}

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start).decode("utf-8", errors="ignore")
        # run pre-tokenization and count frequencies

        for pre_token in re.finditer(PAT, chunk_data):
            local_counts[pre_token.group()] = local_counts.get(pre_token.group(), 0) + 1
    print(local_counts)
    return local_counts

## Usage 
current_directory = os.getcwd()
input_path = os.path.join(current_directory, "cs336_basics/test.txt")

with open(input_path, "rb") as f:

    boundaries = find_EOF_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")
    print(boundaries)

    # Run through every consecutive boundaries
    # Start pre-tokenization process for each

    chunks = [(boundaries[i], boundaries[i+1], input_path) for i in range(len(boundaries)-1)]
    print(chunks)
    
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_chunk, chunks)

    global_pretokenization_dict: Dict[int, str] = {}
    for chunk_result in results:
         for token, count in chunk_result.items():
            global_pretokenization_dict[token] = global_pretokenization_dict.get(token, 0) + count
    
    print(global_pretokenization_dict)



