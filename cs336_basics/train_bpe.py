# UTF-8 is something like this: \x93\xe3\x81\xaa\xe3\x81\x97\xe3\x81\x82!' 0-255 of them
# Unicode string is like this: [0, 104, 101, 108, 108, 111, 33, 32, 12354, 12426, 12399, 12354, 33] a lot of them

import regex as re
import os
# from file_content_split import find_EOF_boundaries
from typing import Dict

# Opens and extracts text from training data
current_directory = os.getcwd()
input_path = os.path.join(current_directory, "cs336_basics/test.txt")
try:
    with open(input_path, "r") as file:
        file_content = file.read()
except FileNotFoundError:
    print(f"File not found: {input_path}")

# Create vocab dictionary 0-255 bytes and special tokens
special_tokens = ["<|endoftext|>"]
vocab: dict[int, bytes] = {
    **{x: bytes([x]) for x in range(256)}, # byte values
    **{256 + i: c.encode("utf-8") for i, c in enumerate(special_tokens)} # special tokens
}
vocab_index = 256 + len(special_tokens) # Store current index of dictionary

# TEMP: strip all special tokens
# TODO: run parallel implementation for each chunk
for token in special_tokens:
    file_content = re.sub(re.escape(token), "", file_content)
print(file_content)

# Pre-tokenization of GPT-2
# Adds escape character to regex recognized patterns like "|" in "<|endoftext|>"
special_patterns = [re.escape(c) for c in special_tokens]
special_group = "|".join(special_patterns)
PAT = rf"""{special_group}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+ | """

# Pre-tokenized pattern with frequency
pre_tokenized_file_content: Dict[str, int] = {}
for pre_token in re.finditer(PAT, file_content):
    pre_tokenized_file_content[pre_token.group()] = pre_tokenized_file_content.get(pre_token.group(), 0) + 1

print(pre_tokenized_file_content)

# Pre-tokenized pattern split into dict[tuple[bytes], int]
tuple_pre_tokenized_file_content = {}

for token, count in pre_tokenized_file_content.items():
    # Encode the string to bytes
    token_bytes = token.encode("utf-8")

    # Represent each byte as a bytes object of length 1
    token_tuple = tuple(bytes([b]) for b in token_bytes)

    # Store in new dictionary
    tuple_pre_tokenized_file_content[token_tuple] = count

print(tuple_pre_tokenized_file_content)

# BPE Processing
merges: list[tuple[bytes, bytes]] = [] # Keep track of merges

NUM_MERGES = 5 # How many merges to make

for i in range(NUM_MERGES):

    # Iterate over all tuples of pre_tokenized_file_content
    bpe = {} # Dict of bpe, frequencies
    for bytes_tuple in tuple_pre_tokenized_file_content.keys():

        # Iterate over tuple of bytes
        for pair in zip(bytes_tuple, bytes_tuple[1:]):
            bpe[pair] = bpe.get(pair, 0) + tuple_pre_tokenized_file_content[bytes_tuple] 

    # Find greatest frequency value
    max_value = max(bpe.values())
    # Find most lexographically signficant value
    greatest_keys = [k for k, v in bpe.items() if v == max_value]
    print(max(greatest_keys))

    # Append merged value
    merges.append(max(greatest_keys))

    # Store new vocab entry with most frequent BPE
    vocab[vocab_index] = max(greatest_keys)

    # Replace all tuples with new vocab
    byte_tuple_file_content_with_merge = {}
    for token_tuple, count in tuple_pre_tokenized_file_content.items():
        new_token_list = []
        i = 0
        while i < len(token_tuple):
            # Check if next two bytes match the most frequent pair
            if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == max(greatest_keys):
                # Replace with the new merged token
                new_token_list.append(vocab_index)
                i += 2
            else:
                new_token_list.append(token_tuple[i])
                i += 1

        # Store back as a tuple
        new_token_tuple = tuple(new_token_list)
        byte_tuple_file_content_with_merge[new_token_tuple] = count

    # Replace old dictionary with updated one
    tuple_pre_tokenized_file_content = byte_tuple_file_content_with_merge
    print(tuple_pre_tokenized_file_content)

    # Update vocab index
    vocab_index += 1
    print(vocab_index)

print(vocab)
print(merges)