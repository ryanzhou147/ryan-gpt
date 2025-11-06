# UTF-8 is something like this: \x93\xe3\x81\xaa\xe3\x81\x97\xe3\x81\x82!' 0-255 of them
# Unicode string is like this: [0, 104, 101, 108, 108, 111, 33, 32, 12354, 12426, 12399, 12354, 33] a lot of them

import regex as re
import os
from collections import defaultdict

# Opens and extracts text from training data
current_directory = os.getcwd()
print(current_directory)
input_path = os.path.join(current_directory, "cs336_basics/test.txt")
try:
    with open(input_path, "r") as file:
        file_content = file.read()
        print(file_content)
except FileNotFoundError:
    print(f"File not found: {input_path}")

# Create vocab dictionary 0-255 bytes and special tokens
special_tokens = ["<|endoftext|>"]
vocab: dict[int, bytes] = {
    **{x: bytes([x]) for x in range(256)}, # byte values
    **{256 + i: c.encode("utf-8") for i, c in enumerate(special_tokens)} # special tokens
}
print(vocab)

# Pre-tokenization of GPT-2
# Adds escape character to regex recognized patterns like "|" in "<|endoftext|>"
special_patterns = [re.escape(c) for c in special_tokens]
special_group = "|".join(special_patterns)

PAT = rf"""{special_group}|'(?:[sdmt]|ll|ve|re)| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+ | """
re.finditer(PAT, file_content)
print(re.findall(PAT, file_content))

merges: list[tuple[int, int], int] = {}
bpe: dict[tuple[str, str], int] = {}
indices = list(map(int, file_content.encode("utf-8")))

#
num_merges = 10

for i in range(num_merges):

    counts = defaultdict(int)
    
    for (a, b) in zip(file_content, file_content[1:]):
        key = (a, b)
        if key in bpe:
            bpe[key] += 1
        else:
            bpe[key] = 1
# print(bpe)
# print(vocab)

# vocab_size: int
# A positive integer that defines the maximum final vocabulary size (including the
# initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
# special_tokens: list[str] 
# A list of strings to add to the vocabulary. These special tokens do not
# otherwise affect BPE training.