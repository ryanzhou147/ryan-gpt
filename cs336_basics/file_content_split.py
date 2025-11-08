# Given a file, we want to find all the places where it has <|endofbytes|>
# Then generate a list of it and chunk it accordingly to NUM_CHUCKS
from typing import BinaryIO
import os

def find_EOF_boundaries(file: BinaryIO, num_chunks: int, token: bytes):
    """
    Find all positions in the binary file where `token` occurs,
    and return a list of chunk boundaries that split the file into
    roughly `num_chunks` parts, aligned to those token positions.
    """
    assert isinstance(token, bytes), "token must be in bytes"

    # Get file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    print(file_size)

    file_content = file.read()

current_directory = os.getcwd()
input_path = os.path.join(current_directory, "cs336_basics/test.txt")

with open(input_path, "rb") as f:
    find_EOF_boundaries(f, 4, b"<|endoftext|>")