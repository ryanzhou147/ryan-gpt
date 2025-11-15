import regex as re
from typing import Dict, List, Tuple


def bpe_encode(input: str, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]) -> List[int]:
    
    # 1. Take input string and apply PAT to get initial tokens as bytes
    PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    matches = re.finditer(PAT, input)
    
    # 2. Convert list of strings into byte arrays
    encode_bytearray: List[Tuple[bytes]] = []
    for token in matches:
        token_bytes = token.group().encode("utf-8")
        token_tuple = tuple(bytes([b]) for b in token_bytes)

        # Store in new dictionary
        encode_bytearray.append(token_tuple)
        
    print(encode_bytearray)

    # 3. Check each byte pair against vocab merges in order of merges
    for index, token_tuple in enumerate(encode_bytearray):
        count = 0
    # 4. Apply merges until no more merges can be applied
        while count < len(token_tuple) - 1:
            left_byte, right_byte = token_tuple[count], token_tuple[count + 1]
            if (left_byte, right_byte) in merges:
                # Merge the two bytes
                print(count, left_byte, right_byte)
                token_tuple = token_tuple[:count] + (left_byte + right_byte,) + token_tuple[count + 2:]
                print("After merge:", token_tuple)
            else:
                count += 1
        encode_bytearray[index] = token_tuple

    print("Final token tuple:", encode_bytearray)

    #5. Replace all with vocab ids
    encoded_ids: List[int] = []
    reversed_dict = {value: key for key, value in vocab.items()}
    for token_tuple in encode_bytearray:
        for merged_bytes in token_tuple:
            encoded_ids.append(reversed_dict[merged_bytes])
    
    print(encoded_ids)

if __name__ == "__main__":
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab[256] = b'el'
    vocab[257] = b'lo'
    vocab[258] = b'wo'
    bpe_encode("Hello hello hello world", vocab, [(b'e', b'l'), (b'w', b'o'), (b'l', b'o')])