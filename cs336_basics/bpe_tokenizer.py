# UTF-8 is something like this: \x93\xe3\x81\xaa\xe3\x81\x97\xe3\x81\x82!' 0-255 of them
# Unicode string is like this: [0, 104, 101, 108, 108, 111, 33, 32, 12354, 12426, 12399, 12354, 33] a lot of them

import time
from .pretokenizer import PreTokenizer

NUM_MERGES = 10 # How many merges to make
SPECIAL_TOKENS = ["<|endoftext|>", "\r"]

class BPEProcessor:
    def __init__(self, pretokenizer: PreTokenizer) -> None:
        self.vocab: dict[int, bytes] = {
            **{x: bytes([x]) for x in range(256)}, # byte values
            **{256 + i: c.encode("utf-8") for i, c in enumerate(pretokenizer.special_tokens)} # special tokens
        }
        self.vocab_index: int = 256 + len(pretokenizer.special_tokens) - 1 # Store current index of dictionary
        self.tuple_pre_tokenized_file_content = pretokenizer.pretokenization_dict_to_bytes
        self.merges: list[tuple[bytes, bytes]] = [] # Keep track of merges
        self.bpe_counter: dict[tuple[bytes, bytes], int] = {} # BPE frequencies
    
    def run_bpe(self, num_merges: int) -> None:
        """Run BPE for NUM_MERGES iterations"""
        for _ in range(num_merges):
            self._get_bpe_pairs()
            greatest_bpe = self._get_greatest_bpe()
            self._update_vocab(greatest_bpe)
            self._update_token_tuples(greatest_bpe)
            self.bpe_counter.clear() # Clear for next iteration
        return

    def _get_bpe_pairs(self):
        """Iterate over all tuples of pre_tokenized_file_content and count BPE pairs frequencies"""
        # Iterate over token tuples and counts to avoid extra lookups
        for bytes_tuple, count in self.tuple_pre_tokenized_file_content.items():
            # Iterate over adjacent pairs in the token tuple
            for pair in zip(bytes_tuple, bytes_tuple[1:]):
                self.bpe_counter[pair] = self.bpe_counter.get(pair, 0) + count

    def _get_greatest_bpe(self) -> tuple[bytes, bytes]:
        """Find the BPE pair with the greatest frequency"""
        max_value = max(self.bpe_counter.values())
        # Find most lexographically signficant value
        greatest_keys = [k for k, v in self.bpe_counter.items() if v == max_value]
        # Keys may contain a mix of ints (merged token ids) and bytes (single-byte tokens).
        # Python cannot compare bytes and ints directly, so normalize keys to tuples of ints
        # for a stable lexicographic tiebreak.
        def _norm_token(t):
            if isinstance(t, int):
                return (t,)
            if isinstance(t, bytes):
                # bytes iterates to ints; this turns b'\n' -> (10,)
                return tuple(t)
            # fallback: try to convert to tuple of ints
            try:
                return tuple(int(x) for x in t)
            except Exception:
                return (0,)

        def _pair_key(pair):
            return _norm_token(pair[0]) + _norm_token(pair[1])

        return max(greatest_keys, key=_pair_key)

    def _update_vocab(self, greatest_bpe: tuple[bytes, bytes]) -> None:
        """Update the vocab and merges with the new BPE merge"""

        self.vocab_index += 1

        # Compute the byte sequences for the two tokens (they may be ints or bytes)
        def _token_bytes(t):
            if isinstance(t, int):
                return self.vocab[t]
            if isinstance(t, bytes):
                return t
            # fallback: try to join if iterable of ints/bytes
            try:
                return b"".join(x if isinstance(x, bytes) else bytes([x]) for x in t)
            except Exception:
                raise TypeError("Unsupported token type in BPE merge")

        left_b = _token_bytes(greatest_bpe[0])
        right_b = _token_bytes(greatest_bpe[1])

        # Append merged value (record merges as byte-pairs)
        self.merges.append((left_b, right_b))

        # Store new vocab entry with the concatenated bytes for this merged token
        self.vocab[self.vocab_index] = left_b + right_b

    def _update_token_tuples(self, greatest_bpe: tuple[bytes, bytes]) -> None:
        """Update the tuple_pre_tokenized_file_content with the new BPE merge"""
        # Replace all tuples with new vocab
        byte_tuple_file_content_with_merge: dict[tuple, int] = {}
        for token_tuple, count in self.tuple_pre_tokenized_file_content.items():
            new_token_list: list = []
            i = 0
            lt = len(token_tuple)
            while i < lt:
                # Check if next two tokens match the most frequent pair
                if i < lt - 1 and (token_tuple[i], token_tuple[i + 1]) == greatest_bpe:
                    # Replace with the new merged token id
                    new_token_list.append(self.vocab_index)
                    i += 2
                else:
                    new_token_list.append(token_tuple[i])
                    i += 1

            # Store back as a tuple and accumulate counts if collisions occur
            new_token_tuple = tuple(new_token_list)
            byte_tuple_file_content_with_merge[new_token_tuple] = (
                byte_tuple_file_content_with_merge.get(new_token_tuple, 0) + count
            )

        # Replace old dictionary with updated one
        self.tuple_pre_tokenized_file_content = byte_tuple_file_content_with_merge



if __name__ == "__main__":
    # Expose the pattern at module-level for convenience when running as a script

    start_time = time.time()
    Pretokenizer = PreTokenizer(SPECIAL_TOKENS)
    Pretokenizer.pretokenize_file_parallel("cs336_basics/test.txt")
    print(Pretokenizer.pretokenization_dict_to_bytes)

    bpe_processor = BPEProcessor(Pretokenizer)
    bpe_processor.run_bpe(NUM_MERGES)
    print(bpe_processor.tuple_pre_tokenized_file_content)
    print(bpe_processor.vocab)
    print(bpe_processor.vocab_index)
    end_time = time.time()
    print(f"Operation took {end_time - start_time} seconds.")

