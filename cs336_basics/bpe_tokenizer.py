from collections import Counter, defaultdict
from pathlib import Path
import regex as re
from typing import Dict, List, Tuple
from cs336_basics.pretokenizer import PreTokenizer


class BPEProcessor:
    """
    Maintains a vocabulary dict id->bytes and a Counter of pair frequencies. Merges
    are applied greedily and pair frequencies are rebuilt after each merge.
    """

    def __init__(self, pretokenizer: PreTokenizer) -> None:
        # Initialize vocab: 0-255 are single-byte tokens
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.pretokenizer = pretokenizer

        # Add special tokens to vocab (ids 256, 257, ...)
        for i, t in enumerate(pretokenizer.special_tokens):
            self.vocab[256 + i] = t.encode("utf-8")

        self.vocab_index: int = max(self.vocab.keys())

        # Convert pretokenizer output (tuple-of-bytes -> count) into sequences of token ids
        # and maintain multiplicity via counts
        self.sequences: list[list[int]] = []
        self.sequence_counts: list[int] = []

        # Build a reverse mapping from byte-values to initial token ids (0-255) and special bytes
        byte_to_id: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        for i, t in enumerate(pretokenizer.special_tokens):
            byte_to_id[t.encode("utf-8")] = 256 + i

        for token_tuple, count in pretokenizer.pretokenization_dict_to_bytes.items():
            seq = []
            for elem in token_tuple:
                seq.append(byte_to_id[elem])
            
            self.sequences.append(seq)
            self.sequence_counts.append(count)

        # pair -> frequency (sum of counts across sequences)
        self.pair_freq: Counter[tuple[int, int]] = Counter()
        # pair -> set of (seq_idx, pos)
        self.pair_positions: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

        # merges as list of (bytes, bytes)
        self.merges: list[tuple[bytes, bytes]] = []

        # initialize pair frequencies
        for index, sequence in enumerate(self.sequences):
            count = self.sequence_counts[index]
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                self.pair_freq[pair] += count

    def _pair_key(self, pair: tuple[int, int]) -> tuple:
        """
        Return a deterministic lexicographic key for tiebreaking pairs.

        Convert each token id to its bytes value and take max of the tuple of ints 
        representing those bytes.
        """
        left_bytes = self.vocab[pair[0]]
        right_bytes = self.vocab[pair[1]]
        return (tuple(left_bytes), tuple(right_bytes))

    def run_bpe(self, num_merges: int) -> None:
        """Run BPE merges incrementally for num_merges iterations."""
        for _ in range(num_merges):
            if not self.pair_freq:
                break
            # find max frequency
            max_freq = max(self.pair_freq.values())
            # collect candidates with that frequency
            candidates = [p for p, f in self.pair_freq.items() if f == max_freq]
            # choose lexicographically greatest per spec
            best = max(candidates, key=self._pair_key)
            left_id, right_id = best

            # create new token id and vocab entry
            self.vocab_index += 1
            new_id = self.vocab_index
            left_bytes = self.vocab[left_id]
            right_bytes = self.vocab[right_id]
            self.vocab[new_id] = left_bytes + right_bytes
            # record merge as byte-pair
            self.merges.append((left_bytes, right_bytes))

            # Apply merges to all sequences (right-to-left) and then rebuild the
            # global pair frequency and position tables from scratch.
            for sequence in self.sequences:
                # collect all positions where the best pair occurs in this seq
                positions = [i for i in range(len(sequence) - 1) if sequence[i] == left_id and sequence[i + 1] == right_id]
                # process right-to-left so deletions don't affect earlier indices
                for pos in reversed(positions):
                    # replace at pos
                    sequence[pos] = new_id
                    del sequence[pos + 1]

            # After modifying sequences, rebuild pair_freq
            self.pair_freq = Counter()
            for index, sequence in enumerate(self.sequences):
                count = self.sequence_counts[index]
                for i in range(len(sequence) - 1):
                    pair = (sequence[i], sequence[i + 1])
                    self.pair_freq[pair] += count

    def encode(self, input: str | Path, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]) -> list[int]:
        """Encode a string into a sequence of BPE token IDs."""

        # 1. Take input string and apply PAT to get initial tokens as bytes with special tokens

        # Escape special tokens so they match literally
        escaped_specials = [re.escape(token) for token in self.pretokenizer.special_tokens]
        # Combine PAT and special tokens with alternation
        full_pattern = "|".join(escaped_specials + [self.pretokenizer.PAT.pattern])
        # Build the final regex
        combined = re.compile(full_pattern)

        matches = re.finditer(combined, input)
        
        # 2. Convert list of strings into byte arrays
        encode_bytearray: List[Tuple[bytes]] = []
        for token in matches:
            token_bytes = token.group().encode("utf-8")
            if not token_bytes.decode("utf-8") in self.pretokenizer.special_tokens:
                token_tuple = tuple(bytes([b]) for b in token_bytes)
            else:
                token_tuple = (token_bytes,)
            # Store in new dictionary
            encode_bytearray.append(token_tuple)
            
        print(encode_bytearray)
        
        # 3. Check each byte pair against vocab merges in order of merges
        for index, token_tuple in enumerate(encode_bytearray):
            token_tuple = list(token_tuple)  # easier to mutate
            i = 0
            # 4. Apply merges until no more merges can be applied
            while i < len(token_tuple) - 1:
                left, right = token_tuple[i], token_tuple[i + 1]
                if (left, right) in merges:
                    # Merge
                    merged = left + right
                    token_tuple[i] = merged
                    del token_tuple[i + 1]
                    
                    # Step back one index to check for a new merge with the previous token
                    if i > 0:
                        i -= 1
                else:
                    i += 1
            encode_bytearray[index] = tuple(token_tuple)
        print("Final token tuple:", encode_bytearray)

        #5. Replace all with vocab ids
        encoded_ids: List[int] = []
        reversed_dict = {value: key for key, value in vocab.items()}
        for token_tuple in encode_bytearray:
            for merged_bytes in token_tuple:
                encoded_ids.append(reversed_dict[merged_bytes])
        
        return encoded_ids
    
