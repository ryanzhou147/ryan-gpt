from collections import Counter, defaultdict
from cs336_basics.pretokenizer import PreTokenizer


class BPEProcessor:
    """
    Maintains a vocabulary dict id->bytes and a Counter of pair frequencies. Merges
    are applied greedily and pair frequencies are rebuilt after each merge.
    """

    def __init__(self, pretokenizer: PreTokenizer) -> None:
        # Initialize vocab: 0-255 are single-byte tokens
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Add special tokens to vocab (ids 256, 257, ...)
        for i, t in enumerate(pretokenizer.special_tokens):
            self.vocab[256 + i] = t.encode("utf-8")

        self.vocab_index: int = max(self.vocab.keys())

        # Convert pretokenizer output (tuple-of-bytes -> count) into sequences of token ids
        # and maintain multiplicity via counts
        # pretokenizer.pretokenization_dict_to_bytes: Dict[tuple[bytes], int]
        self.sequences: list[list[int]] = []
        self.sequence_counts: list[int] = []

        # Build a reverse mapping from byte-values to initial token ids (0-255) and special bytes
        byte_to_id: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        for i, t in enumerate(pretokenizer.special_tokens):
            byte_to_id[t.encode("utf-8")] = 256 + i

        for token_tuple, cnt in pretokenizer.pretokenization_dict_to_bytes.items():
            seq = []
            for elem in token_tuple:
                seq.append(byte_to_id[elem])

            self.sequences.append(seq)
            self.sequence_counts.append(cnt)

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