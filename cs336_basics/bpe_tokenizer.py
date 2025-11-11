from collections import Counter, defaultdict
from pretokenizer import PreTokenizer


class BPEProcessor:
    """Incremental BPE processor that tracks pair frequencies and occurrence locations.

    Represents the corpus as a list of token sequences (lists of int ids).
    Maintains a vocabulary dict id->bytes, a Counter of pair frequencies, and a mapping
    from pair -> set of (sequence_index, position) where the pair occurs. When merging a pair,
    only local neighborhoods are updated which keeps the training fast and deterministic.
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
        self.byte_to_id: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        for i, t in enumerate(pretokenizer.special_tokens):
            self.byte_to_id[t.encode("utf-8")] = 256 + i

        for token_tuple, cnt in pretokenizer.pretokenization_dict_to_bytes.items():
            seq = []
            for elem in token_tuple:
                # elem is bytes (either single-byte b'a' or multi-byte special token)
                if elem in self.byte_to_id:
                    seq.append(self.byte_to_id[elem])
                else:
                    # If it's a multi-byte bytes not in map, add it as a new special token
                    self.vocab_index += 1
                    self.vocab[self.vocab_index] = elem
                    self.byte_to_id[elem] = self.vocab_index
                    seq.append(self.vocab_index)
            self.sequences.append(seq)
            self.sequence_counts.append(cnt)

        # pair -> frequency (sum of counts across sequences)
        self.pair_freq: Counter[tuple[int, int]] = Counter()
        # pair -> set of (seq_idx, pos)
        self.pair_positions: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

        # merges as list of (bytes, bytes)
        self.merges: list[tuple[bytes, bytes]] = []

        # initialize pair frequencies and positions
        for si, seq in enumerate(self.sequences):
            cnt = self.sequence_counts[si]
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                self.pair_freq[pair] += cnt
                self.pair_positions[pair].add((si, i))

    def _pair_key(self, pair: tuple[int, int]) -> tuple:
        """Return a deterministic lexicographic key for tiebreaking pairs.

        We convert each token id to its bytes value and use the tuple of ints representing
        those bytes for lexicographic comparison, matching the assignment spec.
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

            # get occurrences (make a copy since we'll mutate structures)
            occs = list(self.pair_positions.get(best, []))

            # For each occurrence, attempt to merge if still valid
            for (si, pos) in occs:
                seq = self.sequences[si]
                # validate that pair still exists at position
                if pos >= len(seq) - 1:
                    continue
                if seq[pos] != left_id or seq[pos + 1] != right_id:
                    continue

                # perform replacement: seq[pos] = new_id; remove seq[pos+1]
                seq[pos] = new_id
                del seq[pos + 1]

                cnt = self.sequence_counts[si]

                # update affected pairs around the replaced position
                # positions to consider: pos-1 (pair left of left_id), pos (new pair right of new_id)
                # decrement counts for old pairs that included the removed tokens
                # left neighbor old pair: (seq[pos-1], left_id) before replacement
                if pos - 1 >= 0:
                    old_left_pair = (seq[pos - 1], left_id)
                    if old_left_pair in self.pair_freq:
                        self.pair_freq[old_left_pair] -= cnt
                        if self.pair_freq[old_left_pair] <= 0:
                            del self.pair_freq[old_left_pair]
                    self.pair_positions.get(old_left_pair, set()).discard((si, pos - 1))

                # right neighbor old pair: (right_id, seq[pos+1]) before replacement
                # Note: after deletion, seq[pos] is new_id; the old right neighbor was at pos+1
                if pos < len(seq):
                    old_right_pair = (right_id, seq[pos])
                    if old_right_pair in self.pair_freq:
                        self.pair_freq[old_right_pair] -= cnt
                        if self.pair_freq[old_right_pair] <= 0:
                            del self.pair_freq[old_right_pair]
                    self.pair_positions.get(old_right_pair, set()).discard((si, pos + 1))

                # remove the entries for the merged pair at this position
                if best in self.pair_freq:
                    self.pair_freq[best] -= cnt
                    if self.pair_freq[best] <= 0:
                        del self.pair_freq[best]
                self.pair_positions.get(best, set()).discard((si, pos))

                # Now add new pairs formed by the new token
                # left-new pair: (seq[pos-1], new_id)
                if pos - 1 >= 0:
                    new_left = (seq[pos - 1], new_id)
                    self.pair_freq[new_left] += cnt
                    self.pair_positions[new_left].add((si, pos - 1))

                # new-right pair: (new_id, seq[pos+1]) if exists
                if pos + 1 < len(seq):
                    new_right = (new_id, seq[pos + 1])
                    self.pair_freq[new_right] += cnt
                    self.pair_positions[new_right].add((si, pos))

            # finally, clear any stale empty position sets for best
            if best in self.pair_positions and not self.pair_positions[best]:
                del self.pair_positions[best]

        return

