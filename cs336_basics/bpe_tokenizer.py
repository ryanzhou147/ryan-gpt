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
        self.pretokenizer = pretokenizer
        self.vocab: dict[int, bytes] = {}
        self.sequences: list[list[int]] = []
        self.sequence_counts: list[int] = []
        self.pair_freq: Counter[tuple[int, int]] = Counter()
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab_index: int = -1
        self._special_token_bytes: set[bytes] = {t.encode("utf-8") for t in pretokenizer.special_tokens}
        self._combined_pattern: re.Pattern | None = None
        
        self._initialize_vocab()
        self._build_sequences()
        self._initialize_pair_frequencies()

    def _initialize_vocab(self) -> None:
        """Initialize vocab with single-byte tokens and special tokens."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, t in enumerate(self.pretokenizer.special_tokens):
            self.vocab[256 + i] = t.encode("utf-8")
        self.vocab_index = max(self.vocab.keys())

    def _build_sequences(self) -> None:
        """Convert pretokenizer output into sequences of token IDs."""
        byte_to_id = {bytes([i]): i for i in range(256)}
        for i, t in enumerate(self.pretokenizer.special_tokens):
            byte_to_id[t.encode("utf-8")] = 256 + i

        for token_tuple, count in self.pretokenizer.pretokenization_dict_to_bytes.items():
            seq = [byte_to_id[elem] for elem in token_tuple]
            self.sequences.append(seq)
            self.sequence_counts.append(count)

    def _initialize_pair_frequencies(self) -> None:
        """Compute initial pair frequencies from sequences."""
        self.pair_freq = Counter()
        for index, sequence in enumerate(self.sequences):
            count = self.sequence_counts[index]
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                self.pair_freq[pair] += count

    def _pair_key(self, pair: tuple[int, int]) -> tuple:
        """Return a lexicographic key for tiebreaking pairs."""
        return (tuple(self.vocab[pair[0]]), tuple(self.vocab[pair[1]]))

    def _select_best_pair(self) -> tuple[int, int]:
        """Select the most frequent pair with lexicographic tiebreaking."""
        if not self.pair_freq:
            return None
        max_freq = max(self.pair_freq.values())
        candidates = [p for p, f in self.pair_freq.items() if f == max_freq]
        return max(candidates, key=self._pair_key)

    def _apply_merge_to_sequences(self, left_id: int, right_id: int, new_id: int) -> None:
        """Apply a merge to all sequences (right-to-left to preserve indices)."""
        for sequence in self.sequences:
            positions = [i for i in range(len(sequence) - 1) 
                        if sequence[i] == left_id and sequence[i + 1] == right_id]
            for pos in reversed(positions):
                sequence[pos] = new_id
                del sequence[pos + 1]

    def _rebuild_pair_frequencies(self) -> None:
        """Rebuild pair frequencies from current sequences."""
        self.pair_freq = Counter()
        for index, sequence in enumerate(self.sequences):
            count = self.sequence_counts[index]
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                self.pair_freq[pair] += count

    def run_bpe(self, num_merges: int) -> None:
        """Run BPE merges incrementally for num_merges iterations."""
        for _ in range(num_merges):
            best_pair = self._select_best_pair()
            if best_pair is None:
                break
            
            left_id, right_id = best_pair
            self.vocab_index += 1
            new_id = self.vocab_index
            
            # Create new token and record merge
            left_bytes = self.vocab[left_id]
            right_bytes = self.vocab[right_id]
            self.vocab[new_id] = left_bytes + right_bytes
            self.merges.append((left_bytes, right_bytes))
            
            # Apply merge and rebuild frequencies
            self._apply_merge_to_sequences(left_id, right_id, new_id)
            self._rebuild_pair_frequencies()

    def _get_combined_pattern(self) -> re.Pattern:
        """Get or create the combined regex pattern (special tokens + PAT)."""
        if self._combined_pattern is None:
            escaped_specials = [re.escape(token) for token in self.pretokenizer.special_tokens]
            full_pattern = "|".join(escaped_specials + [self.pretokenizer.PAT.pattern])
            self._combined_pattern = re.compile(full_pattern)
        return self._combined_pattern

    def _build_reverse_vocab(self, vocab: Dict[int, bytes]) -> Dict[bytes, int]:
        """Build a reverse mapping from bytes to vocab IDs."""
        return {value: key for key, value in vocab.items()}

    def _tokenize_text(self, text: str, pattern: re.Pattern) -> List[Tuple[bytes]]:
        """Tokenize text and convert to tuple-of-bytes format.
        
        Special tokens are kept as single bytes tuples; regular tokens are split
        into individual bytes.
        """
        tokens: List[Tuple[bytes]] = []
        for match in pattern.finditer(text):
            token_bytes = match.group().encode("utf-8")
            if token_bytes in self._special_token_bytes:
                token_tuple = (token_bytes,)
            else:
                token_tuple = tuple(bytes([b]) for b in token_bytes)
            tokens.append(token_tuple)
        return tokens

    def _apply_merges(self, token_list: List[bytes], merges: List[Tuple[bytes, bytes]]) -> None:
        """Apply merges to a token list in-place, skipping special tokens."""
        i = 0
        while i < len(token_list) - 1:
            left, right = token_list[i], token_list[i + 1]
            
            # Skip if either is a special token
            if left in self._special_token_bytes or right in self._special_token_bytes:
                i += 1
                continue
            
            if (left, right) in merges:
                token_list[i] = left + right
                del token_list[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

    def _tokens_to_ids(self, tokens: List[Tuple[bytes]], reversed_vocab: Dict[bytes, int]) -> List[int]:
        """Convert tokenized tuples to vocabulary IDs."""
        encoded_ids: List[int] = []
        for token_tuple in tokens:
            for byte_sequence in token_tuple:
                encoded_ids.append(reversed_vocab[byte_sequence])
        return encoded_ids

    def encode(self, input: str | Path, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]) -> list[int]:
        """
        Encode a string or file into BPE token IDs.
        
        Automatically detects file paths and streams large files efficiently.
        """
        if isinstance(input, (str, Path)):
            input_path = Path(input)
            if input_path.exists() and input_path.is_file():
                return self._encode_file(input_path, vocab, merges)
        
        return self._encode_text(str(input), vocab, merges)

    def _encode_text(self, text: str, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]) -> list[int]:
        """Encode raw text string."""
        pattern = self._get_combined_pattern()
        tokens = self._tokenize_text(text, pattern)
        
        for i, token_tuple in enumerate(tokens):
            token_list = list(token_tuple)
            self._apply_merges(token_list, merges)
            tokens[i] = tuple(token_list)
        
        reversed_vocab = self._build_reverse_vocab(vocab)
        return self._tokens_to_ids(tokens, reversed_vocab)

    def _encode_file(self, filepath: Path, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]) -> list[int]:
        """Encode a file by streaming in chunks."""
        READ_BLOCK = 1024 * 1024  # 1 MiB
        TAIL_CHARS = 4096

        pattern = self._get_combined_pattern()
        reversed_vocab = self._build_reverse_vocab(vocab)
        encoded_ids: List[int] = []
        carry_bytes = b""

        with open(filepath, "rb") as f:
            while True:
                part = f.read(READ_BLOCK)
                if not part:
                    break
                carry_bytes += part

                # Try to decode as much as possible
                try:
                    decoded = carry_bytes.decode("utf-8")
                    carry_bytes = b""
                except UnicodeDecodeError as e:
                    decoded = carry_bytes[:e.start].decode("utf-8", errors="ignore")
                    carry_bytes = carry_bytes[e.start:]

                # Process safe portion, keep tail for next iteration
                if len(decoded) > TAIL_CHARS:
                    safe_portion = decoded[:-TAIL_CHARS]
                    encoded_ids.extend(self._encode_and_convert(safe_portion, pattern, merges, reversed_vocab))
                    remaining_fragment = decoded[-TAIL_CHARS:]
                    carry_bytes = remaining_fragment.encode("utf-8") + carry_bytes
                else:
                    carry_bytes = decoded.encode("utf-8") + carry_bytes

        # Flush remaining bytes
        if carry_bytes:
            final_decoded = carry_bytes.decode("utf-8", errors="ignore")
            encoded_ids.extend(self._encode_and_convert(final_decoded, pattern, merges, reversed_vocab))

        return encoded_ids

    def _encode_and_convert(self, text: str, pattern: re.Pattern, merges: List[Tuple[bytes, bytes]], 
                           reversed_vocab: Dict[bytes, int]) -> List[int]:
        """Tokenize text, apply merges, and convert to IDs."""
        tokens_as_bytearray = self._tokenize_text(text, pattern)
        
        for i, token_tuple in enumerate(tokens_as_bytearray):
            token_list = list(token_tuple)
            self._apply_merges(token_list, merges)
            tokens_as_bytearray[i] = tuple(token_list)

        return self._tokens_to_ids(tokens_as_bytearray, reversed_vocab)
    
