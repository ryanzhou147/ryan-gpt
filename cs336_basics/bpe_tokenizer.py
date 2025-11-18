from collections import Counter, defaultdict
from pathlib import Path
import json
import regex as re
from typing import Dict, List, Tuple, Iterable, Iterator
from cs336_basics.pretokenizer import PreTokenizer


class BPEProcessor:
    """
    BPE tokenizer that can encode/decode
    Supports special tokens and streaming for large files.
    """

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]] = None, 
                 special_tokens: List[str] | None = None, pretokenizer: "PreTokenizer | None" = None) -> None:
        """
        Initialize tokenizer from vocab, merges, and special tokens.
        """
        self.vocab = vocab
        self.merges = merges or []
        self.special_tokens = special_tokens or []
        self._special_token_bytes = {t.encode("utf-8") for t in self.special_tokens}
        self._reverse_vocab = {v: k for k, v in self.vocab.items()}  # Build reverse vocab immediately
        
        # Generate merges for special tokens (to reconstruct them byte-by-byte from individual bytes)
        special_token_merges = []
        for special_token_str in self.special_tokens:
            special_token_bytes = special_token_str.encode("utf-8")
            # Create merges to reconstruct this special token byte by byte
            # E.g., for b'<|e', we need: b'<' + b'|' -> b'<|', then b'<|' + b'e' -> b'<|e', etc.
            for i in range(1, len(special_token_bytes)):
                left = special_token_bytes[:i]
                right = bytes([special_token_bytes[i]])  # Single byte!
                special_token_merges.append((left, right))
        
        # Add special token merges at the BEGINNING with highest priority (lowest rank)
        self.merges = special_token_merges + self.merges
        
        # Build merge ranks for efficient priority-based lookup
        self._merge_ranks = {merge: i for i, merge in enumerate(self.merges)}
        self._pat = self._build_pat()
        
        # Training-related attributes
        self.pretokenizer = pretokenizer
        self.sequences: list[list[int]] = []
        self.sequence_counts: list[int] = []
        self.pair_freq: Counter[tuple[int, int]] = Counter()
        self.vocab_index: int = -1
        
        self._build_sequences_for_training()
        self._build_pair_frequencies()

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, 
                   special_tokens: List[str] | None = None) -> "BPEProcessor":
        """Load tokenizer from serialized vocab and merges files.
        
        Args:
            vocab_filepath: path to JSON vocab file (maps int -> bytes as list)
            merges_filepath: path to merges file (one merge per line)
            special_tokens: optional list of special token strings
        
        Returns:
            BPEProcessor instance
        """

        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_json = json.load(vf)
            vocab = {int(k): bytes(v) for k, v in vocab_json.items()}
        
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges = []
            for line in mf:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append((left.encode("utf-8"), right.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """Encode text string into token IDs."""
        tokens = self._tokenize_text(text)
        self._apply_merges(tokens)
        return [self._reverse_vocab[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings"""

        for text in iterable:
            tokens = self._tokenize_text(text)  # flat list of bytes
            self._apply_merges(tokens)          # merge across the whole sequence
            for token in tokens:
                yield self._reverse_vocab[token]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into text."""
        byte_sequence = b""
        for token_id in ids:
            byte_sequence += self.vocab[token_id]

        return byte_sequence.decode("utf-8", errors="replace")

    def _build_pat(self) -> re.Pattern:
        """Build regex pattern for tokenization (special tokens + PAT).
        
        Special tokens must be matched with absolute priority to prevent greedy space matching
        from consuming spaces before them. We match spaces separately BEFORE punctuation.
        """
        # Sort special tokens by length (longest first) to match greedily
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        escaped_specials = [re.escape(token) for token in sorted_specials]
        
        # GPT-2 canonical PAT pattern, with space pattern BEFORE punctuation
        # This prevents ` ?[^\s...]+ ` from greedily consuming space before special tokens
        pat_str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|\s+(?!\S)|\s+| ?[^\s\p{L}\p{N}]+"
        
        # Build pattern with special tokens first (they have absolute priority)
        full_pattern = "|".join(escaped_specials + [pat_str]) if escaped_specials else pat_str
        return re.compile(full_pattern)

    def _tokenize_text(self, text: str) -> List[bytes]:
        """Tokenize text into bytes format using longest-match-first in vocabulary.
        
        Special tokens are kept as single tokens. Regular tokens are matched against
        the vocabulary to find the longest matching sequence, using merges to guide
        tokenization for sequences not in vocab.
        """
        tokens: List[bytes] = []
        for match in self._pat.finditer(text):
            token_str = match.group()
            token_bytes = token_str.encode("utf-8")
            
            # If it's a special token, keep it whole
            if token_bytes in self._special_token_bytes:
                tokens.append(token_bytes)
            # If the whole token is in vocab, keep it whole
            elif token_bytes in self._reverse_vocab:
                tokens.append(token_bytes)
            else:
                # Token not in vocab - break into individual bytes for BPE merging
                for b in token_bytes:
                    tokens.append(bytes([b]))

        return tokens

    def _apply_merges(self, token_list: List[bytes]) -> None:
        """Apply merges to token list in-place, skipping special tokens.
        
        Uses a greedy algorithm: scans left-to-right, applying the highest-priority
        merge found at each position, then backtracks to check earlier pairs.
        """
        if not self._merge_ranks:
            return
        
        i = 0
        while i < len(token_list) - 1:
            left, right = token_list[i], token_list[i + 1]
            
            # Skip if either is a special token
            if left in self._special_token_bytes or right in self._special_token_bytes:
                i += 1
                continue
            
            # Check if this pair is a merge (O(1) lookup via dict)
            if (left, right) in self._merge_ranks:
                token_list[i] = left + right
                del token_list[i + 1]
                # Backtrack to check if the new token can merge with the previous one
                if i > 0:
                    i -= 1
            else:
                i += 1

    def _tokens_to_ids(self, tokens: List[bytes]) -> List[int]:
        """Convert token tuples to vocabulary IDs."""
        return [self._reverse_vocab[token] for token in tokens]

    def _build_sequences_for_training(self) -> None:
        """Convert pretokenizer output into sequences for training."""
        if self.pretokenizer is None:
            return
        
        # Initialize vocab for training
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, t in enumerate(self.special_tokens):
            self.vocab[256 + i] = t.encode("utf-8")
        self.vocab_index = max(self.vocab.keys())
        
        byte_to_id = {bytes([i]): i for i in range(256)}
        for i, t in enumerate(self.special_tokens):
            byte_to_id[t.encode("utf-8")] = 256 + i

        for token_tuple, count in self.pretokenizer.pretokenization_dict_to_bytes.items():
            seq = [byte_to_id[elem] for elem in token_tuple]
            self.sequences.append(seq)
            self.sequence_counts.append(count)

    def _pair_key(self, pair: tuple[int, int]) -> tuple:
        """Return lexicographic key for tiebreaking."""
        return (tuple(self.vocab[pair[0]]), tuple(self.vocab[pair[1]]))

    def _select_best_pair(self) -> tuple[int, int]:
        """Select best pair for merging."""
        if not self.pair_freq:
            return None
        max_freq = max(self.pair_freq.values())
        candidates = [p for p, f in self.pair_freq.items() if f == max_freq]
        return max(candidates, key=self._pair_key)

    def _apply_merge_to_sequences(self, left_id: int, right_id: int, new_id: int) -> None:
        """Apply merge to all training sequences."""
        if not self.sequences:
            return
        
        for sequence in self.sequences:
            positions = [i for i in range(len(sequence) - 1) 
                        if sequence[i] == left_id and sequence[i + 1] == right_id]
            for pos in reversed(positions):
                sequence[pos] = new_id
                del sequence[pos + 1]

    def _build_pair_frequencies(self) -> None:
        """Build pair frequencies after merge."""
        if not self.sequences:
            return
        
        self.pair_freq = Counter()
        for index, sequence in enumerate(self.sequences):
            count = self.sequence_counts[index]
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                self.pair_freq[pair] += count

    def run_bpe(self, num_merges: int) -> None:
        """Run BPE training for num_merges iterations."""
        self._build_pair_frequencies()
        for _ in range(num_merges):
            best_pair = self._select_best_pair()
            if best_pair is None:
                break
            
            left_id, right_id = best_pair
            self.vocab_index += 1
            new_id = self.vocab_index
            
            left_bytes = self.vocab[left_id]
            right_bytes = self.vocab[right_id]
            self.vocab[new_id] = left_bytes + right_bytes
            self.merges.append((left_bytes, right_bytes))
            self._merge_ranks = {merge: i for i, merge in enumerate(self.merges)}  # Rebuild ranks
            
            self._apply_merge_to_sequences(left_id, right_id, new_id)
            self._build_pair_frequencies()

        self._reverse_vocab = {v: k for k, v in self.vocab.items()}