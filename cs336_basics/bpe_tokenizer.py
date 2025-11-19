from collections import Counter
from pathlib import Path
import json
import regex as re
from typing import Dict, List, Tuple, Iterable, Iterator, Union
from functools import lru_cache
from cs336_basics.pretokenizer import PreTokenizer


def _gpt2_bytes_to_unicode() -> Dict[int, str]:
    """Map bytes to GPT-2 printable unicode representations."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


class BPEProcessor:
    """
    BPE tokenizer that can encode/decode.
    Supports special tokens and streaming for large files.
    """

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]] = None, 
                 special_tokens: List[str] | None = None, pretokenizer: "PreTokenizer | None" = None) -> None:
        """Initialize tokenizer from vocab, merges, and special tokens."""
        self.vocab = vocab
        self.merges = merges or []
        self.merges_dict = {merge: i for i, merge in enumerate(self.merges)}
        self.vocab_size = len(self.vocab)
        
        # Add special tokens to vocab if not already present
        special_tokens = special_tokens or []
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        
        # Build bidirectional vocab mappings
        self.id_to_bytes = {k: v for k, v in self.vocab.items()}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        # Build special token ID map (sorted by length for greedy matching)
        self.special_tokens_to_id = {}
        for token in sorted([token.encode("utf-8") for token in special_tokens], key=len, reverse=True):
            self.special_tokens_to_id[token] = self.bytes_to_id[token]
        
        self.pretokenizer = pretokenizer
        self.sequences: list[list[int]] = []
        self.sequence_counts: list[int] = []
        self.pair_freq: Counter[tuple[int, int]] = Counter()
        self.vocab_index: int = max(self.vocab.keys()) if self.vocab else -1
        
        self._build_sequences_for_training()
        self._build_pair_frequencies()

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, 
                   special_tokens: List[str] | None = None) -> "BPEProcessor":
        """Load tokenizer from serialized vocab and merges files."""
        gpt2_bytes_to_unicode = _gpt2_bytes_to_unicode()
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode.items()}

        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_json = json.load(vf)
            vocab = {
                token_id: bytes([gpt2_byte_decoder[c] for c in token_str])
                for token_str, token_id in vocab_json.items()
            }
        
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges = []
            for line in mf:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                merge_left, merge_right = parts
                left = bytes([gpt2_byte_decoder[c] for c in merge_left])
                right = bytes([gpt2_byte_decoder[c] for c in merge_right])
                merges.append((left, right))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """Encode text string into token IDs."""
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings."""
        for text in iterable:
            for token_id in self._encode_single(text):
                yield token_id
    
    def _encode_single(self, text: str) -> List[int]:
        """Encode a single text string into token IDs."""
        # First, handle special tokens by splitting text around them
        results = self._tokenize_special_tokens(text)
        
        # Then tokenize each regular text segment
        token_ids = []
        for elem in results:
            if isinstance(elem, str):
                # Regular text - tokenize with PAT and merges
                token_ids.extend(self._tokenize_normal_text(elem))
            elif isinstance(elem, int):
                # Already a special token ID
                token_ids.append(elem)
            else:
                raise TypeError(f"Unexpected type {type(elem)} in results.")
        
        return token_ids
    
    def _tokenize_special_tokens(self, text: str) -> List[Union[str, int]]:
        """Split text by special tokens, returning a mix of strings and special token IDs."""
        if not self.special_tokens_to_id:
            return [text]
        
        # Build regex pattern for all special tokens
        patterns = []
        replacements = {}
        for special_token in self.special_tokens_to_id.keys():
            token_str = special_token.decode("utf-8")
            patterns.append(re.escape(token_str))
            replacements[token_str] = self.special_tokens_to_id[special_token]
        
        pattern = re.compile("|".join(patterns))
        results = []
        last_end = 0
        
        for match in pattern.finditer(text):
            # Add regular text before the match
            if match.start() > last_end:
                results.append(text[last_end:match.start()])
            
            # Add special token ID
            matched_text = match.group()
            results.append(replacements[matched_text])
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            results.append(text[last_end:])
        
        return results
    
    def _tokenize_normal_text(self, text: str) -> List[int]:
        """Tokenize normal text (no special tokens) into token IDs using PAT + BPE.
        
        Tokenizes each PAT word independently using BPE merges, then concatenates the tokens.
        """
        # PAT regex from GPT-2 with proper ordering for greedy matching
        # The pattern ` ?[^\s\p{L}\p{N}]+` must come before `\s+` to match space + punctuation as one token
        pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        words = pat.findall(text)
        
        # Tokenize each word independently using BPE
        token_ids = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            token_ids.extend(word_tokens)
        
        return token_ids
    
    @lru_cache(maxsize=100000)
    def _tokenize_word(self, word: str) -> tuple[int, ...]:
        """Tokenize a single word using BPE merges.
        
        Applies BPE merges to the UTF-8 bytes of the word.
        Cached for performance.
        """
        word_bytes = word.encode("utf-8")
        
        # First, try to match the full word directly in vocab
        if word_bytes in self.bytes_to_id:
            return (self.bytes_to_id[word_bytes],)
        
        # Apply BPE merges starting from individual bytes
        word_bytes_list = [bytes([b]) for b in word_bytes]
        
        # Apply merges greedily by rank
        while True:
            # Find the best (earliest/lowest rank) merge available
            best_rank = float("inf")
            best_idx = -1
            
            for i in range(len(word_bytes_list) - 1):
                pair = (word_bytes_list[i], word_bytes_list[i + 1])
                if pair in self.merges_dict:
                    rank = self.merges_dict[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
            
            if best_idx == -1:
                break
            
            # Apply the best merge
            left = word_bytes_list[best_idx]
            right = word_bytes_list[best_idx + 1]
            word_bytes_list[best_idx] = left + right
            del word_bytes_list[best_idx + 1]
        
        # Convert bytes to token IDs
        token_ids = tuple(self.bytes_to_id[b] for b in word_bytes_list)
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into text."""
        byte_sequence = b""
        for token_id in ids:
            byte_sequence += self.id_to_bytes[token_id]
        return byte_sequence.decode("utf-8", errors="replace")

    def _build_sequences_for_training(self) -> None:
        """Convert pretokenizer output into sequences for training."""
        if self.pretokenizer is None:
            return
        
        # Initialize vocab for training
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, t in enumerate([]):  # No special tokens in training init
            self.vocab[256 + i] = t.encode("utf-8")
        self.vocab_index = max(self.vocab.keys())
        
        byte_to_id = {bytes([i]): i for i in range(256)}
        for i, t in enumerate([]):
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
            self.merges_dict = {merge: i for i, merge in enumerate(self.merges)}
            
            self._apply_merge_to_sequences(left_id, right_id, new_id)
            self._build_pair_frequencies()

        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.id_to_bytes = {k: v for k, v in self.vocab.items()}
