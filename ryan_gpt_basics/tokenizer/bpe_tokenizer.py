from collections import Counter
import regex as re
import json
import heapq
from typing import Dict, List, Tuple, Iterable, Iterator, Union
from functools import lru_cache
from time import time
from ryan_gpt_basics.tokenizer.pretokenizer import PreTokenizer
from collections import defaultdict


class BPEProcessor:
    """BPE tokenizer that can encode/decode."""

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]] = None, 
                 special_tokens: List[str] | None = None, pretokenizer: "PreTokenizer | None" = None) -> None:
        self.vocab = vocab
        self.merges = merges or []
        self.merges_dict = {merge: i for i, merge in enumerate(self.merges)}
        self.pretokenizer = pretokenizer
        self.sequences: list[list[int]] = []
        self.sequence_counts: list[int] = []
        self.pair_freq: Counter[tuple[int, int]] = Counter()
        
        special_tokens = special_tokens or []
        special_token_bytes = []
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            special_token_bytes.append(token_bytes)
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        
        self._rebuild_vocab_mapping()
        
        self.special_tokens_to_id = {}
        for token_bytes in sorted(special_token_bytes, key=len, reverse=True):
            self.special_tokens_to_id[token_bytes] = self.bytes_to_id[token_bytes]
        
        self.vocab_index: int = max(self.vocab.keys()) if self.vocab else -1
        
        self._build_sequences_for_training()
        self._build_pair_frequencies()

    def _rebuild_vocab_mapping(self) -> None:
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: List[str] | None = None) -> "BPEProcessor":
        with open(vocab_path, "r", encoding="utf-8") as vf:         
            vocab_data = json.load(vf)
            vocab = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}

        with open(merges_path, "r", encoding="utf-8") as mf:
            merges = []
            for line in mf:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                left, right = bytes.fromhex(parts[0]), bytes.fromhex(parts[1])
                merges.append((left, right))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self._encode_single(text):
                yield token_id
    
    def _encode_single(self, text: str) -> List[int]:
        results = self._tokenize_special_tokens(text)
        token_ids = []
        for elem in results:
            if isinstance(elem, str):
                token_ids.extend(self._tokenize_normal_text(elem))
            elif isinstance(elem, int):
                token_ids.append(elem)
        return token_ids
    
    def _tokenize_special_tokens(self, text: str) -> List[Union[str, int]]:
        if not self.special_tokens_to_id:
            return [text]
        
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
            if match.start() > last_end:
                results.append(text[last_end:match.start()])
            results.append(replacements[match.group()])
            last_end = match.end()
        
        if last_end < len(text):
            results.append(text[last_end:])
        
        return results
    
    def _tokenize_normal_text(self, text: str) -> List[int]:
        pat = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        words = pat.findall(text)
        token_ids = []
        for word in words:
            token_ids.extend(self._tokenize_word(word))
        return token_ids
    
    @lru_cache(maxsize=100000)
    def _tokenize_word(self, word: str) -> tuple[int, ...]:
        word_bytes = word.encode("utf-8")
        if word_bytes in self.bytes_to_id:
            return (self.bytes_to_id[word_bytes],)
        
        word_bytes_list = [bytes([b]) for b in word_bytes]
        
        while True:
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
            word_bytes_list[best_idx] = word_bytes_list[best_idx] + word_bytes_list[best_idx + 1]
            del word_bytes_list[best_idx + 1]
        
        return tuple(self.bytes_to_id[b] for b in word_bytes_list)
    
    def decode(self, ids: List[int]) -> str:
        byte_sequence = b"".join(self.vocab[tid] for tid in ids)
        return byte_sequence.decode("utf-8", errors="replace")

    def _build_sequences_for_training(self) -> None:
        if self.pretokenizer is None:
            return
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        special_token_ids = {}
        for token_bytes in self.special_tokens_to_id:
            self.vocab[256 + len(special_token_ids)] = token_bytes
            special_token_ids[token_bytes] = 256 + len(special_token_ids)
        
        self.vocab_index = max(self.vocab.keys())
        
        byte_to_id = {bytes([i]): i for i in range(256)}
        for token_bytes, token_id in special_token_ids.items():
            byte_to_id[token_bytes] = token_id

        for token_tuple, count in self.pretokenizer.pretokenization_dict_to_bytes.items():
            seq = [byte_to_id[elem] for elem in token_tuple]
            self.sequences.append(seq)
            self.sequence_counts.append(count)

    def _build_pair_frequencies(self) -> None:
        """Build pair frequencies and track which sequences contain each pair."""
        self.pair_freq = Counter()
        self.pair_to_seqs = defaultdict(set)  # pair -> set of seq_idx
        
        for seq_idx, sequence in enumerate(self.sequences):
            count = self.sequence_counts[seq_idx]
            seen_pairs = set()
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                self.pair_freq[pair] += count
                seen_pairs.add(pair)
            for pair in seen_pairs:
                self.pair_to_seqs[pair].add(seq_idx)
    
    def _rebuild_heap(self) -> None:
        self.heap = [(-freq, pair) for pair, freq in self.pair_freq.items()]
        heapq.heapify(self.heap)

    def _select_best_pair(self) -> tuple[int, int]:
        while self.heap:
            neg_freq, pair = heapq.heappop(self.heap)
            freq = -neg_freq
            if pair in self.pair_freq and self.pair_freq[pair] == freq:
                return pair
        return None

    def _update_pair_freq(self, pair: tuple[int, int], delta: int) -> None:
        if delta == 0:
            return
        new_freq = self.pair_freq.get(pair, 0) + delta
        if new_freq <= 0:
            self.pair_freq.pop(pair, None)
        else:
            self.pair_freq[pair] = new_freq
            heapq.heappush(self.heap, (-new_freq, pair))

    def _apply_merge_to_sequences(self, left_id: int, right_id: int, new_id: int) -> None:
        """Apply merge only to sequences that contain this pair."""
        merge_pair = (left_id, right_id)
        
        if merge_pair not in self.pair_to_seqs:
            return
        
        affected_seqs = self.pair_to_seqs.pop(merge_pair)
        
        for seq_idx in affected_seqs:
            sequence = self.sequences[seq_idx]
            count = self.sequence_counts[seq_idx]
            
            i = 0
            while i < len(sequence) - 1:
                if sequence[i] == left_id and sequence[i + 1] == right_id:
                    # Remove old neighbor pairs
                    if i > 0:
                        old_left = (sequence[i - 1], left_id)
                        self._update_pair_freq(old_left, -count)
                    
                    if i + 2 < len(sequence):
                        old_right = (right_id, sequence[i + 2])
                        self._update_pair_freq(old_right, -count)
                    
                    # Merge
                    sequence[i] = new_id
                    del sequence[i + 1]
                    
                    # Add new neighbor pairs
                    if i > 0:
                        new_left = (sequence[i - 1], new_id)
                        self._update_pair_freq(new_left, count)
                        self.pair_to_seqs[new_left].add(seq_idx)
                    
                    if i + 1 < len(sequence):
                        new_right = (new_id, sequence[i + 1])
                        self._update_pair_freq(new_right, count)
                        self.pair_to_seqs[new_right].add(seq_idx)
                else:
                    i += 1
        
        self.pair_freq.pop(merge_pair, None)

    def run_bpe(self, num_merges: int) -> None:
        self._build_pair_frequencies()
        self._rebuild_heap()
        start_time = time()
        
        for merge_num in range(num_merges):
            if merge_num % 500 == 0:
                elapsed = time() - start_time
                if elapsed > 0 and merge_num > 0:
                    eta = (num_merges - merge_num) / (merge_num / elapsed) / 60
                    print(f"Merge {merge_num}/{num_merges} | vocab: {len(self.vocab)} | ETA: {eta:.1f}m")
                else:
                    print(f"Merge {merge_num}/{num_merges} | vocab: {len(self.vocab)} | ETA: calculating...")

            best_pair = self._select_best_pair()
            if best_pair is None:
                print(f"No more pairs at iteration {merge_num}")
                break

            left_id, right_id = best_pair
            self.vocab_index += 1
            new_id = self.vocab_index

            self.vocab[new_id] = self.vocab[left_id] + self.vocab[right_id]
            self.merges.append((self.vocab[left_id], self.vocab[right_id]))
            self.merges_dict[(self.vocab[left_id], self.vocab[right_id])] = len(self.merges) - 1

            self._apply_merge_to_sequences(left_id, right_id, new_id)

        self._rebuild_vocab_mapping()
        print(f"BPE complete: {len(self.vocab)} tokens, {len(self.merges)} merges")