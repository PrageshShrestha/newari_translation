"""
correction_engine.py

Runtime reader + corrector for the compact artifacts exported by the
"Newari Autocorrect — Full Pipeline" notebook:

    dictionary.bin       word list + quantized unigram log-probs
    symspell_index.bin   delete-variant -> word-id lookup (SymSpell)
    bigrams.bin          (word_id_a, word_id_b, count) triples

This mirrors the notebook's in-memory logic (deletes(), weighted_edit_distance(),
lm_logprob(), correct_text()) exactly, but loads from the shipped binary files
instead of keeping the training-time Python objects around, so it's cheap to
import into a server process.
"""

import os
import re
import struct
import math
import statistics
import unicodedata
import csv
from collections import defaultdict


WORD_RE = re.compile(r"[\u0900-\u097F:]+")

MAX_EDIT_DISTANCE = 2
LAMBDA = 3.0  # weight of error cost vs language model, same as notebook

CONFUSABLE_GROUPS = [
    "श ष स",
    "ि ी",
    "ु ू",
    "ं ँ",
    "व ब",
    "क ख",
    "ग घ",
    "ट ठ",
    "ड ढ",
    "प फ",
    "त थ",
    "द ध",
    "ः",
]

SUBSTITUTION_COST = {}
for group in CONFUSABLE_GROUPS:
    chars = group.split()
    for a in chars:
        for b in chars:
            if a != b:
                SUBSTITUTION_COST[(a, b)] = 0.3

DEFAULT_SUB_COST = 1.0
DEFAULT_INSDEL_COST = 1.0


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text)
    return text


def weighted_edit_distance(a: str, b: str) -> float:
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i * DEFAULT_INSDEL_COST
    for j in range(m + 1):
        d[0][j] = j * DEFAULT_INSDEL_COST

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if a[i - 1] == b[j - 1] else SUBSTITUTION_COST.get(
                (a[i - 1], b[j - 1]), DEFAULT_SUB_COST
            )
            d[i][j] = min(
                d[i - 1][j] + DEFAULT_INSDEL_COST,
                d[i][j - 1] + DEFAULT_INSDEL_COST,
                d[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[n][m]


def deletes(word, max_dist):
    results = {word}
    frontier = {word}
    for _ in range(max_dist):
        new_frontier = set()
        for w in frontier:
            for i in range(len(w)):
                new_frontier.add(w[:i] + w[i + 1:])
        results |= new_frontier
        frontier = new_frontier
    return results

from typing import Tuple, List, Dict
from symspellpy import SymSpell, Verbosity

class CorrectionEngine:
    def __init__(self, artifacts_dir="artifacts", confusion_csv=None):
        self.artifacts_dir = artifacts_dir
        self._load_dictionary(f"{artifacts_dir}/dictionary.bin")  # already loads self.words
        self._load_bigrams(f"{artifacts_dir}/bigrams.bin")
        self.total_unigrams = sum(self.counts)
        self._load_gazetteer_if_present(f"{artifacts_dir}/gazetteer_everestner.txt")
        self.vocab_set = set(self.words)
        
        # --- BUILD SYMSPELL INDEX FROM self.words (instead of loading .bin) ---
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # Add all words from dictionary.bin
        for word in self.words:
            self.sym_spell.create_dictionary_entry(word, 1)
        
        # REMOVE self.delete_index entirely (or keep as empty dict for compatibility)
        self.delete_index = {}
        
        # Load confusion pairs if provided
        self.word_confusion = {}
        if confusion_csv and os.path.exists(confusion_csv):
            self._load_confusion_pairs(confusion_csv)
    # ---- loaders -----------------------------------------------------

    def _load_dictionary(self, path):
        with open(path, "rb") as f:
            (word_count,) = struct.unpack("<I", f.read(4))
            (blob_len,) = struct.unpack("<I", f.read(4))
            words_blob = f.read(blob_len)
            words = words_blob.decode("utf-8").split("\x00")
            probs_raw = f.read(word_count * 2)
            quantized = struct.unpack(f"<{word_count}H", probs_raw)

        self.words = words  # index -> word (word_id)
        self.word_to_id = {w: i for i, w in enumerate(words)}
        self.quantized = quantized

        # Reconstruct approximate per-word counts from quantized log-probs is lossy,
        # so instead we recover a *relative* probability directly from the quantized
        # value for language-model scoring (monotonic with the original log-prob).
        self._q_min = min(quantized) if quantized else 0
        self._q_max = max(quantized) if quantized else 1

        # We don't have exact raw counts anymore (they were quantized away), so we
        # derive a pseudo-count that preserves relative ordering for lm scoring.
        self.counts = [q + 1 for q in quantized]

    def _load_symspell_index(self, path):
        with open(path, "rb") as f:
            (variant_count,) = struct.unpack("<I", f.read(4))
            (blob_len,) = struct.unpack("<I", f.read(4))
            variants_blob = f.read(blob_len)
            variants = variants_blob.decode("utf-8").split("\x00")

            index = {}
            for v in variants:
                (n_ids,) = struct.unpack("<H", f.read(2))
                ids = struct.unpack(f"<{n_ids}I", f.read(4 * n_ids))
                index[v] = set(ids)

        self.delete_index = index

    def _load_bigrams(self, path):
        bigram = defaultdict(int)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<I", f.read(4))
            for _ in range(n):
                a_id, b_id, c = struct.unpack("<IIH", f.read(10))
                bigram[(a_id, b_id)] = c
        self.bigram = bigram

    def _load_gazetteer_if_present(self, path):
        """If gazetteer_everestner.txt (word<TAB>count per line, as built by
        the training notebook / evaluation script) sits next to the three
        .bin artifacts, merge its entries directly into self.words /
        self.word_to_id / self.counts / self.delete_index. Optional --
        silently does nothing if the file isn't there, so existing
        deployments without a gazetteer are unaffected."""
        if not os.path.exists(path):
            return

        fallback_count = int(statistics.median(self.counts)) if self.counts else 1
        n_new = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                entity, _, _count = line.rstrip("\n").partition("\t")
                if not entity:
                    continue
                for tok in entity.split():
                    if not WORD_RE.fullmatch(tok) or tok in self.word_to_id:
                        continue
                    wid = len(self.words)
                    self.words.append(tok)
                    self.word_to_id[tok] = wid
                    self.counts.append(fallback_count)
                    for variant in deletes(tok, MAX_EDIT_DISTANCE):
                        self.delete_index.setdefault(variant, set()).add(wid)
                    n_new += 1

        if n_new:
            self.total_unigrams = sum(self.counts)
            # Update vocab set with new words
            self.vocab_set = set(self.words)
        print(f"[correction_engine] Gazetteer '{path}': merged {n_new} new proper-noun "
              f"entries (fallback_count={fallback_count}). Vocab size now {len(self.words)}.")

    def _load_confusion_pairs(self, csv_path, min_count=3):
        """Load word substitution pairs from a CSV file."""
        if not os.path.exists(csv_path):
            return
        
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref = row['reference_token'].strip()
                mis = row['misrecognized_as'].strip()
                # Skip placeholders
                if mis in ('⁇', '?', '', '??', '⁇', '?', '??'):
                    continue
                try:
                    count = int(row['count_before_correction'])
                except (ValueError, KeyError):
                    count = 1
                
                # Only use frequent pairs to avoid over-correction
                if count < min_count:
                    continue
                    
                if mis not in self.word_confusion:
                    self.word_confusion[mis] = {}
                self.word_confusion[mis][ref] = self.word_confusion[mis].get(ref, 0) + count

    # ---- scoring --------------------------------------------------

    def lm_logprob(self, word, prev_word):
        wid = self.word_to_id.get(word)
        if wid is None:
            return math.log(1e-9)

        if prev_word is not None:
            pid = self.word_to_id.get(prev_word)
            if pid is not None and (pid, wid) in self.bigram and self.bigram[(pid, wid)] > 0:
                prev_count = self.counts[pid]
                p = 0.4 * self.bigram[(pid, wid)] / max(prev_count, 1)
                return math.log(p + 1e-12)

        count = self.counts[wid]
        p = (count + 0.1) / (self.total_unigrams + 0.1 * len(self.counts))
        return math.log(p)

    def get_candidates(self, typed_word):
    """Use SymSpell for fast fuzzy lookup."""
        candidates = set()
        
        if typed_word in self.word_to_id:
            candidates.add(typed_word)
        
        # SymSpell lookup (edit distance 2)
        suggestions = self.sym_spell.lookup(
            typed_word, 
            verbosity=Verbosity.ALL, 
            max_edit_distance=MAX_EDIT_DISTANCE
        )
        for suggestion in suggestions:
            candidates.add(suggestion.term)
        
        return candidates

    def correct(self, typed_word, prev_word=None, top_k=5):
        candidates = self.get_candidates(typed_word)
        if not candidates:
            return []
        scored = []
        for cand in candidates:
            edit_cost = weighted_edit_distance(typed_word, cand)
            score = self.lm_logprob(cand, prev_word) - LAMBDA * edit_cost
            scored.append((cand, score, edit_cost))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    
    def _apply_confusion_correction(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Apply confusion-based correction to ALL words, not just unknown ones."""
        if not self.word_confusion:
            return text, []

        # Split the same way Pass 1 and the rest of the codebase do: capture
        # Devanagari word-runs and leave everything else (punctuation, danda,
        # whitespace) untouched and in place. Plain `text.split()` (whitespace
        # only) would leave punctuation glued to a word (e.g. "शब्द।"), which
        # then never exactly matches a word_confusion key -- silently losing
        # matches whenever ASR/reference output includes any punctuation.
        pieces = re.split(r"([\u0900-\u097F:]+)", text)
        word_positions = [i for i, p in enumerate(pieces) if WORD_RE.fullmatch(p)]
        words = [pieces[i] for i in word_positions]
        if not words:
            return text, []

        changes = []
        new_words = list(words)
        n = len(words)

        for i, w in enumerate(words):
            # Check if this word has a confusion mapping
            if w not in self.word_confusion:
                continue

            candidates = self.word_confusion[w]
            # Filter to candidates that exist in vocabulary
            valid = [(cand, count) for cand, count in candidates.items() if cand in self.vocab_set]
            if not valid:
                continue

            # Score with bigram context
            def score(cand: str, count: int) -> float:
                s = float(count)
                if i > 0 and (words[i-1], cand) in self.bigram:
                    s *= self.bigram[(words[i-1], cand)]
                if i < n-1 and (cand, words[i+1]) in self.bigram:
                    s *= self.bigram[(cand, words[i+1])]
                return s

            best_cand = max(valid, key=lambda x: score(x[0], x[1]))[0]
            if best_cand != w:
                changes.append({'original': w, 'corrected': best_cand})
                new_words[i] = best_cand

        if not changes:
            return text, []

        for pos, new_word in zip(word_positions, new_words):
            pieces[pos] = new_word
        return "".join(pieces), changes




    def correct_text(self, text: str, min_score_gap: float = 0.0):
        """Autocorrect with two-pass approach:
        1. Standard correction (unknown words only)
        2. Confusion correction (ALL words, using ASR error patterns)
        """
        text = clean_text(text)

        # Pass 1: Standard correction for unknown words
        pieces = re.split(r"([\u0900-\u097F:]+)", text)
        corrected_pieces = []
        changes = []
        prev_word = None

        for piece in pieces:
            if WORD_RE.fullmatch(piece):
                word = piece
                # Only correct if word is NOT in vocabulary
                if word in self.vocab_set:
                    corrected_pieces.append(word)
                    prev_word = word
                    continue

                candidates = self.get_candidates(word)
                if not candidates:
                    corrected_pieces.append(word)
                    prev_word = word
                    continue

                scored = []
                for cand in candidates:
                    edit_cost = weighted_edit_distance(word, cand)
                    score = self.lm_logprob(cand, prev_word) - LAMBDA * edit_cost
                    scored.append((cand, score))
                scored.sort(key=lambda x: -x[1])

                best_word, best_score = scored[0]
                # word is OOV here (we already returned above if word in vocab_set),
                # so this mirrors the notebook's own comparison: the typed word gets
                # its own (very low, OOV) lm_logprob, and best_word must beat it by
                # at least min_score_gap -- min_score_gap actually does something now,
                # instead of the previous hardcoded, non-adaptive `> -10.0` cutoff.
                typed_as_is_score = self.lm_logprob(word, prev_word)
                if best_score - typed_as_is_score >= min_score_gap and best_word != word:
                    corrected_pieces.append(best_word)
                    changes.append({"original": word, "corrected": best_word})
                    prev_word = best_word
                else:
                    corrected_pieces.append(word)
                    prev_word = word
            else:
                corrected_pieces.append(piece)

        corrected_text = "".join(corrected_pieces)
        
        # Pass 2: ALWAYS apply confusion correction to ALL words
        if self.word_confusion:
            confusion_corrected, conf_changes = self._apply_confusion_correction(corrected_text)
            if conf_changes:
                # Merge changes, avoiding duplicates
                existing_originals = {c['original'] for c in changes}
                for c in conf_changes:
                    if c['original'] not in existing_originals:
                        changes.append(c)
                corrected_text = confusion_corrected
        
        return corrected_text, changes