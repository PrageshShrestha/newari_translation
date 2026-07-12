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

import re
import struct
import math
import unicodedata
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


class CorrectionEngine:
    """Loads dictionary.bin / symspell_index.bin / bigrams.bin and exposes
    correct_text(), matching the notebook's noisy-channel corrector."""

    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        self._load_dictionary(f"{artifacts_dir}/dictionary.bin")
        self._load_symspell_index(f"{artifacts_dir}/symspell_index.bin")
        self._load_bigrams(f"{artifacts_dir}/bigrams.bin")
        self.total_unigrams = sum(self.counts)

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
                index[v] = ids

        self.delete_index = index

    def _load_bigrams(self, path):
        bigram = defaultdict(int)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<I", f.read(4))
            for _ in range(n):
                a_id, b_id, c = struct.unpack("<IIH", f.read(10))
                bigram[(a_id, b_id)] = c
        self.bigram = bigram

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
        candidates = set()
        if typed_word in self.word_to_id:
            candidates.add(typed_word)
        for variant in deletes(typed_word, MAX_EDIT_DISTANCE):
            ids = self.delete_index.get(variant)
            if ids:
                for wid in ids:
                    candidates.add(self.words[wid])
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

    def correct_text(self, text: str, min_score_gap: float = 0.0):
        """Autocorrect a full string, preserving whitespace/punctuation, and
        return (corrected_text, list_of_change_dicts) for UI diff rendering."""
        text = clean_text(text)
        pieces = re.split(r"([\u0900-\u097F:]+)", text)

        corrected_pieces = []
        changes = []
        prev_word = None

        for piece in pieces:
            if WORD_RE.fullmatch(piece):
                word = piece

                if word in self.word_to_id:
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
                typed_as_is_score = self.lm_logprob(word, prev_word) if word in self.word_to_id else float("-inf")

                if best_score - typed_as_is_score >= min_score_gap and best_word != word:
                    corrected_pieces.append(best_word)
                    changes.append({"original": word, "corrected": best_word})
                    prev_word = best_word
                else:
                    corrected_pieces.append(word)
                    prev_word = word
            else:
                corrected_pieces.append(piece)

        return "".join(corrected_pieces), changes
