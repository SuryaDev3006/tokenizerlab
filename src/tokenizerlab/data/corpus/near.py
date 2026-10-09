"""Near-duplicate keys: one-permutation MinHash over word 5-grams, with LSH banding.

Two texts are near-duplicates when they share any key. Long texts get one key per band of
their MinHash signature; texts under NGRAM words get a single key, equal only for texts that
are equal ignoring case and spacing. Text is only normalized for comparing, never changed.
The settings follow FineWeb: 112 hashes in 14 bands of 8 rows at the default threshold 0.75.
"""

from __future__ import annotations

import functools
import struct

from tokenizerlab.shared.hashing import digest16, encode_text, hash64

HASHES = 112
NGRAM = 5  # word 5-grams
METHOD = "minhash-oph-v1"  # recorded in the history; changes whenever any key changes
EMPTY = 1 << 64  # larger than any bin value, so it marks a bin no shingle reached
MASK = (1 << 64) - 1


def words(text: str) -> list[str]:
    """The words of text in the form used only for comparing: case-folded, split on spaces."""
    return text.casefold().split()


def shingle_hashes(text_words: list[str]) -> set[int]:
    """One 64-bit BLAKE2b value per run of NGRAM consecutive words."""
    return {
        hash64(encode_text(" ".join(text_words[start : start + NGRAM])))
        for start in range(len(text_words) - NGRAM + 1)
    }


def probe(bin_index: int, attempt: int) -> int:
    """The bin that empty bin `bin_index` copies from on its n-th attempt (a fixed mix of both)."""
    mixed = (
        bin_index * 0x9E3779B97F4A7C15 + attempt * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB
    ) & MASK
    mixed ^= mixed >> 31
    mixed = (mixed * 0xD6E8FEB86659FD93) & MASK
    mixed ^= mixed >> 32
    return mixed % HASHES


def signature(text: str) -> list[int] | None:
    """The HASHES-value MinHash signature of text, or None for a text under NGRAM words.

    One permutation: each shingle hash picks a bin and competes for its minimum. Bins no
    shingle reached copy a filled bin chosen by probe(), so short texts still fill every bin.
    """
    text_words = words(text)
    if len(text_words) < NGRAM:
        return None

    bins = [EMPTY] * HASHES
    for shingle in shingle_hashes(text_words):
        bin_index, value = shingle % HASHES, shingle // HASHES
        bins[bin_index] = min(bins[bin_index], value)

    filled = bins[:]
    for bin_index in range(HASHES):
        if filled[bin_index] == EMPTY:
            attempt = 1
            while filled[probe(bin_index, attempt)] == EMPTY:
                attempt += 1
            bins[bin_index] = filled[probe(bin_index, attempt)]
    return bins


def detection(similarity: float, bands: int, rows: int) -> float:
    """How often two texts with this similarity share at least one band key."""
    return 1 - (1 - similarity**rows) ** bands


# How often FineWeb's settings catch a pair at exactly their threshold (about 0.772).
TARGET = detection(0.75, 14, 8)


@functools.cache
def banding(threshold: float) -> tuple[int, int]:
    """(bands, rows) that catch a pair at `threshold` about as often as FineWeb's settings
    catch a pair at 0.75; banding(0.75) is (14, 8)."""
    rows = min(
        range(1, HASHES + 1),
        key=lambda rows: abs(detection(threshold, HASHES // rows, rows) - TARGET),
    )
    return HASHES // rows, rows


def band_keys(text_signature: list[int], bands: int, rows: int) -> list[bytes]:
    """One 16-byte key per band; the band number is part of the key."""
    return [
        digest16(struct.pack(f"<B{rows}Q", band, *text_signature[band * rows : (band + 1) * rows]))
        for band in range(bands)
    ]


def short_key(text: str) -> bytes:
    """The key of a text under NGRAM words: equal for texts equal ignoring case and spacing."""
    return digest16(encode_text(" ".join(words(text))))


def near_keys(text: str, bands: int, rows: int) -> list[bytes]:
    """Every key a near-duplicate of text would share with it."""
    text_signature = signature(text)
    if text_signature is None:
        return [short_key(text)]
    return band_keys(text_signature, bands, rows)
