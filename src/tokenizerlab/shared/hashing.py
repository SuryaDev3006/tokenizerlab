"""Hashing primitives shared by documents, fingerprints and selection steps.

Everything that hashes text uses these, so one change to HASH_VERSION covers all of them.
"""

import hashlib

# Recorded in manifests: changing how any hash is computed must change this string.
HASH_VERSION = "blake2b-128-v1"
DIGEST_SIZE = 16


def encode_text(text: str) -> bytes:
    """Encode text as UTF-8, keeping lone surrogates so that every str can be hashed."""
    return text.encode("utf-8", "surrogatepass")


def utf8_size(text: str) -> int:
    """Size of text in UTF-8 bytes, counted the same way it is hashed."""
    return len(encode_text(text))


def digest_hex(data: bytes) -> str:
    """The hexadecimal BLAKE2b-128 digest of data."""
    return hashlib.blake2b(data, digest_size=DIGEST_SIZE).hexdigest()


def digest16(data: bytes) -> bytes:
    """The raw 16-byte BLAKE2b digest of data, for compact keys."""
    return hashlib.blake2b(data, digest_size=DIGEST_SIZE).digest()


def hash64(data: bytes) -> int:
    """The 8-byte BLAKE2b digest of data as a big-endian integer; unlike hash(), the same in
    every process."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def incremental_digest() -> "hashlib.blake2b":
    """An empty BLAKE2b-128 hasher, for digests built up piece by piece."""
    return hashlib.blake2b(digest_size=DIGEST_SIZE)


def unit_interval(key: str) -> float:
    """A deterministic value in [0, 1): the first 8 bytes of BLAKE2b(key) / 2^64.

    BLAKE2b instead of hash(), so the value is identical across processes and machines.
    """
    digest = hashlib.blake2b(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
