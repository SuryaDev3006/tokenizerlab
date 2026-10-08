# tokenizerlab

**Train, test and evaluate tokenizers, with experiments you can reproduce.**

tokenizerlab is a Python library for building tokenizer-training corpora and
(soon) training, comparing and ablating BPE tokenizers. Every step is
deterministic and recorded, so you can always say exactly what data a tokenizer
was trained on.

> **Status: early alpha (0.1).** The data layer is complete: reading data and
> building, inspecting, saving and reloading training corpora. Tokenizer
> training is the next milestone (see [Roadmap](#roadmap)).

## Install

```bash
pip install "tokenizerlab[parquet]"        # [parquet] is needed to save/load corpora
pip install "tokenizerlab[parquet,hf]"     # add Hugging Face datasets support
```

Requires Python 3.11+.

## Quickstart

```python
from tokenizerlab import Corpus, read

# Read one source per language and tag it
stream = read([
    read("data/te", name="te", lang="te"),
    read("data/en", name="en", lang="en"),
])

# Build a corpus: drop short documents, remove exact duplicates
corpus = Corpus(stream).filter(min_chars=5).dedup()

stats = corpus.stats()
print(stats.documents, stats.bytes, stats.fingerprint)

# Save it once, reuse it in every experiment
corpus.save("artifacts/te-en-v1")
corpus = Corpus.load("artifacts/te-en-v1")

for text in corpus.texts():   # plain strings, ready for tokenizer training
    ...
```

## Reading data

`read()` turns files, directories, Hugging Face datasets and Python iterables
into a lazy stream of documents.

| Source | Example |
|---|---|
| Directory (formats auto-detected) | `read("data/")` |
| Text, one document per file or per line | `read("book.txt")`, `read("lines.txt", unit="line")` |
| JSONL / NDJSON (also `.gz`) | `read("news.jsonl", text_field="text")` |
| CSV / TSV | `read("reviews.csv", text_field="review")` |
| Parquet | `read("shard.parquet", metadata_fields=["url"])` |
| Hugging Face dataset | `read("hf:wikitext", config="wikitext-2-raw-v1", split="train")` |
| Python iterable | `read(["first doc", "second doc"], name="toy")` |
| Code files | `read("repo/", format="text", glob="*.py")` |

- **Exact text.** For structured formats only the chosen `text_field` becomes
  text, so JSON or CSV syntax never leaks into training data. Text is never
  normalized or modified.
- **Deterministic.** Files are read in a fixed, platform-independent order;
  hidden files and folders (`.git`, `.venv`) are skipped.
- **Robust.** A broken file or row is skipped with a warning
  (`on_error="warn"`), recorded in `stream.errors`, and the rest keeps loading.
  Errors are complete once a pass has finished.

## Building a corpus

Each operation returns a new, lazy `Corpus`; nothing is read until you iterate.

```python
sources = read([
    read("data/te", name="te", lang="te"),
    read("data/en", name="en", lang="en"),
])

corpus = (
    Corpus(sources)
    .filter(min_chars=100)               # also always drops empty documents
    .dedup()                             # exact duplicates, first occurrence kept
    .sample(max_bytes=5_000_000_000, seed=42)
)

# Language mixing uses the lang tags (untagged documents are dropped).
# alpha follows mT5 / XLM-R: 1.0 keeps natural proportions, 0.3 moves toward uniform.
balanced = corpus.mix(alpha=0.3, seed=42)
custom = corpus.mix({"te": 0.5, "en": 0.5}, seed=42)
```

Sampling is decided by a seeded hash of each document's id, so it is
reproducible and independent of input order. Mixing only downsamples, never
repeats documents.

## Saving and provenance

`corpus.save(path)` writes `corpus.parquet` and a `manifest.json` recording
every step and its parameters, per-step document counts, reader errors and
skipped files, resolved Hugging Face revisions, statistics and the corpus
**fingerprint**: a hash of all document contents in order. Two corpora with
the same fingerprint are identical training inputs.

`Corpus.load(path)` verifies every document's hashes and the fingerprint. The
manifest also states whether the corpus is `reproducible` (for example, data
from an in-memory list cannot be recreated from the manifest).

## Roadmap

1. ✅ Data foundation: readers, corpus building, statistics, manifests
2. BPE core: normalizers, pre-tokenizers, byte-level BPE trainer
3. Runtime: encode / decode, special tokens
4. Saving and loading tokenizers
5. Analysis and evaluation: fertility, compression, cross-language parity
6. Benchmarks and comparison with Hugging Face `tokenizers`, SentencePiece and tiktoken
7. Experiments: sweeps and ablations with reproducible reports

## Development

```bash
git clone https://github.com/SuryaDev3006/tokenizerlab
cd tokenizerlab
uv sync
uv run pytest
uv run ruff check . && uv run mypy
```

## License

MIT
