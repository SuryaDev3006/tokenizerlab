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
stream = read(
    [
        read("data/te", name="te", lang="te"),
        read("data/en", name="en", lang="en"),
    ]
)

# Build a corpus: drop short documents, remove exact duplicates
corpus = Corpus(stream).filter(min_chars=5).dedup()

stats = corpus.stats()
print(stats.documents, stats.bytes, stats.fingerprint)

# Save it once, reuse it in every experiment
corpus.save("artifacts/te-en-v1")
corpus = Corpus.load("artifacts/te-en-v1")

for text in corpus.texts():  # plain strings, ready for tokenizer training
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
  Errors are complete once a pass has finished. Files with unsupported
  extensions are listed in `stream.skipped` and produce one warning per read,
  as does a directory with no readable files.

### Training on code and raw files

Dataset formats (JSONL, CSV/TSV, Parquet, Hugging Face) give one document per
row. Anything read with `format="text"` gives one document per file, with the
file's exact contents.

So a folder of source code, Markdown or JSON files is read with
`format="text"`:

```python
code = read("repo/", format="text", glob="*.py")  # one document per .py file
table = read("users.csv", format="text")  # the whole CSV file, as written
config = read("settings.json", format="text")  # the whole JSON file, as written
```

- A text file with one record per line needs `unit="line"`:
  `read("lines.txt", unit="line")`.
- Read dataset files with `text_field` when you want their records, not their
  syntax: reading a JSONL file with `format="text"` gives JSON-escaped text
  (`\n`, `\"`, `త`) instead of the records' own text.

## Building a corpus

Each operation returns a new, lazy `Corpus`; nothing is read until you iterate.

```python
sources = read(
    [
        read("data/te", name="te", lang="te"),
        read("data/en", name="en", lang="en"),
    ]
)

corpus = (
    Corpus(sources)
    .filter(min_chars=100)  # also always drops empty documents
    .dedup()  # exact duplicates, first occurrence kept
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

`dedup()` remembers the content hashes it has seen in memory, about 100 MB per
million unique documents. For larger corpora, `dedup(on_disk=True)` keeps them
in a temporary SQLite database instead: the same documents are kept, memory
stays flat, and it runs slower.

## Removing duplicates and splitting

Two rules keep a validation set honest:

1. Validation must not share duplicates with training.
2. Dedup happens before the split, or consistently across the two sides.

The recommended order is read, filter, dedup, sample or mix, split:

```python
corpus = Corpus(sources).filter(min_chars=100)

corpus.dedup()  # exact duplicates
corpus.dedup(near=True)  # near-duplicates too (FineWeb's settings)
train, validation = corpus.dedup(near=True).split(validation=0.02, seed=0)

# Dedup after the split instead: validation keeps every conflict, training loses it.
train, validation = corpus.split(validation=0.02, seed=0)
validation = validation.dedup(near=True)
train = train.dedup(near=True, against=validation)
```

A document's side depends only on the seed and its content, so identical texts
always land on the same side, and adding documents or earlier steps never moves
one. `split(by="source")` (or `by=` a function) keeps whole sources together
instead; then identical texts in different sources can land on different sides,
so dedup first or use `against=`. Deduplicating the train side without
`against=` raises an error, except for exact dedup after a content split.

**What near-duplicate means here.** Texts are compared by their 5-word phrases,
ignoring case and spacing (MinHash with 112 hashes in 14 bands of 8). At the
default threshold of 0.75, a copy with about 1% of its words changed is always
caught, about 3% changed is caught three times in four, and 8% changed almost
never. `dedup(near=True, threshold=0.9)` drops only very similar texts.

Near dedup is opt-in: more dedup is not always better (FineWeb found that
global dedup hurt). Its limits:

- It is word-based, so for scripts written without spaces it only finds
  near-exact copies.
- Texts under 5 words are compared exactly, ignoring case and spacing.
- It costs about 1.1 GB of memory per million kept documents, unless
  `on_disk=True`.
- It processes a few MB of text per second.

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

1. ✅ Data foundation: readers, corpus building, near-duplicate dedup, leakage-resistant
   train/validation split, statistics, manifests
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
