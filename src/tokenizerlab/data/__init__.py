"""The data foundation, one feature per module or package:

Source → readers → document → corpus → tokenizer trainer
                                ↘ stats (CorpusStats, fingerprint, manifest)
"""

from tokenizerlab.data.corpus import Corpus
from tokenizerlab.data.document import Document, InvalidDocumentError
from tokenizerlab.data.readers import DocumentStream, ReadError, read
from tokenizerlab.data.stats import CorpusStats

__all__ = [
    "Corpus",
    "CorpusStats",
    "Document",
    "DocumentStream",
    "InvalidDocumentError",
    "ReadError",
    "read",
]
