"""Mix: downsample languages to target shares, from weights or from bytes ** alpha (mT5/XLM-R).

The mix workflow is invariant; how target shares are chosen varies, so it is a ShareStrategy.
The share arithmetic is kept in pure functions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tokenizerlab.data.corpus.steps import StepContext, selection_value, validate_max_bytes
from tokenizerlab.data.document import UNDETERMINED_LANG, Document
from tokenizerlab.errors import ConfigurationError, TokenizerLabError
from tokenizerlab.shared.hashing import utf8_size


class UnreachableMixError(TokenizerLabError):
    """A mix target cannot be met, e.g. a weighted language has no documents."""


@dataclass(frozen=True, slots=True)
class MixResult:
    """Target vs achieved language shares of a mix, and what it dropped."""

    target_bytes: int | None
    target_shares: dict[str, float]
    keep_fractions: dict[str, float]
    achieved_shares: dict[str, float]
    dropped: dict[str, int]  # documents per unweighted language; untagged under "und"

    def as_record(self) -> dict[str, Any]:
        """The result as JSON-ready fields."""
        return asdict(self)


# ---------------------------------------------------------------- share strategies


class ShareStrategy(Protocol):
    """How a mix chooses each language's share of the output bytes."""

    def target_shares(self, available_bytes: Mapping[str, int]) -> dict[str, float]:
        """Shares summing to 1, from the bytes available per language."""
        ...

    def parameters(self) -> dict[str, Any]:
        """The strategy's parameters as recorded in the history."""
        ...


@dataclass(frozen=True, slots=True)
class WeightedShares:
    """Shares proportional to fixed weights; languages not weighted are dropped."""

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        """Weights must be a non-empty mapping of positive numbers; keep a private copy."""
        if not self.weights or any(weight <= 0 for weight in self.weights.values()):
            raise ConfigurationError("weights must be a non-empty mapping of positive numbers")
        object.__setattr__(self, "weights", dict(self.weights))

    def target_shares(self, available_bytes: Mapping[str, int]) -> dict[str, float]:
        """Normalized weights; every weighted language must have data."""
        # A weighted language without data would force the whole corpus to be empty.
        missing = sorted(lang for lang in self.weights if lang not in available_bytes)
        if missing:
            raise UnreachableMixError(f"mix: no documents for weighted languages {missing}")
        return normalize(self.weights)

    def parameters(self) -> dict[str, Any]:
        """weights (and no alpha)."""
        return {"weights": dict(self.weights), "alpha": None}


@dataclass(frozen=True, slots=True)
class AlphaShares:
    """Shares proportional to bytes ** alpha: 1 keeps natural proportions, smaller is closer
    to uniform. Named alpha as in mT5 and XLM-R, where the temperature T = 1/alpha."""

    alpha: float

    def __post_init__(self) -> None:
        """alpha must be in (0, 1]."""
        if not 0 < self.alpha <= 1:
            raise ConfigurationError(f"alpha must be in (0, 1], got {self.alpha}")

    def target_shares(self, available_bytes: Mapping[str, int]) -> dict[str, float]:
        """Normalized bytes ** alpha over every language present."""
        return normalize({lang: size**self.alpha for lang, size in available_bytes.items()})

    def parameters(self) -> dict[str, Any]:
        """alpha (and no weights)."""
        return {"weights": None, "alpha": self.alpha}


def share_strategy(weights: Mapping[str, float] | None, alpha: float | None) -> ShareStrategy:
    """The ShareStrategy for mix(weights=...) or mix(alpha=...); exactly one is allowed."""
    if weights is not None and alpha is None:
        return WeightedShares(weights)
    if alpha is not None and weights is None:
        return AlphaShares(alpha)
    raise ConfigurationError("mix needs exactly one of weights or alpha")


# ---------------------------------------------------------------- the step


@dataclass(frozen=True, slots=True)
class Mix:
    """Control language composition using Document.lang, downsampling only.

    Untagged documents and languages without a target share are dropped and counted.
    """

    shares: ShareStrategy
    max_bytes: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """The byte budget, if any, must not be negative."""
        validate_max_bytes(self.max_bytes)

    @property
    def name(self) -> str:
        """Recorded as "mix"."""
        return "mix"

    @property
    def needs_prepass(self) -> bool:
        """Target shares depend on the bytes per language entering the step."""
        return True

    @property
    def reproducible(self) -> bool:
        """The share strategy, max_bytes and seed fully determine the selection."""
        return True

    def parameters(self) -> dict[str, Any]:
        """weights or alpha, max_bytes and seed."""
        return {**self.shares.parameters(), "max_bytes": self.max_bytes, "seed": self.seed}

    def apply(self, documents: Iterator[Document], context: StepContext) -> Iterator[Document]:
        """Yield each language's documents with probability equal to its keep fraction."""
        available_bytes = tagged_bytes(context.bytes_by_lang)
        target_shares = self.shares.target_shares(available_bytes)
        keep_fractions = downsampling_fractions(target_shares, available_bytes, self.max_bytes)

        kept_bytes: Counter[str] = Counter()
        dropped: Counter[str] = Counter()
        for document in documents:
            lang = document.lang
            if lang is None or lang not in keep_fractions:
                dropped[lang or UNDETERMINED_LANG] += 1
            elif selection_value(self.seed, context.step_index, document.id) < keep_fractions[lang]:
                kept_bytes[lang] += utf8_size(document.text)
                yield document

        context.result = MixResult(
            target_bytes=self.max_bytes,
            target_shares=target_shares,
            keep_fractions=keep_fractions,
            achieved_shares=normalize({lang: kept_bytes[lang] for lang in target_shares}),
            dropped=dict(dropped),
        )


# ---------------------------------------------------------------- pure share arithmetic


def tagged_bytes(bytes_by_lang: Mapping[str | None, int]) -> dict[str, int]:
    """Bytes per language, leaving out untagged documents and empty languages."""
    return {lang: size for lang, size in bytes_by_lang.items() if lang is not None and size > 0}


def normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Scale values to sum to 1 (all zeros if they sum to 0)."""
    total = sum(values.values())
    return {key: value / total if total else 0.0 for key, value in values.items()}


def downsampling_fractions(
    target_shares: Mapping[str, float],
    available_bytes: Mapping[str, int],
    max_bytes: int | None,
) -> dict[str, float]:
    """The fraction of each language to keep so the output has the target shares.

    Downsampling only: the output size K = min over languages of bytes / share (capped by
    max_bytes), so every share is reachable without repeating documents and the scarcest
    language limits the total. Each language keeps share * K / bytes <= 1.
    """
    output_bytes = min(
        (available_bytes[lang] / share for lang, share in target_shares.items()),
        default=0.0,
    )
    if max_bytes is not None:
        output_bytes = min(output_bytes, max_bytes)

    return {
        lang: min(1.0, share * output_bytes / available_bytes[lang])
        for lang, share in target_shares.items()
    }
