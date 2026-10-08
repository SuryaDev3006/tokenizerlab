"""The base of tokenizerlab's exception hierarchy, and the errors shared by every feature.

Every error tokenizerlab raises on purpose derives from TokenizerLabError. Feature-specific
errors live next to their feature and are re-exported by its package.
"""


class TokenizerLabError(Exception):
    """Base class of every error tokenizerlab raises on purpose."""


class ConfigurationError(TokenizerLabError):
    """An argument or setting is invalid; raised before any data is read."""


class MissingDependencyError(TokenizerLabError):
    """An optional dependency needed for this operation is not installed."""
