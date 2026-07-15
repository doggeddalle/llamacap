class LlamacapError(Exception):
    """Base error for all llamacap fatal (startup) failures."""


class BinaryNotFoundError(LlamacapError):
    pass


class ProfileError(LlamacapError):
    pass


class ModelNotFoundError(LlamacapError):
    pass


class CaptionGenerationError(LlamacapError):
    """Raised for a single-image failure; caught by the runner, never fatal."""
