"""Typed errors raised at module boundaries."""


class EchoError(Exception):
    """Base class for all echo_routing errors."""


class ConfigError(EchoError, ValueError):
    """Missing or inconsistent configuration."""


class LabelError(EchoError, ValueError):
    """Unknown label code or inconsistent mapping table."""


class ManifestError(EchoError, ValueError):
    """Malformed or inconsistent split files / manifests."""


class FrameError(EchoError, RuntimeError):
    """Cine has no readable frames or invalid sampling parameters."""


class CacheMissError(EchoError, FileNotFoundError):
    """Requested cached features are absent."""


class RecipeError(EchoError, ValueError):
    """Invalid constructed-stream recipe."""
