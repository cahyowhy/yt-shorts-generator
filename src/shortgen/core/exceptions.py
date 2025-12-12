"""Custom exceptions for ShortGen."""


class ShortGenError(Exception):
    """Base exception for ShortGen."""

    pass


class DownloadError(ShortGenError):
    """Error during video download."""

    pass


class TranscriptionError(ShortGenError):
    """Error during audio transcription."""

    pass


class AnalysisError(ShortGenError):
    """Error during video analysis."""

    pass


class ProcessingError(ShortGenError):
    """Error during video processing."""

    pass


class RenderError(ShortGenError):
    """Error during final rendering."""

    pass


class ConfigurationError(ShortGenError):
    """Error in configuration."""

    pass


class ModelNotFoundError(ShortGenError):
    """Required AI model not found."""

    pass
