"""Exceptions for the harmonic package."""


class HarmonicError(Exception):
    """Base class so callers can catch all harmonic errors together."""


class ConfigurationError(HarmonicError):
    """Invalid or missing configuration."""


class DataError(HarmonicError):
    """Invalid or missing input data."""


class PredictionError(HarmonicError):
    """Transit prediction cannot proceed."""
