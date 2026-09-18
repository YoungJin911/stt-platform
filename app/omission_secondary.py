"""
Backward-compatibility shim.

The full-audio word timeline is now an alignment responsibility and
lives in app.alignment_secondary.

New code should import from app.alignment_secondary directly.
This shim can be removed after all imports are migrated.
"""

from app.alignment_secondary import build_secondary_passes

__all__ = [
    "build_secondary_passes",
]
