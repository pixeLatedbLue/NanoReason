"""HTTP serving layer: a FastAPI app and its filesystem/analytics backing.

Kept in its own subpackage (and out of the top-level imports) so that the
training and evaluation entrypoints never pay for a FastAPI import — the GPU
boxes that run training do not have the web stack installed.
"""

from __future__ import annotations
