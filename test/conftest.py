"""Pytest configuration: use a headless matplotlib backend before any plotting imports."""

import matplotlib

matplotlib.use("Agg", force=True)
