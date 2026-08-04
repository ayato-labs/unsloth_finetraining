import pytest
from src.training.trainer import RecentWindowTqdm


def test_recent_window_tqdm_format_dict():
    """Verify that RecentWindowTqdm format_dict is a property returning a mapping dict."""
    t = RecentWindowTqdm(total=100)
    t.update(1)
    t.update(1)

    # Check format_dict property access
    d = t.format_dict
    assert isinstance(d, dict), "format_dict must return a dictionary mapping"
    assert "rate" in d

    # Verify tqdm format_meter compatibility with **d
    formatted = t.format_meter(**t.format_dict)
    assert isinstance(formatted, str)
