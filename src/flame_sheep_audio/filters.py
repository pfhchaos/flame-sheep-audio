"""Onset event filters — composable post-detection gating.

Each filter takes a BeatEvent + timestamp and returns True to keep it.
Filters are stateful (they track history) and chainable.

Usage:
    filters = [SharpnessFilter(), TempoFilter()]
    for event in raw_events:
        if all(f.accept(event, timestamp) for f in filters):
            handle_event(event)
"""

from ._types import BeatEvent


class OnsetFilter:
    """Base class for onset filters. Default: pass everything."""

    def accept(self, event: BeatEvent, timestamp: float) -> bool:
        return True

    def reset(self):
        """Clear state — call on song change."""
        pass
