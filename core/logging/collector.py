from typing import List, Optional
from core.logging.events import Event


class EventCollector:
    """
    Lightweight in-memory event collector.

    Preserves event order and supports retrieving or clearing recorded events.
    Does not write to external storage, databases, or network services.
    """

    def __init__(self) -> None:
        self._events: List[Event] = []

    def record(self, event: Event) -> None:
        """
        Record a structured event.

        Args:
            event: The Event instance to store.

        Raises:
            TypeError: If event is not an instance of Event.
        """
        if not isinstance(event, Event):
            raise TypeError(f"Expected Event instance, got {type(event).__name__}")
        self._events.append(event)

    def get_events(self, session_id: Optional[str] = None) -> List[Event]:
        """
        Retrieve recorded events, optionally filtered by session_id.

        Args:
            session_id: Optional session identifier filter.

        Returns:
            A list of matching Event objects in the order they were recorded.
        """
        if session_id is not None:
            return [e for e in self._events if e.session_id == session_id]
        return list(self._events)

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)
