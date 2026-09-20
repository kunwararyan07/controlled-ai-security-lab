from datetime import datetime
import unittest

from core.logging import Event, EventCollector


class TestLogging(unittest.TestCase):
    def test_event_defaults_and_timestamp(self):
        event = Event(user_input="hello")
        self.assertIsNotNone(event.timestamp)
        # Verify ISO-8601 format
        parsed_time = datetime.fromisoformat(event.timestamp)
        self.assertIsNotNone(parsed_time)
        self.assertEqual(event.user_input, "hello")
        self.assertIsNone(event.tool_call)

    def test_event_to_dict(self):
        event = Event(session_id="s123", event_type="test", user_input="hi")
        data_full = event.to_dict(include_none=True)
        self.assertEqual(data_full["session_id"], "s123")
        self.assertIn("tool_call", data_full)
        self.assertIsNone(data_full["tool_call"])

        data_compact = event.to_dict(include_none=False)
        self.assertEqual(data_compact["session_id"], "s123")
        self.assertEqual(data_compact["user_input"], "hi")
        self.assertNotIn("tool_call", data_compact)

    def test_event_collector_record_and_order(self):
        collector = EventCollector()
        self.assertEqual(len(collector), 0)

        e1 = Event(session_id="s1", event_type="input")
        e2 = Event(session_id="s1", event_type="output")
        e3 = Event(session_id="s2", event_type="input")

        collector.record(e1)
        collector.record(e2)
        collector.record(e3)

        self.assertEqual(len(collector), 3)
        events = collector.get_events()
        self.assertEqual(events, [e1, e2, e3])

    def test_event_collector_filter_by_session(self):
        collector = EventCollector()
        collector.record(Event(session_id="s1", event_type="e1"))
        collector.record(Event(session_id="s2", event_type="e2"))
        collector.record(Event(session_id="s1", event_type="e3"))

        s1_events = collector.get_events(session_id="s1")
        self.assertEqual(len(s1_events), 2)
        self.assertEqual([e.event_type for e in s1_events], ["e1", "e3"])

    def test_event_collector_clear(self):
        collector = EventCollector()
        collector.record(Event(session_id="s1"))
        self.assertEqual(len(collector), 1)
        collector.clear()
        self.assertEqual(len(collector), 0)
        self.assertEqual(collector.get_events(), [])

    def test_event_collector_type_check(self):
        collector = EventCollector()
        with self.assertRaises(TypeError):
            collector.record("not an event")  # type: ignore


if __name__ == "__main__":
    unittest.main()
