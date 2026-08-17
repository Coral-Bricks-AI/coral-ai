import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import codex_usage


def event(timestamp, event_type, payload):
    return {"timestamp": timestamp, "type": event_type, "payload": payload}


class CodexUsageTest(unittest.TestCase):
    def write_rollout(self, root, events, name="rollout-test.jsonl"):
        path = Path(root) / "2026" / "08" / "17" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(item) for item in events) + "\n")
        return path

    def test_model_switch_and_window(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_rollout(
                directory,
                [
                    event("2026-08-17T00:00:00Z", "session_meta", {"source": "cli", "thread_source": "user"}),
                    event("2026-08-17T00:00:01Z", "turn_context", {"model": "gpt-sol"}),
                    event("2026-08-17T00:00:02Z", "event_msg", {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10, "reasoning_output_tokens": 4, "total_tokens": 110}}}),
                    event("2026-08-17T01:00:00Z", "turn_context", {"model": "gpt-terra"}),
                    event("2026-08-17T01:00:01Z", "event_msg", {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 50, "cached_input_tokens": 20, "output_tokens": 5, "reasoning_output_tokens": 2, "total_tokens": 55}}}),
                    event("2026-08-18T00:00:00Z", "event_msg", {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 999, "total_tokens": 999}}}),
                ],
            )
            report = codex_usage.analyze(
                Path(directory),
                datetime(2026, 8, 17, tzinfo=timezone.utc),
                datetime(2026, 8, 18, tzinfo=timezone.utc),
            )
            self.assertEqual(report["totals"]["calls"], 2)
            self.assertEqual(report["totals"]["total_tokens"], 165)
            self.assertEqual(report["by_model"]["gpt-sol"]["total_tokens"], 110)
            self.assertEqual(report["by_model"]["gpt-terra"]["total_tokens"], 55)
            self.assertEqual(report["by_agent_type"]["primary:interactive"]["calls"], 2)

    def test_subagent_role_and_cumulative_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_rollout(
                directory,
                [
                    event("2026-08-17T00:00:00Z", "session_meta", {"source": "subagent", "thread_source": {"sub_agent": {"agent_role": "researcher"}}}),
                    event("2026-08-17T00:00:01Z", "turn_context", {"model": "gpt-luna"}),
                    event("2026-08-16T23:59:59Z", "event_msg", {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23}}}),
                    event("2026-08-17T00:00:03Z", "event_msg", {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 50, "output_tokens": 7, "total_tokens": 57}}}),
                ],
            )
            report = codex_usage.analyze(
                Path(directory),
                datetime(2026, 8, 17, tzinfo=timezone.utc),
                datetime(2026, 8, 18, tzinfo=timezone.utc),
            )
            self.assertEqual(report["fallback_records"], 1)
            self.assertEqual(report["totals"]["total_tokens"], 34)
            self.assertEqual(report["by_agent_type"]["subagent:researcher"]["calls"], 1)

    def test_malformed_lines_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_rollout(directory, [])
            path.write_text("not-json\n")
            report = codex_usage.analyze(
                Path(directory),
                datetime(2026, 8, 17, tzinfo=timezone.utc),
                datetime(2026, 8, 18, tzinfo=timezone.utc),
            )
            self.assertEqual(report["malformed_lines"], 1)
            self.assertEqual(report["totals"]["calls"], 0)


if __name__ == "__main__":
    unittest.main()
