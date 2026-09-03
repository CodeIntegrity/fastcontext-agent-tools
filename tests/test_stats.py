from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastcontext_mcp import server
from fastcontext_mcp.stats import (
    StatsStore,
    analyze_trajectory,
    estimate_tokens,
    format_stats_text,
)


class StatsTests(unittest.TestCase):
    def test_estimate_tokens(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        # 40 ASCII chars -> ~10 tokens
        ascii_text = "a" * 40
        self.assertEqual(estimate_tokens(ascii_text), 10)
        # 10 CJK chars -> 10 tokens
        cjk_text = "测试中文字符统计功能"
        self.assertEqual(estimate_tokens(cjk_text), 10)
        # Mixed: 20 ascii + 5 CJK -> 5 + 5 = 10
        mixed_text = "abcdefghijklmnopqrst" + "一二三四五"
        self.assertEqual(estimate_tokens(mixed_text), 10)

    def test_analyze_trajectory_with_mock_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            traj_file = Path(tmp_dir) / "trajectory.jsonl"
            lines = [
                {"role": "system", "content": "You are a code explorer."},
                {"role": "user", "content": "Find where auth token is verified."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Grep", "arguments": json.dumps({"pattern": "auth_token"})},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "Line 42: def verify_auth_token(token):\nLine 43:     return True\n" * 20,
                },
                {
                    "role": "assistant",
                    "content": "<final_answer>\nsrc/auth.py:42-50\n</final_answer>",
                },
            ]
            with traj_file.open("w", encoding="utf-8") as f:
                for item in lines:
                    f.write(json.dumps(item) + "\n")

            returned_answer = "<final_answer>\nsrc/auth.py:42-50\n</final_answer>"
            savings = analyze_trajectory(traj_file, returned_answer)

            self.assertEqual(savings["turns_used"], 2)
            self.assertEqual(savings["tool_calls_count"], 1)
            self.assertGreater(savings["raw_context_tokens"], savings["returned_tokens"])
            self.assertGreater(savings["tokens_saved"], 0)
            self.assertTrue(savings["compression_ratio"].endswith("%"))

    def test_analyze_trajectory_missing_file_fallback(self) -> None:
        savings = analyze_trajectory(Path("/nonexistent/file.jsonl"), "src/app.py:1-10")
        self.assertEqual(savings["turns_used"], 0)
        self.assertEqual(savings["tool_calls_count"], 0)
        self.assertEqual(savings["tokens_saved"], 0)
        self.assertEqual(savings["compression_ratio"], "0.0%")

    def test_stats_store_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_file = Path(tmp_dir) / "stats.json"
            store = StatsStore(stats_file)

            # Initial state
            initial = store.get_stats()
            self.assertEqual(initial["global"]["total_calls"], 0)

            # Record a successful call
            savings_1 = {
                "turns_used": 3,
                "tool_calls_count": 2,
                "raw_context_tokens": 1000,
                "returned_tokens": 50,
                "tokens_saved": 950,
            }
            store.record_call("/path/to/repo_a", True, savings_1)

            updated = store.get_stats()
            self.assertEqual(updated["global"]["total_calls"], 1)
            self.assertEqual(updated["global"]["successful_calls"], 1)
            self.assertEqual(updated["global"]["total_tokens_saved"], 950)
            self.assertIn("/path/to/repo_a", updated["repositories"])

            # Query by specific repo
            repo_a_stats = store.get_stats("/path/to/repo_a")
            self.assertEqual(repo_a_stats["repository"], "/path/to/repo_a")
            self.assertEqual(repo_a_stats["stats"]["total_calls"], 1)

            # Reset
            store.reset()
            after_reset = store.get_stats()
            self.assertEqual(after_reset["global"]["total_calls"], 0)
            self.assertEqual(len(after_reset["repositories"]), 0)

    def test_format_stats_text(self) -> None:
        stats = {
            "global": {
                "total_calls": 5,
                "successful_calls": 4,
                "failed_calls": 1,
                "total_turns": 15,
                "total_tool_calls": 8,
                "total_raw_tokens": 20000,
                "total_returned_tokens": 500,
                "total_tokens_saved": 19500,
                "overall_compression_ratio": "97.5%",
            },
            "repositories": {
                "/repo/foo": {
                    "total_calls": 3,
                    "total_tokens_saved": 12000,
                    "overall_compression_ratio": "98.0%",
                }
            },
        }
        text = format_stats_text(stats)
        self.assertIn("Total Calls:            5", text)
        self.assertIn("Successful:         4", text)
        self.assertIn("Tokens Saved:           19,500", text)
        self.assertIn("Overall Savings Ratio:  97.5%", text)
        self.assertIn("Repo: /repo/foo", text)

    def test_cli_stats_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_file = Path(tmp_dir) / "stats.json"
            with mock.patch.dict(os.environ, {"FASTCONTEXT_STATS_FILE": str(stats_file)}):
                # Test --stats-reset
                code = server.main(["--stats-reset"])
                self.assertEqual(code, 0)

                # Test --stats
                stdout = io.StringIO()
                with mock.patch("sys.stdout", stdout):
                    code = server.main(["--stats"])
                self.assertEqual(code, 0)
                self.assertIn("FastContext Usage & Savings Statistics", stdout.getvalue())

                # Test --stats-json
                stdout_json = io.StringIO()
                with mock.patch("sys.stdout", stdout_json):
                    code = server.main(["--stats-json"])
                self.assertEqual(code, 0)
                parsed = json.loads(stdout_json.getvalue())
                self.assertIn("global", parsed)


if __name__ == "__main__":
    unittest.main()
