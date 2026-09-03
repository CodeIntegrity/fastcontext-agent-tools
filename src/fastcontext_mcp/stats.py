from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_STATS_PATH = Path.home() / ".fastcontext" / "stats.json"
CHARS_PER_TOKEN_ASCII = 4.0
CJK_UNASSIGNED_RATIO = 1.0


def get_stats_file_path() -> Path:
    env_path = os.environ.get("FASTCONTEXT_STATS_FILE")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_STATS_PATH.expanduser().resolve()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Count CJK characters vs other characters
    cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    other_count = len(text) - cjk_count
    estimated = cjk_count * CJK_UNASSIGNED_RATIO + (other_count / CHARS_PER_TOKEN_ASCII)
    return max(1, int(round(estimated)))


def _extract_message_tokens(data: dict[str, Any]) -> int:
    usage = data.get("usage")
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int) and total > 0:
            return total

    text_parts: list[str] = []
    content = data.get("content")
    if isinstance(content, str):
        text_parts.append(content)

    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                if isinstance(fn, dict):
                    text_parts.append(str(fn.get("name", "")))
                    text_parts.append(str(fn.get("arguments", "")))

    return estimate_tokens("\n".join(text_parts))


def analyze_trajectory(trajectory_file: Path | str, returned_text: str) -> dict[str, Any]:
    traj_path = Path(trajectory_file).expanduser().resolve()
    raw_tokens = 0
    turns_used = 0
    tool_calls_count = 0

    if traj_path.exists() and traj_path.is_file():
        try:
            with traj_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(data, dict):
                        role = data.get("role")
                        if role == "assistant":
                            turns_used += 1
                            tcs = data.get("tool_calls")
                            if isinstance(tcs, list):
                                tool_calls_count += len(tcs)
                        raw_tokens += _extract_message_tokens(data)
        except OSError:
            pass

    returned_tokens = estimate_tokens(returned_text)
    # If trajectory couldn't be read or was empty, ensure raw_tokens >= returned_tokens
    raw_tokens = max(raw_tokens, returned_tokens)
    tokens_saved = max(0, raw_tokens - returned_tokens)
    ratio = (tokens_saved / raw_tokens) if raw_tokens > 0 else 0.0

    return {
        "turns_used": turns_used,
        "tool_calls_count": tool_calls_count,
        "raw_context_tokens": raw_tokens,
        "returned_tokens": returned_tokens,
        "tokens_saved": tokens_saved,
        "compression_ratio": f"{ratio * 100:.1f}%",
    }


@dataclass
class SummaryStats:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_raw_tokens: int = 0
    total_returned_tokens: int = 0
    total_tokens_saved: int = 0
    total_turns: int = 0
    total_tool_calls: int = 0

    def record(self, ok: bool, savings: dict[str, Any]) -> None:
        self.total_calls += 1
        if ok:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_raw_tokens += int(savings.get("raw_context_tokens", 0))
        self.total_returned_tokens += int(savings.get("returned_tokens", 0))
        self.total_tokens_saved += int(savings.get("tokens_saved", 0))
        self.total_turns += int(savings.get("turns_used", 0))
        self.total_tool_calls += int(savings.get("tool_calls_count", 0))

    def to_dict(self) -> dict[str, Any]:
        ratio = (
            (self.total_tokens_saved / self.total_raw_tokens)
            if self.total_raw_tokens > 0
            else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_raw_tokens": self.total_raw_tokens,
            "total_returned_tokens": self.total_returned_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "overall_compression_ratio": f"{ratio * 100:.1f}%",
        }


class StatsStore:
    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or get_stats_file_path()

    def _load_data(self) -> dict[str, Any]:
        if not self.storage_path.exists():
            return {"global": asdict(SummaryStats()), "repositories": {}}
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"global": asdict(SummaryStats()), "repositories": {}}

    def _save_data(self, data: dict[str, Any]) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.storage_path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.storage_path)
        except OSError:
            pass

    def record_call(self, repo_path: str, ok: bool, savings: dict[str, Any]) -> None:
        data = self._load_data()
        global_stats = SummaryStats(**data.get("global", {}))
        global_stats.record(ok, savings)
        data["global"] = asdict(global_stats)

        repos = data.setdefault("repositories", {})
        repo_data = repos.get(repo_path, {})
        repo_stats = SummaryStats(**repo_data)
        repo_stats.record(ok, savings)
        repos[repo_path] = asdict(repo_stats)

        self._save_data(data)

    def get_stats(self, repo_path: str | None = None) -> dict[str, Any]:
        data = self._load_data()
        global_stats = SummaryStats(**data.get("global", {})).to_dict()
        repos = data.get("repositories", {})

        if repo_path:
            specific_data = repos.get(repo_path)
            if specific_data:
                return {
                    "repository": repo_path,
                    "stats": SummaryStats(**specific_data).to_dict(),
                    "global_stats": global_stats,
                }
            return {
                "repository": repo_path,
                "stats": SummaryStats().to_dict(),
                "global_stats": global_stats,
            }

        repo_summaries: dict[str, Any] = {}
        for path_str, s_data in repos.items():
            repo_summaries[path_str] = SummaryStats(**s_data).to_dict()

        return {
            "global": global_stats,
            "repositories": repo_summaries,
        }

    def reset(self) -> None:
        empty = {"global": asdict(SummaryStats()), "repositories": {}}
        self._save_data(empty)


def format_stats_text(stats: dict[str, Any]) -> str:
    lines: list[str] = ["=== FastContext Usage & Savings Statistics ==="]
    global_stats = stats.get("global", stats.get("stats", {}))
    lines.append(f"Total Calls:            {global_stats.get('total_calls', 0)}")
    lines.append(f"  - Successful:         {global_stats.get('successful_calls', 0)}")
    lines.append(f"  - Failed:             {global_stats.get('failed_calls', 0)}")
    lines.append(f"Total Turns:            {global_stats.get('total_turns', 0)}")
    lines.append(f"Total Tool Calls:       {global_stats.get('total_tool_calls', 0)}")
    lines.append(f"Raw Context Tokens:     {global_stats.get('total_raw_tokens', 0):,}")
    lines.append(f"Returned Tokens:        {global_stats.get('total_returned_tokens', 0):,}")
    lines.append(f"Tokens Saved:           {global_stats.get('total_tokens_saved', 0):,}")
    lines.append(f"Overall Savings Ratio:  {global_stats.get('overall_compression_ratio', '0.0%')}")

    repos = stats.get("repositories")
    if isinstance(repos, dict) and repos:
        lines.append("\n--- Per-Repository Breakdown ---")
        for repo, r_stats in repos.items():
            lines.append(f"\nRepo: {repo}")
            lines.append(f"  Calls: {r_stats.get('total_calls', 0)} (saved {r_stats.get('total_tokens_saved', 0):,} tokens, {r_stats.get('overall_compression_ratio', '0.0%')})")

    return "\n".join(lines)
