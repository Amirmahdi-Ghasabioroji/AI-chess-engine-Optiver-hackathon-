"""Sequential 120s+0.5s round-robin of mix variants vs v2.1 and v2.3.

Each unordered pair plays 7 games (colour-alternating book tails). Pairings
run one after another, one game at a time. Re-running resumes from the jsonl.

    .\\.venv\\Scripts\\python.exe experiments\\run_mix_field_rr.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gauntlet import GameResult, _format_elo, elo, opening_positions, play_one  # noqa: E402

HERE = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / "local" / "engine_snapshots"
LOG_PATH = HERE / "logs" / "mix_field_rr_120s.jsonl"
SUMMARY_PATH = HERE / "logs" / "mix_field_rr_120s_summary.txt"

BASE_MS = 120_000
INCREMENT_MS = 500
GAMES_PER_PAIR = 7

AGENTS: list[tuple[str, Path]] = [
    ("v71111_n40", SNAPSHOTS / "v7.1.1.1_n40"),
    ("v71111_n70", SNAPSHOTS / "v7.1.1.1_n70"),
    ("v71111_n100", SNAPSHOTS / "v7.1.1.1_n100"),
    ("v711_n40", SNAPSHOTS / "v7.1.1_n40"),
    ("v711_n60", SNAPSHOTS / "v7.1.1"),
    ("v2.1", SNAPSHOTS / "self_v2.1"),
    ("v2.3", SNAPSHOTS / "v2.3_NNUE"),
]


def _jobs(positions: list[str]) -> list[tuple[int, str, bool]]:
    jobs = [
        (index, fen, agent_white)
        for index, fen in enumerate(positions)
        for agent_white in (True, False)
    ]
    return jobs[:GAMES_PER_PAIR]


def _standings(all_results: dict[tuple[str, str], list[GameResult]]) -> str:
    names = [name for name, _ in AGENTS]
    totals: dict[str, list[float]] = {name: [] for name in names}
    for (a, b), results in all_results.items():
        for r in results:
            totals[a].append(r.score)
            totals[b].append(1.0 - r.score)

    ranked: list[tuple[float, str, int, int, int, int]] = []
    for name in names:
        scores = totals[name]
        if not scores:
            continue
        wins = sum(1 for s in scores if s == 1.0)
        draws = sum(1 for s in scores if s == 0.5)
        losses = sum(1 for s in scores if s == 0.0)
        score = (wins + draws / 2) / len(scores)
        ranked.append((score, name, len(scores), wins, draws, losses))
    ranked.sort(reverse=True)

    lines = [
        "| Engine | Games | W | D | L | Score | Elo vs field |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for score, name, games, wins, draws, losses in ranked:
        lines.append(
            f"| {name} | {games} | {wins} | {draws} | {losses} | "
            f"{score:.1%} | {_format_elo(elo(score))} |"
        )
    return "\n".join(lines)


def _crosstable(all_results: dict[tuple[str, str], list[GameResult]]) -> str:
    names = [name for name, _ in AGENTS]
    header = "| | " + " | ".join(names) + " |"
    sep = "|---|" + "|".join("---" for _ in names) + "|"
    rows = [header, sep]
    lookup: dict[tuple[str, str], float] = {}
    for (a, b), results in all_results.items():
        if not results:
            continue
        score = sum(r.score for r in results) / len(results)
        lookup[(a, b)] = score
    for a in names:
        cells = [a]
        for b in names:
            if a == b:
                cells.append("—")
            elif (a, b) in lookup:
                cells.append(f"{lookup[(a, b)]:.0%}")
            elif (b, a) in lookup:
                cells.append(f"{1.0 - lookup[(b, a)]:.0%}")
            else:
                cells.append("")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _result_from_record(record: dict[str, object]) -> GameResult:
    return GameResult(
        opening=int(record["opening"]),
        agent_white=bool(record["agent_white"]),
        score=float(record["score"]),
        termination=str(record["termination"]),
        fen=str(record.get("fen") or ""),
        stderr=str(record.get("stderr") or ""),
        pgn=str(record.get("pgn") or ""),
    )


def _load_log() -> tuple[dict[tuple[str, str], list[GameResult]], set[tuple[str, str, int, bool]]]:
    all_results: dict[tuple[str, str], list[GameResult]] = {}
    done: set[tuple[str, str, int, bool]] = set()
    if not LOG_PATH.exists():
        return all_results, done
    with LOG_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name_a = str(record["agent"])
            name_b = str(record["opponent"])
            result = _result_from_record(record)
            all_results.setdefault((name_a, name_b), []).append(result)
            done.add((name_a, name_b, result.opening, result.agent_white))
    return all_results, done


def _write_summary(
    started: datetime,
    pair_i: int,
    n_pairs: int,
    all_results: dict[tuple[str, str], list[GameResult]],
) -> None:
    SUMMARY_PATH.write_text(
        f"started {started.isoformat(timespec='seconds')}\n"
        f"pairings reached {pair_i}/{n_pairs}\n\n"
        + _standings(all_results)
        + "\n\n"
        + _crosstable(all_results)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    positions = opening_positions()
    jobs = _jobs(positions)
    pairings = list(combinations(AGENTS, 2))
    all_results, done = _load_log()
    resumed = sum(len(v) for v in all_results.values())
    print(
        f"{len(pairings)} pairings × {len(jobs)} games, "
        f"{BASE_MS}ms+{INCREMENT_MS}ms, sequential"
        + (f", resuming with {resumed} games already logged" if resumed else ""),
        flush=True,
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    clock = (BASE_MS, INCREMENT_MS)

    with LOG_PATH.open("a", encoding="utf-8") as stream:
        for pair_i, ((name_a, path_a), (name_b, path_b)) in enumerate(pairings, start=1):
            pair_jobs = [
                (opening, fen, agent_white if pair_i % 2 == 1 else (not agent_white))
                for opening, fen, agent_white in jobs
            ]
            remaining = [
                job
                for job in pair_jobs
                if (name_a, name_b, job[0], job[2]) not in done
            ]
            if not remaining and (name_a, name_b) in all_results:
                print(
                    f"\n== pairing {pair_i}/{len(pairings)}: {name_a} vs {name_b} "
                    f"(already complete) ==",
                    flush=True,
                )
                _write_summary(started, pair_i, len(pairings), all_results)
                continue
            print(f"\n== pairing {pair_i}/{len(pairings)}: {name_a} vs {name_b} ==", flush=True)
            results = list(all_results.get((name_a, name_b), []))
            for done_i, (opening, fen, agent_white) in enumerate(pair_jobs, start=1):
                key = (name_a, name_b, opening, agent_white)
                if key in done:
                    print(
                        f"  game {done_i}/{len(pair_jobs)}: skipped (already logged)",
                        flush=True,
                    )
                    continue
                result = play_one(path_a, path_b, fen, opening, agent_white, clock)
                results.append(result)
                all_results[(name_a, name_b)] = results
                done.add(key)
                record = {
                    "agent": name_a,
                    "opponent": name_b,
                    **asdict(result),
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                colour = "white" if result.agent_white else "black"
                print(
                    f"  game {done_i}/{len(pair_jobs)}: {name_a} {colour} "
                    f"-> {result.score} by {result.termination}",
                    flush=True,
                )
            wins = sum(1 for r in results if r.score == 1.0)
            draws = sum(1 for r in results if r.score == 0.5)
            losses = sum(1 for r in results if r.score == 0.0)
            score = (wins + draws / 2) / len(results) if results else 0.0
            print(
                f"  {name_a} vs {name_b}: +{wins} ={draws} -{losses} ({score:.1%})",
                flush=True,
            )
            _write_summary(started, pair_i, len(pairings), all_results)

    print("\n# Final standings\n")
    print(_standings(all_results))
    print("\n# Crosstable (row score vs column)\n")
    print(_crosstable(all_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
