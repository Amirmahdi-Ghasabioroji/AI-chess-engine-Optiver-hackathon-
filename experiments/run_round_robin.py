"""Four-engine round-robin through the official harness referee.

Rated Chessathon games start from unpublished curated openings, so this driver
reuses `tools.gauntlet` colour-paired opening positions rather than
`harness.arena`'s fixed startpos. The protocol, init budget, ply cap, and
adjudication are unchanged. The clock is a compressed proxy of 120s+0.5s so
the whole field finishes in a 20-40 minute wall budget.

Run from the repo root:

    .\\.venv\\Scripts\\python.exe experiments\\run_round_robin.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gauntlet import (  # noqa: E402
    GameResult,
    _format_elo,
    elo,
    opening_positions,
    play_one,
    report,
)

HERE = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / "local" / "engine_snapshots"
RESULTS_PATH = HERE / "RESULTS.md"
LOG_DIR = HERE / "logs"

# Compressed clock. Official is 120_000 + 500. Init budget stays 60s (harness).
BASE_MS = 5_000
INCREMENT_MS = 50
OPENINGS = 4  # each played twice with colours swapped -> 8 games / pairing
WORKERS = 1  # sequential: honest 1-core wall time on this machine

AGENTS: list[tuple[str, Path]] = [
    ("self_v2", SNAPSHOTS / "self_v2"),
    ("self_v3", SNAPSHOTS / "self_v3"),
    ("frozen-classical", SNAPSHOTS / "frozen-classical"),
    ("latest", SNAPSHOTS / "latest"),
]


def _score_line(results: list[GameResult]) -> tuple[int, int, int, float]:
    wins = sum(1 for r in results if r.score == 1.0)
    draws = sum(1 for r in results if r.score == 0.5)
    losses = sum(1 for r in results if r.score == 0.0)
    games = max(len(results), 1)
    return wins, draws, losses, (wins + draws / 2) / games


def _crosstable(names: list[str], pairing_scores: dict[tuple[str, str], float]) -> str:
    header = "| | " + " | ".join(names) + " |"
    sep = "|---|" + "|".join("---" for _ in names) + "|"
    rows = [header, sep]
    for a in names:
        cells = [a]
        for b in names:
            if a == b:
                cells.append("—")
            else:
                key = (a, b) if (a, b) in pairing_scores else (b, a)
                score = pairing_scores.get(key)
                if score is None:
                    cells.append("")
                elif a == key[0]:
                    cells.append(f"{score:.1%}")
                else:
                    cells.append(f"{1.0 - score:.1%}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _write_results(
    started: datetime,
    elapsed_s: float,
    positions: list[str],
    pairing_rows: list[str],
    pairing_scores: dict[tuple[str, str], float],
    all_results: dict[tuple[str, str], list[GameResult]],
    done: bool,
) -> None:
    names = [name for name, _ in AGENTS]
    totals: dict[str, list[float]] = {name: [] for name in names}
    for (a, b), results in all_results.items():
        for r in results:
            totals[a].append(r.score)
            totals[b].append(1.0 - r.score)

    standings_lines = [
        "| Engine | Games | W | D | L | Score | Elo vs field |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
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
    for score, name, games, wins, draws, losses in ranked:
        standings_lines.append(
            f"| {name} | {games} | {wins} | {draws} | {losses} | "
            f"{score:.1%} | {_format_elo(elo(score))} |"
        )

    status = "complete" if done else "in progress"
    body = f"""# Experiment round-robin

Status: **{status}**
Started: {started.isoformat(timespec="seconds")}
Elapsed: {elapsed_s / 60:.1f} min

## Match conditions

Official Chessathon (canonical: https://aichessathon.com/docs/rules.md):
120 s + 0.5 s/move, 60 s init, ply cap 300, 1 core, FIDE draws, material
adjudication at 300 plies. Rated games start from unpublished curated openings.

This evaluation keeps the harness protocol, 60 s init, ply cap 300, and
sequential (1-core) games. The clock is compressed so four engines can finish
in a 20–40 minute wall budget:

| Setting | Official | This run |
|---|---|---|
| Base | 120 000 ms | {BASE_MS} ms |
| Increment | 500 ms | {INCREMENT_MS} ms |
| Init | 60 s | 60 s (harness, unchanged) |
| Ply cap | 300 | 300 |
| Openings | unpublished curated set | {OPENINGS} colour-paired book tails |
| Games / pairing | n/a | {OPENINGS * 2} |
| Pairings | n/a | 6 (round-robin) |
| Parallel games | 2 containers on the platform | 1 (this machine) |

Field:

| Label | Source |
|---|---|
| self_v2 | frozen `baselines/self_v2` (Stage 2 numba bitboard, PeSTO) |
| self_v3 | frozen `baselines/self_v3` (Stage 3, PeSTO + PST residual) |
| frozen-classical | `local/engine_snapshots/frozen-classical.zip` |
| latest | `local/engine_snapshots/latest.zip` |

Local Python is the repo `.venv` (not the platform's 3.12 image). Treat Elo
deltas as ordering evidence, not platform Elo.

## Standings

{chr(10).join(standings_lines) if ranked else "_No games finished yet._"}

## Crosstable (score from row engine)

{_crosstable(names, pairing_scores) if pairing_scores else "_No pairings finished yet._"}

## Pairings

{chr(10).join(pairing_rows) if pairing_rows else "_Waiting for first pairing._"}

## Openings used

"""
    for index, fen in enumerate(positions):
        body += f"{index}. `{fen}`\n"

    body += "\n## Per-game log\n\n"
    body += "Full JSONL: `experiments/logs/*.jsonl`\n\n"
    for (a, b), results in all_results.items():
        body += f"### {a} vs {b}\n\n"
        body += "| Opening | White | Result (from " + a + ") | Termination |\n"
        body += "|---:|---|---:|---|\n"
        for r in results:
            colour = "white" if r.agent_white else "black"
            body += f"| {r.opening} | {colour} | {r.score:g} | {r.termination} |\n"
        body += "\n"

    RESULTS_PATH.write_text(body, encoding="utf-8")


def main() -> int:
    for name, path in AGENTS:
        if not (path / "agent.py").is_file():
            print(f"missing {path / 'agent.py'}", file=sys.stderr)
            return 1

    positions = opening_positions()[:OPENINGS]
    if len(positions) < OPENINGS:
        print(f"only {len(positions)} openings available, need {OPENINGS}", file=sys.stderr)
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    wall0 = time.monotonic()
    pairing_rows: list[str] = []
    pairing_scores: dict[tuple[str, str], float] = {}
    all_results: dict[tuple[str, str], list[GameResult]] = {}

    pairings = list(combinations(AGENTS, 2))
    total_games = len(pairings) * OPENINGS * 2
    played = 0
    print(
        f"{len(pairings)} pairings, {OPENINGS} openings x 2 colours = {total_games} games, "
        f"{BASE_MS}ms+{INCREMENT_MS}ms, sequential",
        flush=True,
    )
    _write_results(
        started, 0.0, positions, pairing_rows, pairing_scores, all_results, done=False
    )

    clock = (BASE_MS, INCREMENT_MS)
    try:
        for (name_a, path_a), (name_b, path_b) in pairings:
            label = f"{name_a} vs {name_b}"
            log_path = LOG_DIR / f"{name_a}_vs_{name_b}.jsonl"
            results: list[GameResult] = []
            print(f"\n=== {label} ===", flush=True)
            with log_path.open("w", encoding="utf-8") as stream:
                for index, fen in enumerate(positions):
                    for agent_white in (True, False):
                        result = play_one(path_a, path_b, fen, index, agent_white, clock)
                        results.append(result)
                        stream.write(json.dumps(asdict(result)) + "\n")
                        stream.flush()
                        played += 1
                        colour = "white" if result.agent_white else "black"
                        print(
                            f"game {played}/{total_games}: {label} opening {index} as {colour} "
                            f"-> {result.score} by {result.termination}",
                            flush=True,
                        )
                        all_results[(name_a, name_b)] = results
                        _write_results(
                            started,
                            time.monotonic() - wall0,
                            positions,
                            pairing_rows,
                            pairing_scores,
                            all_results,
                            done=False,
                        )

            wins, draws, losses, score = _score_line(results)
            pairing_scores[(name_a, name_b)] = score
            pairing_rows.append(
                f"- **{label}**: +{wins} ={draws} -{losses}, score {score:.1%} "
                f"(from {name_a}; {OPENINGS * 2} games)"
            )
            report(results, OPENINGS, label)
            all_results[(name_a, name_b)] = results
            _write_results(
                started,
                time.monotonic() - wall0,
                positions,
                pairing_rows,
                pairing_scores,
                all_results,
                done=False,
            )
    except KeyboardInterrupt:
        print("\ninterrupted, writing what finished", flush=True)

    _write_results(
        started,
        time.monotonic() - wall0,
        positions,
        pairing_rows,
        pairing_scores,
        all_results,
        done=True,
    )
    print(f"\nresults written to {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
