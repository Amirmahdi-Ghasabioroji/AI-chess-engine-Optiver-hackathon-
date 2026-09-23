# Experiment round-robin

Status: **complete**
Started: 2026-09-02T19:28:42+00:00
Elapsed: 25.8 min

## Match conditions

Official Chessathon (canonical: https://aichessathon.com/docs/rules.md):
120 s + 0.5 s/move, 60 s init, ply cap 300, 1 core, FIDE draws, material
adjudication at 300 plies. Rated games start from unpublished curated openings.

This evaluation keeps the harness protocol, 60 s init, ply cap 300, and
sequential (1-core) games. The clock is compressed so four engines can finish
in a 20–40 minute wall budget:

| Setting | Official | This run |
|---|---|---|
| Base | 120 000 ms | 5000 ms |
| Increment | 500 ms | 50 ms |
| Init | 60 s | 60 s (harness, unchanged) |
| Ply cap | 300 | 300 |
| Openings | unpublished curated set | 4 colour-paired book tails |
| Games / pairing | n/a | 8 |
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

| Engine | Games | W | D | L | Score | Elo vs field |
|---|---:|---:|---:|---:|---:|---:|
| self_v2 | 24 | 22 | 1 | 1 | 93.8% | +470 |
| self_v3 | 24 | 17 | 1 | 6 | 72.9% | +172 |
| frozen-classical | 24 | 2 | 5 | 17 | 18.8% | -255 |
| latest | 24 | 1 | 5 | 18 | 14.6% | -307 |

## Crosstable (score from row engine)

| | self_v2 | self_v3 | frozen-classical | latest |
|---|---|---|---|---|
| self_v2 | — | 81.2% | 100.0% | 100.0% |
| self_v3 | 18.8% | — | 100.0% | 100.0% |
| frozen-classical | 0.0% | 0.0% | — | 56.2% |
| latest | 0.0% | 0.0% | 43.8% | — |

## Pairings

- **self_v2 vs self_v3**: +6 =1 -1, score 81.2% (from self_v2; 8 games)
- **self_v2 vs frozen-classical**: +8 =0 -0, score 100.0% (from self_v2; 8 games)
- **self_v2 vs latest**: +8 =0 -0, score 100.0% (from self_v2; 8 games)
- **self_v3 vs frozen-classical**: +8 =0 -0, score 100.0% (from self_v3; 8 games)
- **self_v3 vs latest**: +8 =0 -0, score 100.0% (from self_v3; 8 games)
- **frozen-classical vs latest**: +2 =5 -1, score 56.2% (from frozen-classical; 8 games)

## Openings used

0. `r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 1 9`
1. `r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 2 8`
2. `r1bqk2r/pppp1ppp/2n2n2/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5`
3. `r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6`

## Per-game log

Full JSONL: `experiments/logs/*.jsonl`

### self_v2 vs self_v3

| Opening | White | Result (from self_v2) | Termination |
|---:|---|---:|---|
| 0 | white | 0.5 | threefold_repetition |
| 0 | black | 1 | checkmate |
| 1 | white | 1 | checkmate |
| 1 | black | 1 | checkmate |
| 2 | white | 1 | checkmate |
| 2 | black | 1 | checkmate |
| 3 | white | 1 | checkmate |
| 3 | black | 0 | checkmate |

### self_v2 vs frozen-classical

| Opening | White | Result (from self_v2) | Termination |
|---:|---|---:|---|
| 0 | white | 1 | checkmate |
| 0 | black | 1 | checkmate |
| 1 | white | 1 | checkmate |
| 1 | black | 1 | checkmate |
| 2 | white | 1 | checkmate |
| 2 | black | 1 | checkmate |
| 3 | white | 1 | checkmate |
| 3 | black | 1 | checkmate |

### self_v2 vs latest

| Opening | White | Result (from self_v2) | Termination |
|---:|---|---:|---|
| 0 | white | 1 | checkmate |
| 0 | black | 1 | checkmate |
| 1 | white | 1 | checkmate |
| 1 | black | 1 | checkmate |
| 2 | white | 1 | checkmate |
| 2 | black | 1 | checkmate |
| 3 | white | 1 | checkmate |
| 3 | black | 1 | checkmate |

### self_v3 vs frozen-classical

| Opening | White | Result (from self_v3) | Termination |
|---:|---|---:|---|
| 0 | white | 1 | checkmate |
| 0 | black | 1 | checkmate |
| 1 | white | 1 | checkmate |
| 1 | black | 1 | checkmate |
| 2 | white | 1 | checkmate |
| 2 | black | 1 | checkmate |
| 3 | white | 1 | checkmate |
| 3 | black | 1 | checkmate |

### self_v3 vs latest

| Opening | White | Result (from self_v3) | Termination |
|---:|---|---:|---|
| 0 | white | 1 | checkmate |
| 0 | black | 1 | checkmate |
| 1 | white | 1 | checkmate |
| 1 | black | 1 | checkmate |
| 2 | white | 1 | checkmate |
| 2 | black | 1 | checkmate |
| 3 | white | 1 | checkmate |
| 3 | black | 1 | checkmate |

### frozen-classical vs latest

| Opening | White | Result (from frozen-classical) | Termination |
|---:|---|---:|---|
| 0 | white | 0.5 | threefold_repetition |
| 0 | black | 0.5 | threefold_repetition |
| 1 | white | 0 | checkmate |
| 1 | black | 1 | checkmate |
| 2 | white | 0.5 | insufficient_material |
| 2 | black | 1 | checkmate |
| 3 | white | 0.5 | threefold_repetition |
| 3 | black | 0.5 | threefold_repetition |

