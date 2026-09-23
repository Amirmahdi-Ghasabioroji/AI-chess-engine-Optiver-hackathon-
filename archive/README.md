# archive/

Stage 1 Python PVS lives in `search.py`. The rated agent uses `engine/bb_search.py`.
This copy stays so `tools/search_bench.py` can still time the old search, and so
`baselines/self_v1` has a sibling implementation to read.

`_migrate_layout.py` is the one-shot move that put the bitboard modules in
`engine/`. It has already been run.
