# Data

Everything here was produced by one complete run of the notebook (exported from `MyDrive/bbnet_repro/` with
`scripts/export_artifacts_colab.py`).

| File | Content |
|---|---|
| `dataset.json` | 57,236 codes: `l, m, k, A, B, C, D` (exponent pairs), `fam` (generator family), `key` (symmetry-orbit id), `d` (label), `label_methods` (per-method results), `split` (train / val / test) |
| `bench.jsonl` | `[method|key, {d, valid, sec}]` for the six heuristics on the test set (Table 1) |
| `cert.jsonl` | `[key, {status, d, bound, sec}]` exact certification with CP-SAT (`OPTIMAL` = certified, `SKIPPED` = label d > 14) |
| `table7.jsonl` | reference distances and heuristic results for the 18 codes of Table 7 |
| `relabel_groups.json`, `relabel.jsonl` | label recomputation of notebook §13 (targets and per-method results) |
| `runs/**/history.json` | per-epoch training logs: main model, 29 ablation runs, 4 diagnostic variants |
| `runs/variants/*/test_eval.json` | test predictions of the diagnostic variants |

`.jsonl` files hold one `[key, value]` pair per line; when a key appears twice, the later line wins.

The trained main-model checkpoint (`best.pt`, 51 MB) is attached to the GitHub Release rather than committed.
Load it with the official `modeling_bbnet.py`: `python modeling_bbnet.py --depth 8 --width 256 --checkpoint best.pt`
prints 6 / 10 / 12 for the README examples.
