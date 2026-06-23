# Hardware Evaluation Pipeline (`tools/orchestrator/`)

Closes the loop between approximate-multiplier generation and CNN accuracy:
for every candidate multiplier circuit produced by **SubXPAT**, the
orchestrator synthesizes it, measures its hardware cost and numerical error,
and (if useful) retrains/evaluates a CNN with it — all concurrently, and all
logged to a single `results.csv`.

```
SubXPAT (main.py)              tools/orchestrator/npy_generator.py        src/cnn_training.py
   generates .v  ───────────►   synth (area/power/delay via OpenSTA)  ──►   trains/evaluates CNN
   multiplier circuits           + simulates the truth table → .npy         with the .npy table
   into output/ver/              + mean/max abs. error vs. exact mult.      as the multiplier LUT
                                  + writes one row to results.csv            (--conv_type)
```

`orchestrator.py` watches `output/ver/` for newly written `.v` files while
SubXPAT is still running, and schedules analysis + training for each one as
soon as it appears — it does not wait for the whole batch to finish.

---

## Prerequisites

| Requirement | Why |
|---|---|
| [SubXPAT](https://github.com/LP-RG/subxpat/tree/114-new-zone-constraint-with-re-metric) SubXPAT checked out at `<repo_root>/subxpat/` on branch `114-new-zone-constraint-with-re-metric`, with its own Makefile exposing a setup target | provides `main.py` and `sxpat.specifications.Specifications`, imported directly by `orchestrator.py` |
| Yosys + OpenSTA on `PATH` (or configured via `PATH_TO_LOCAL_OPEN_STA` in `tools/synthesis_npy_generation/circuits_analizer.py`) | used by `vpadanalyzer` (the `verilog-pad-analyzer` package) for area/power/delay synthesis |
| `syn_lib/nangate_45nm_typ.lib` | standard-cell library consumed by the OpenSTA backend |
| `heat_maps/npy_matrix/8bit_resnet_20/*.npy` | per-layer input co-occurrence histograms (see below); needed for the CNN-weighted error metric |

The orchestrator copies the entire `subxpat/` project into
`tools/orchestrator/experiments/<experiment-name>/` for each run and runs
`make setup` there if no `.venv` is found yet, so each experiment gets its
own isolated SubXPAT environment.

---

## Running

```bash
python3 tools/orchestrator/orchestrator.py \
    --experiment-name my_run \
    --conv-type 3 \
    --model-name resnet \
    --exact-accuracy 91 \
    <... SubXPAT-specific arguments ...>
```

| Flag | Default | Description |
|---|---|---|
| `--experiment-name` | *(required)* | Unique name for this run; used for log/output directory naming and as a suffix on generated `.npy` filenames. |
| `--conv-type` | `3` | Convolution/training stage passed to `src/cnn_training.py` for each generated multiplier. |
| `--model-name` | `resnet` | CNN model key passed to `src/cnn_training.py`. |
| `--exact-accuracy` | `None` | Known exact-model accuracy, forwarded to training as `--exact_accuracy`. |

Any unrecognized arguments are forwarded as-is to SubXPAT's own
`Specifications.parse_args()` and to `main.py` — see SubXPAT's documentation
for the full list. The orchestrator additionally requires the SubXPAT
benchmark name to encode the multiplier bit width as `..._i<N>_...`
(`bitwidth = N // 2`), since that value is needed before synthesis.

---

## Per-circuit pipeline (`npy_generator.py`)

For each new `<circuit>.v` file detected in SubXPAT's `output/ver/`, the
orchestrator runs:

```bash
python3 tools/orchestrator/npy_generator.py \
    <circuit>.v <bitwidth> <output>.npy --experiment-name my_run
```

which:

1. Generates a pure-Python `approx_mult(a, b)` function from the Verilog
   (`tools/synthesis_npy_generation/sub_xpat_circuits_generator.py`).
2. Brute-forces the full truth table to build the `.npy` lookup table used as
   the CNN's approximate multiplier, computing:
   - `mean_ae` — mean absolute error vs. exact multiplication, uniform over all input pairs.
   - `mean_ae_cnn` — the same, but weighted by the empirical input-pair
     probability distribution recorded in `heat_maps/npy_matrix/8bit_resnet_20/`
     (collected by `src/cnn_training.py --conv_type 5`). This is what actually
     correlates with CNN accuracy, since most multiplier inputs in a trained
     network are far from uniformly distributed.
   - `max_ae` — worst-case absolute error.
3. Synthesizes the circuit via `vpadanalyzer.synthesis.Synthesis` (Yosys + OpenSTA) to get `area`, `power`, `delay`, and derives `pda = area * power * delay`.
4. Appends one row to `results.csv` (columns:
   `file,area,power,delay,pda,mean_ae,mean_ae_cnn,max_ae,accuracy`). If a row
   with matching hardware/error metrics already has a recorded accuracy, that
   accuracy is reused and training is skipped.

If no matching row exists, the orchestrator then runs
`src/cnn_training.py --conv_type <conv-type> --model_name <model-name>
--input_path <output>.npy --bit_width <bitwidth>` and parses the
`FINAL_ACCURACY:<value>` line from its stdout to fill in the `accuracy`
column.

---

## Outputs

```
tools/orchestrator/experiments/
├── <experiment-name>/            ← copied SubXPAT project + its own .venv
└── npy_outputs/
    └── <circuit>_<experiment-name>.npy
tools/orchestrator/experiments/log/<experiment-name>/
├── subxpat.log
├── analyzer.log
└── training.log
results.csv                       ← written in the current working directory
```

> `results.csv` is written relative to the process's working directory —
> run the orchestrator from the repo root for a predictable location.

For standalone use without the live SubXPAT watcher (e.g. re-analyzing a
folder of already-generated `.v` files), see the lower-level functions in
`tools/synthesis_npy_generation/circuits_analizer.py`
(`analyze_multipliers`, `create_matrices`, `merge_results`, `write_csv`).
