# TabGenBench

TabGenBench 0.3.0 evaluates single-table synthetic data independently of the generator.
It provides strict validity diagnostics, fidelity, binary prediction utility, proximity
diagnostics, compatible-run comparisons and offline HTML reports.

Python 3.11 is the tested baseline. The runner uses Unix file locks on macOS/Linux.
SDV and PyTorch are optional; Evaluation Only works without either package.

## Install

From this source directory:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
tabbench --help
```

For Gaussian Copula and CTGAN, install the optional dependency separately:

```sh
python -m pip install '.[sdv]'
```

Installing dependencies normally needs network access. Once inputs and dependencies
are available, evaluation and reporting work offline. The wheel contains the Python
package and public dataset provenance; the source archive also includes configurations,
artificial examples and tests. Use a source checkout for the supplied configuration files.

## Evaluate existing data

The included CSV fixtures are hand-written artificial examples, not benchmark outputs.
Run the complete example without downloading real data or installing SDV:

```sh
tabbench evaluate --config configs/examples/evaluate_only.yaml
```

The configuration supplies a complete DatasetSpec and separate train, validation, test
and synthetic files. CSV and Parquet are supported. For your own data, copy the example
configuration and update its paths and schema. Paths are relative to the configuration.
Only empty CSV fields encode missingness; invalid values are reported without repair.

Results are written to `runs/toy-evaluation/`. The command prints the run and report paths.
The offline report includes PNG plots, metric values, validity reasons, failures, warnings
and unavailable metrics. JSONL/Parquet retain standard records; reports also export CSV.
Outputs are immutable: choose a new `output` in the configuration for another evaluation.

A reference-only configuration can replace train/validation/test with `reference`.
Metrics lacking required inputs remain `not_applicable`, with a reason and a null value.

## Run, compare and report

```sh
# Artificial example; dry-run creates no run or cache and performs no model work.
tabbench run --config configs/examples/toy_run.yaml --dry-run
tabbench run --config configs/examples/toy_run.yaml

# Replace RUN_A and RUN_B with actual run paths printed by the command.
tabbench compare --runs RUN_A RUN_B --output comparison/
tabbench report --run RUN_A --output reports/redrawn/
```

Report generation reads persisted results. It does not train, sample or recompute metrics.
Comparisons reject incompatible datasets, splits, schemas, rules, metric versions or suites.
Each metric keeps its own direction and scale; there is no combined quality score.

## Public benchmark

```sh
# Explicit network preparation, using the fixed dataset objects.
tabbench datasets fetch
tabbench datasets verify

# Inspect 60 planned configurations without executing them.
tabbench run --config configs/experiments/standard_v1.yaml --dry-run

# Full benchmark; includes substantial CPU training and evaluation.
tabbench run --config configs/experiments/standard_v1.yaml
```

The fixed dataset objects are Adult (OpenML 1590 v2), Bank Marketing (1461 v1) and
Credit-G (31 v1). Four generators—Bootstrap, Independent Marginal, Gaussian Copula
and CTGAN—use five seeds on the same frozen 60/20/20 real split, with synthetic row
count equal to training row count. CTGAN uses 100 epochs and CPU execution.
The data and protocol locks reject changes instead of replacing them silently.

Raw data, generated tables, models and caches are not distributed. Dataset attribution:
[Adult — Becker & Kohavi](https://archive.ics.uci.edu/dataset/2/adult),
[Bank Marketing — Moro, Laureano & Cortez](https://archive.ics.uci.edu/dataset/222/bank+marketing),
[German Credit — Hofmann](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
Source citations and license notes are retained in the dataset configurations.

## Python API

```python
from tabgen_bench.interfaces.api import evaluate, run, compare, report

result = evaluate('configs/examples/evaluate_only.yaml')
plan = run('configs/examples/toy_run.yaml', dry_run=True)
redrawn = report(result['run'], output='reports/python-redraw')
# compare([first_run_path, second_run_path], output='comparison/python')
```

CLI and API use the same services. The context-based `evaluate_only` API and injected
metric registry remain available. Optional `dependency_lock` configuration can identify
an explicitly supplied dependency snapshot. Otherwise the installed Python/distribution
versions determine the dependency fingerprint. Changes invalidate affected caches;
this does not promise identical output across platforms or dependency versions.

## Metrics and limits

See [the benchmark and evaluation protocol](docs/protocol.md) for definitions and sampling.

| Area | Included |
| --- | --- |
| Validity | Schema Pass, Invalid Cell/Row Rate, per-column reasons |
| Fidelity | KS, TV, missingness, pairwise TV, held-out C2ST and real/real calibration |
| Utility | TRTR, TSTR and signed performance gap |
| Proximity | DCR train/holdout, train-closer/tie rates and exact train copy rate |

DCR and related proximity diagnostics do not provide a privacy guarantee. Advanced
Diversity, statistical inference and formal privacy attacks are not implemented.

Unique outputs do not prove independent repetitions. The pinned SDV API resets sampling
to a fixed stream; distinct requested seeds can produce identical datasets. Reports show
planned, attempted, completed and unique counts separately. Independence remains
unconfirmed unless established; no unconfirmed Monte Carlo standard errors or confidence
intervals are produced. Five seeds do not create five independent real-data splits.
Adult has duplicate content across frozen row partitions; the overlap warning is retained.

Fit caches store metadata, not serialized executable models. Missing sampled data may
require an explicit refit. Cache corruption is an error. Reports are static offline HTML;
cross-platform byte-for-byte reproducibility is not guaranteed.

## Verify locally

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m ruff check .
```

Tests use artificial inputs, block network access and check the eight-module import DAG.
No external automation service or real benchmark data is required for these checks.

## License

The project's own code and artificial fixtures use the [MIT License](LICENSE).
The package does not bundle third-party SDK code, fonts, model weights or dataset tables.
Its dependencies remain separately licensed and retain their installed license files.

Optional SDV 1.38.5, CTGAN 0.12.1, Copulas 0.14.1, RDT 1.22.0 and DeepEcho 0.8.1 use
BUSL-1.1 with component-specific additional grants. They are not relicensed by this
project's MIT License. Review the pinned [SDV license](https://raw.githubusercontent.com/sdv-dev/SDV/v1.38.5/LICENSE)
and [CTGAN license](https://raw.githubusercontent.com/sdv-dev/CTGAN/v0.12.1/LICENSE), and the
licenses supplied by the other installed components, before use or redistribution.
In particular, the grants restrict specified commercial synthetic-data or transformation
services. SDMetrics uses MIT; scientific computing, plotting and PyTorch dependencies
retain their own terms. A bundled distribution or hosted service needs its own review.
