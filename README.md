# AgriEdge

[![Code DOI](https://img.shields.io/badge/Code%20DOI-10.5281%2Fzenodo.21941210-1682D4)](https://doi.org/10.5281/zenodo.21941210)
[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.5281%2Fzenodo.21941319-1682D4)](https://doi.org/10.5281/zenodo.21941319)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Audit tooling and a leakage-free precision-agriculture benchmark for the
Edge-IIoTset dataset.

This repository accompanies the paper *"Provenance, Not Behaviour: A
Serialisation Artifact in Edge-IIoTset and a Leakage-Free Benchmark for
Precision-Agriculture Intrusion Detection"* (`paper/manuscript.md`).

## The short version

Edge-IIoTset ships a `Readme.txt` telling researchers to one-hot encode seven
categorical columns. Four of those columns recover the attack/normal label with
an accuracy of **1.0000** on their own — not through network behaviour, but
through the *spelling* of the placeholder written for an absent protocol field.
Normal-traffic and attack captures were parsed separately and concatenated, so
one branch wrote `0` and the other wrote `0.0`. Encoding turns that build
artifact into a feature, and file provenance becomes the label.

Under the distributed recipe, all six standard classifiers plus an MLP and a
1D-CNN reach 1.0000 ± 0.0000 accuracy across fifteen cross-validation folds. The
leak survives label, ordinal and frequency encoding identically.

## Check your own pipeline

If you have a model trained on Edge-IIoTset, this takes about a minute:

```python
from agriedge.audit.leakage import audit_columns, summarize

reports = audit_columns(your_frame, your_categorical_columns, "Attack_label")
print(summarize(reports))
```

Any column with `separation_rate` at or near 1.0 is a relabelling of your
target. A `provenance_marker` of `True` means the placeholder spelling alone
determines the label.

A second diagnostic needs no tooling at all: **add Gaussian naive Bayes to your
model suite.** If it scores near-perfectly, your features contain a direct label
encoding — it is too weak a model to do otherwise.

## Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

PyTorch is optional and needed only for the federated simulation and deep
baselines:

```bash
uv pip install --python .venv/bin/python 'agriedge[torch]'
```

Point `AGRIEDGE_DATASET_ROOT` at your copy of the Edge-IIoTset directory if it
is not adjacent to this repository.

## Experiments

Each script writes its own result tables to `results/`.

| Script | What it does |
|---|---|
| `01_leakage_audit.py` | Audits the recipe's columns for label leakage |
| `02_leaky_vs_clean.py` | Trains the model suite under both protocols |
| `03_build_agribench.py` | Builds the AgriEdge benchmark and re-audits it |
| `04_agriedge_centralized.py` | Centralised scores plus edge deployment cost |
| `05_federated.py` | FedAvg under three client constructions |
| `06_repeated_kfold.py` | Protocol comparison with 15-fold confidence intervals |
| `07_lodo_sweep.py` | Leave-one-device-out across all five devices |
| `08_deep_baselines.py` | MLP and 1D-CNN under both protocols |
| `09_encoding_agnostic.py` | Shows the leak survives every common encoding |

Run order: `01` → `02` → `03` (builds the benchmark other scripts consume) →
the rest in any order.

```bash
.venv/bin/python experiments/01_leakage_audit.py --subset ml
.venv/bin/python experiments/03_build_agribench.py
.venv/bin/python experiments/07_lodo_sweep.py --sample 400000
```

## The AgriEdge benchmark

`03_build_agribench.py` rebuilds a benchmark from the raw per-device captures,
because the curated subsets cannot support agricultural research: Modbus is
absent from them (0 of 157,800 ML rows) and per-device identity has been
stripped to a single surviving MQTT topic.

The rebuilt benchmark carries 1,276,122 rows across five agricultural devices
with full device attribution and 149,996 Modbus rows. Uniform parsing across
every capture makes the serialisation artifact structurally impossible rather
than merely corrected after the fact — re-auditing it finds no column
separating the classes above 0.0288.

Device attribution is what makes non-IID federated partitioning by farm
possible, and it is exactly what the curated subsets discard.

## Layout

```
src/agriedge/
  audit/leakage.py         separation rate, single-column accuracy, NMI, provenance probe
  data/placeholders.py     placeholder canonicalisation
  data/recipes.py          the two preprocessing protocols
  data/agribench.py        benchmark construction under uniform parsing
  data/textfeatures.py     structural features for high-cardinality strings
  models/zoo.py            classical model suite
  models/deep.py           MLP and 1D-CNN baselines
  federated/               non-IID partitioners and FedAvg
  evaluation/              metrics, repeated k-fold, edge cost, cross-domain alignment
experiments/               one script per experiment
notebooks/                 Colab driver for remote runs
paper/body.tex             the manuscript body, shared by both LaTeX versions
paper/abstract.tex         the abstract, shared by both
paper/preamble-common.tex  packages and macros, shared by both
paper/manuscript.tex       Elsevier wrapper (elsarticle, single column)
paper/manuscript-ieee.tex  IEEE wrapper (IEEEtran, two column)
paper/refs.bib             bibliography
paper/manuscript.md        the same manuscript in Markdown
paper/build_html.py        Markdown -> self-contained HTML preview
```

## Building the paper

```bash
cd paper && make
```

That produces both versions: `manuscript.pdf` (Elsevier, 27 pages) and
`manuscript-ieee.pdf` (IEEE, 11 pages). `make elsevier` and `make ieee` build
one at a time; `make html` regenerates the HTML preview from `manuscript.md`.
`make arxiv` packages the Elsevier version for arXiv submission.

The two `.tex` files are thin class wrappers over the same `body.tex`,
`abstract.tex` and `preamble-common.tex`, so they cannot drift apart. **Edit
the shared files, not the wrappers.** The only content difference the wrappers
introduce is float width: tables use `\wtable` / `\ntable`, which each wrapper
binds to `table*` or `table` as its column layout requires.

Building needs [tectonic](https://tectonic-typesetting.github.io)
(`brew install tectonic`), which fetches LaTeX packages on demand — the first
build needs network access, later ones do not.

One caveat on the IEEE build: it emits several `Font shape TU/ptm/... undefined`
warnings. They are harmless — IEEEtran declares its own Times defaults before
the preamble runs, and `\setmainfont` overrides them afterwards. What matters
is the embedded fonts, which `make fonts` prints; they should all be
TeXGyreTermes (a metric-compatible Times clone), never Latin Modern.

## Reproducibility

All randomness is seeded (`RANDOM_SEED = 20260814`). Every result in the paper
was produced on a 12-core Apple M4 Pro with 25 GB of memory; the deep baselines
used the integrated GPU via PyTorch's Metal backend and trained in under a
minute each. No experiment requires datacentre hardware.

## Getting the benchmark

The rebuilt AgriEdge benchmark (1,276,122 rows, 38 MB Parquet) is archived
separately at **doi:10.5281/zenodo.21941319**, so you can download it directly
without first obtaining the 10 GB Edge-IIoTset distribution. To regenerate it
from source instead, run `python experiments/03_build_agribench.py` against a
local copy.

## Citing this work

Two DOIs, both resolving to the newest version:

| | DOI |
|---|---|
| Code | `10.5281/zenodo.21941210` |
| Benchmark | `10.5281/zenodo.21941319` |

To pin exact releases instead, use `10.5281/zenodo.21941211` (code v1.0.0) and
`10.5281/zenodo.21941320` (benchmark v1.0.0).

GitHub renders a formatted citation from [`CITATION.cff`](CITATION.cff) under
*Cite this repository* in the sidebar.

## Citing the dataset

Edge-IIoTset is by Ferrag, Friha, Hamouda, Maglaras and Janicke (*IEEE Access*,
2022). Free for academic use per its own licence. Nothing here is a criticism of
the effort that produced it — building a physical IIoT testbed is hard, and the
artifact we document is a serialisation detail invisible under default pandas
type inference.
