# PermissionBench Dataset Description

## Overview

PermissionBench is the first large-scale benchmark for mobile permission risk
analysis with longitudinal runtime traces.

| Property | Value |
|----------|-------|
| Total records | 87,412 |
| Benign apps | 61,840 |
| Malicious apps | 14,512 |
| Permission classes | 42 (all Android API-34 dangerous permissions) |
| App categories | 33 Google Play categories |
| Annotation | Binary + per-permission risk labels |
| Runtime traces | Per-permission invocation counts (500 UI-automator events) |
| License | CC-BY-4.0 |

---

## Data Sources

| Source | Type | Count | Notes |
|--------|------|-------|-------|
| AndroZoo | Benign | 61,840 | Uniform category sampling, 2018–2024 APKs |
| Drebin | Malicious | 8,560 | Classic malware benchmark (Arp et al., 2014) |
| MalDroid-2020 | Malicious | 5,952 | Recent malware corpus (Mahdavifar et al., 2020) |

---

## Schema

Each record in `permissionbench_*.parquet` has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `app_id` | str | Unique identifier (package name + version) |
| `category` | str | Google Play category |
| `description` | str | Cleaned app-store description (≤2000 chars) |
| `permissions` | JSON str | List of declared dangerous permission names |
| `api_features` | str | Space-separated API method short names |
| `risk_label` | int | 0 = benign, 1 = malicious |
| `perm_labels` | JSON str | 42-float list: per-permission risk (0=legit, 1=anomalous) |
| `runtime_trace` | JSON str | 42-float list: per-permission invocation counts |
| `taint_flags` | JSON str | 42-float list: taint-tracking anomaly flags |
| `source` | str | "androzoo", "drebin", or "maldroid2020" |
| `sha256` | str | APK SHA-256 hash |
| `needs_human_review` | bool | Stage 3 ambiguous annotation flag |

---

## Annotation Protocol

### Stage 1 — Automated (TaintDroid-style)
Permissions that transmit sensitive data outside declared endpoints during
sandbox execution are labelled `perm_labels[p] = 1.0`.

Coverage: ~34% of malicious permission usages flagged automatically.

### Stage 2 — Rule-based threshold
Using predicted probabilities p̂ᵢ,ₚ from the trained `PermissionPredictionModel`:

- `p̂ < 0.05` AND permission declared → labelled anomalous (1.0)
- `p̂ > 0.70` AND permission declared → labelled legitimate (0.0)

Coverage: ~58% of records resolved at this stage.

### Stage 3 — Human review
The remaining ~8% of ambiguous records (0.05 ≤ p̂ ≤ 0.70) are reviewed by
two annotators with Cohen's κ = 0.84 inter-annotator agreement.

---

## Class Distribution

| Split | Benign | Malicious | Imbalance Ratio |
|-------|--------|-----------|-----------------|
| Train | 43,288 | 10,158 | 4.3:1 |
| Val | 6,184 | 1,451 | 4.3:1 |
| Test | 12,368 | 2,903 | 4.3:1 |

We recommend using `balance_classes=True` (WeightedRandomSampler) during
training to address the 4.3:1 class imbalance.

---

## Splits

Pre-split Parquet files are available:

```
data/permissionbench/
  train.parquet      — 53,446 records
  val.parquet        —  7,635 records
  test.parquet       — 15,271 records
  dataset_card.json  — summary statistics
```

---

## Extending the Dataset

To add new records, use the `DatasetBuilder` API:

```python
from trustguard.dataset import DatasetBuilder, AppRecord

builder = DatasetBuilder("data/permissionbench")
builder.add_record(AppRecord(
    app_id="com.example.app",
    category="TOOLS",
    description="A utility app...",
    permissions=["CAMERA", "READ_CONTACTS"],
    api_calls=["android.hardware.Camera.open()", ...],
    risk_label=0,
))
df_train, df_val, df_test = builder.finalise()
```

---

## Citation

```bibtex
@dataset{akarma2025permissionbench,
  title     = {{PermissionBench}: A Large-Scale Benchmark for Mobile
               Permission Risk Analysis},
  author    = {Akarma, Ali and Jan, Salman and Syed, Toqeer Ali},
  year      = {2025},
  url       = {https://github.com/Ali-Akarma/PermissionBench},
  license   = {CC-BY-4.0},
}
```
