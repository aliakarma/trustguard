# TrustGuard — Final Reported Results

This directory contains the **final results reported in the TrustGuard paper**
(AAAI 2027 submission), one JSON file per paper table or result section.
These files are the reference against which reproduction runs should be
compared. All learned-method cells are the mean ± sample standard deviation
over the five training seeds `{7, 42, 123, 777, 2024}`.

## File → paper table mapping

| File | Paper location | Reproduction script |
|------|----------------|---------------------|
| `task1_prediction.json` | Appendix Table "Task 1 risk prediction" (`tab:task1`) | `experiments/evaluate_prediction.py` |
| `task2_enforcement.json` | Main text Table "Task 2 enforcement quality" (`tab:task2`) | `experiments/evaluate_enforcement.py` |
| `task2_stress_tests.json` | §"Out-of-Generator and Low-Prevalence Stress Tests" + Appendix "Held-Out Generator (AASE-B)" | `experiments/stress_tests.py` |
| `ablations.json` | Appendix Table "Factorial, component, and null ablations" (`tab:ablation`) | `experiments/run_ablations.py` |
| `adversarial_robustness.json` | Appendix Table "Adversarial robustness" (`tab:adversarial`) | `experiments/adversarial_evaluation.py` |
| `temporal_holdout.json` | Appendix Table "Temporal hold-out" (`tab:temporal`) | `experiments/evaluate_temporal.py` |
| `sensitivity_analyses.json` | Appendix Tables `tab:tau`, `tab:lambda`, `tab:modality` + EMA-α prose | `experiments/sensitivity_analysis.py` |
| `constraint_dynamics.json` | Appendix Table "Constraint dynamics during training" (`tab:curves`) + per-category FRR (Fairness) | produced during training (`experiments/train_trustguard.py` logs) |
| `statistical_tests.json` | Appendix "Statistical Tests" (DeLong, bootstrap CIs) | `experiments/statistical_tests.py` |
| `pilot_summary.json` | §"Real-Device Pilot" | not re-runnable (14-day IRB-approved field study; see notes inside the file) |

## Metric definitions

- **AIPR** — Anomalous Invocation Prevention Rate: share of ground-truth
  anomalous permission invocations blocked or rate-limited before completion.
- **EPR** — Exfiltration Prevention Rate: share of taint-verified
  sensitive-flow events prevented.
- **AET-R** — median Anomalous Exposure Time reduction.
- **PRR** — Privacy Risk Reduction (TrustGuard's internal risk score;
  comparability only, cannot carry headline claims).
- **FRR** — False Revocation Rate, the *ratio* form of Eq. (3):
  E[Σ false revokes] / max(1, E[Σ revokes]) — budget ε_safe = 0.025.
- **FIR** — cost-weighted False Intervention Rate (Eq. 4), a diagnostic,
  not a trained constraint.

## Notes for reviewers

- Install-time methods (Android Static Policy, DREBIN, MaMaDroid, DexRay,
  MaskDroid) cannot act at runtime; their Task-2 cells are `null` (N/A).
- Headline Task-2 metrics (AIPR/EPR/AET-R) are measured against
  model-independent ground truth, not TrustGuard's internal risk score.
- The 72-hour simulation uses Δ = 60 s governance windows (4,320 steps)
  with N = 700 applications (500 benign / 200 malicious) unless a stress
  protocol states otherwise.
- The real-device pilot false-alert rate (3.4%) is a *different construct*
  from simulated FRR (human-adjudicated, notification-only mode).
