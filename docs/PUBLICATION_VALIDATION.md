# Publication validation record

Date: 2026-09-28. This records checks performed while preparing the public repository.

| Check | Result |
| --- | --- |
| Source/evidence provenance | All 40 copied files match their source SHA-256 hashes |
| Python syntax | All published `presses/` and `tools/` Python files compile |
| Independent saved-prediction audit | 650 matched examples; 633 identical strings; 17 changed strings; 650 unchanged sample scores |
| Metrics reconciliation | All 13 task metrics match saved metrics; both macro scores are 36.39 |
| Timing reconciliation | Median prefill reductions are 4.72%, 4.52%, and 5.96% at 8K, 16K, and 32K |
| Offline CI command rehearsal | Recomputed audit JSON equals the checked-in audit |
| New documentation whitespace | Passed |
| Common credential-pattern scan | No matching GitHub/Hugging Face token or private-key header patterns in the selected public files |

The original Windows line endings and raw log whitespace are preserved to keep source hashes intact. This check record does not claim new GPU validation, a fresh dependency installation, or GitHub-hosted CI execution. The automated workflow performs CPU-only evidence auditing; historical GPU results remain scoped to the recorded experiment environment.
