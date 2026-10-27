# Security Requirements Traceability (S1 to S10)

Owner: Member 5. Last updated: October 2026.

Status values: **Done** (proven by a test or check), **Partly**, **To verify**,
**Not implemented**.

| Rule | What is promised | How it is proven | Status |
|---|---|---|---|
| S1 | Invalid predictions are rejected | `test_prediction.py`, `test_engine.py`, `test_security.py` | **Done** |
| S2 | Only signs from the allowlist are accepted | `test_allowlist.py` (36 tests: unknown words rejected, list frozen, lexicon cannot bypass it) | **Done** (engine level) |
| S3 | No image or video recorded or sent | The live app stores nothing. `record_videos.py` saves clips locally; `*.mp4` is blocked by `.gitignore` (rule line 229). Signer consent and deletion of clips still to confirm | **Partly** |
| S4 | Model hash verified before loading | `models/predictor.py` `_load()` raises an error on mismatch; `test_model_integrity.py` | **Done** |
| S5 | No system command built from text | No `subprocess` / `os.system` / `shell=True` in project code; `test_voice.py` forbids them; injection-like signs rejected in `test_security.py` | **Done** |
| S6 | Dependencies pinned and free of known vulnerabilities | `requirements.txt` fixed (joblib typo). `pip-audit` is not in the CI | **Not implemented** |
| S7 | No secret or personal data in the repository | `.gitignore` in place. No secret scanner in the CI | **To verify** |
| S8 | Logs contain no images or raw landmarks | Code review needed | **To verify** |
| S9 | Errors handled without crash | Bounded memory and 50,000-prediction burst test. Camera and model failure handling not fully verified | **Partly** |
| S10 | Consent message at startup | No consent text found in `ui/`, `demo.py`, `demo_ui.py` | **Not implemented** |

## Residual risks

- Model hash check: the file is read twice (hash, then load), and the expected hash
  is stored in the repo next to the code. It protects against a swapped model file,
  not against someone who can change both.
- The CI runs ruff and pytest only (no Bandit, pip-audit or secret scan).
- No full penetration test.