# Security Requirements Traceability (S1 to S10)

Owner: Member 5. Update this table every week.
Last updated: October 2026 (week 1)

Status values: **Done** (proven by a test or check), **Partly**, **To verify**
(needs a check we have not done yet), **Not implemented**.

| Rule | What is promised | Owner | How it is proven | Status |
|---|---|---|---|---|
| S1 | Invalid predictions are rejected (confidence outside 0 to 1, empty or malformed sign) | Member 3 | `tests/test_prediction.py`, `tests/test_engine.py`, `tests/test_security.py` (garbage input) | **Done** |
| S2 | Only signs from the official list are accepted (allowlist) | Member 3, tests by Member 5 | `test_unknown_sign_is_rejected_by_allowlist` in `tests/test_security.py`, `tests/test_allowlist.py` | **Done** |
| S3 | No image or video is recorded or sent; only landmarks are kept | Members 2, 4, 5 | Code review of `vision/`, dataset check, `data/README.md` policy | **To verify** (check `vision/record_videos.py`) |
| S4 | Model file hash is recorded at export and checked at load | Members 1, 3 | Integration test (hash mismatch is refused) | **Not implemented** |
| S5 | No system command is built from text | Member 3 | `test_injection_like_signs_are_rejected` (strict sign format), Bandit in CI, code review of `voice/` | **Partly** |
| S6 | Dependencies are pinned and free of known serious vulnerabilities | Member 3 | pip-audit in CI, Dependabot | **To verify** (check `.github/`) |
| S7 | No secret or personal data in the repository | Everyone | Secret detection (pre-commit and CI), `.gitignore` | **To verify** |
| S8 | Logs contain no images or raw landmarks | Everyone | Code review | **To verify** |
| S9 | Errors (camera, model, voice) are handled without a crash or technical details on screen | Members 2, 3, 4 | `test_burst_of_predictions_is_fast_and_bounded`, bounded memory test; integration tests still needed | **Partly** |
| S10 | A consent message is shown at startup | Member 4 | Interface review | **Not implemented** (`ui/` is empty) |

## Notes

- The tests are in the `tests/` folder and run with `pytest`.
- S2: the engine only accepts signs from its allowlist (the model words in the
  demos, HELLO/THANKS/YES in keyboard mode).
- Statuses marked "To verify" must be checked before the final delivery.