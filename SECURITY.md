# Security Policy

This is a university prototype (AI + AR Sign Language Translator).

## Data and privacy
- Recognition runs locally. No image or video is sent anywhere.
- Only hand landmarks are used for training. Recorded clips stay on the local
  computer and are never committed (`*.mp4` is in `.gitignore`).
- Model files are not stored in Git. The loader verifies a SHA-256 hash before
  loading `rf_demo.joblib`, because joblib files can run code when loaded.

## Reporting a problem
Open a GitHub issue, or message the team. Do not post secrets or personal data.

## Known limitations
See `docs/threat-model.md` and `docs/security-traceability.md`.