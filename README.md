# AI + AR Sign Language Translator

Real-time translation of sign language into English,
with an augmented reality overlay and speech synthesis.
Module: AI for Augmented Reality.

## Team
| Member | Role | Folders |
|---|---|---|
| 1 | AI / ML | models/ |
| 2 | Computer Vision | vision/ |
| 3 | Backend / Translation + Voice | translation/, voice/ |
| 4 | AR / UI | ui/ |
| 5 | Dataset / Testing / Docs | data/, tests/, docs/ |

## Contract between modules
| Module | Input | Output |
|---|---|---|
| vision | webcam image | normalized landmark values (2 hands) |
| models | landmark sequence | (sign, confidence) |
| translation | (sign, confidence) | validated word or nothing |
| voice | text | audio |
| ui | image + text | image with overlay |

## Working rules
- Never work directly on `main` or `develop`: one `feature/...` branch per task.
- Every merge goes through a Pull Request reviewed by a teammate.
- Everyone only modifies their own folders.
- No secrets, videos or heavy model files in the repository.