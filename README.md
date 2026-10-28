# AI + AR Sign Language Translator

A prototype that translates American Sign Language (ASL) signs into English in
real time from a webcam. It reads the signer's hands and upper body, recognizes
one of 40 words, shows the words as a live overlay on the camera image and
speaks each new word with an English voice. Module: AI for Augmented Reality.

## How it works

```
webcam frame
  -> MediaPipe Holistic landmarks   vision/landmarks.py   (2 hands + upper body, 67 points)
  -> GRU model (ONNX, 40 words)     models/predictor.py   (sign + confidence)
  -> stability vote                 translation/stabilizer.py
  -> translation engine             translation/engine.py (allowlist, threshold,
                                                           display lexicon)
  -> screen overlay + voice         ui/, demo.py          voice/speaker.py
```

- Each frame gives 67 landmarks: left hand (21), right hand (21) and the first
  25 MediaPipe Pose points (face and upper body).
- The predictor keeps the last 30 frames and turns them into features relative
  to the nose and the shoulder width.
- The stability vote accepts a sign only when it wins the vote in the last
  frames (default: 7 votes out of 10) above the confidence threshold.
- The translation engine accepts only words from the model word list
  (allowlist), drops a word that repeats the previous one, and keeps at most 50
  words in the sentence.
- After about 10 frames without any hand, the model buffer and the vote are
  reset.

Only landmarks are used. The demos do not save or send any image or video: the
camera frame is processed in memory and then dropped. Note that
`vision/record_videos.py` is a separate dataset tool that does write video
files; it is not used by the demos.

## The 40 words

From `models/gru40.words.json`:

| | | | | |
|---|---|---|---|---|
| HELLO | BYE | THANKYOU | PLEASE | FINE |
| YES | NO | NOT | MINEMY | YOURSELF |
| HESHEIT | WEUS | MOM | DAD | BOY |
| GIRL | GO | LIKE | HAVE | DRINK |
| SEE | WAIT | SLEEP | GIVE | WATER |
| FOOD | HOME | BED | HAPPY | SAD |
| HUNGRY | THIRSTY | SICK | WHERE | WHO |
| WHY | NOW | LATER | TOMORROW | YESTERDAY |

### Display lexicon

Some sign identifiers are shown and spoken differently
(`translation/lexicon.py`). All other words are shown as they are.

| Sign identifier | Shown as |
|---|---|
| THANKYOU | thank you |
| MINEMY | my |
| HESHEIT | he/she/it |
| WEUS | we |

The voice speaks the displayed text in lower case.

## Setup

Python 3.12 (the version used by CI). The code needs Python 3.10 or newer.

Windows (PowerShell), from the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model files (private, never committed)

Two model files are needed for the real pipeline. They are not in the
repository and must never be committed.

| File | Source | Used by |
|---|---|---|
| `holistic_landmarker.task` | Google MediaPipe, [official download](https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task) | `vision/landmarks.py` |
| `gru40.onnx` | Given by the ML member (Member 1) | `models/predictor.py` |

### Where to put them

By default both files are read from the same folder:

```
<home folder>\Projects\asl-data\holistic_landmarker.task
<home folder>\Projects\asl-data\gru40.onnx
```

On Windows this is `%USERPROFILE%\Projects\asl-data\`:

```powershell
New-Item -ItemType Directory -Force "$HOME\Projects\asl-data"
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" -OutFile "$HOME\Projects\asl-data\holistic_landmarker.task"
# Copy gru40.onnx (from the ML member) into the same folder.
```

### Environment variables

| Variable | Changes the path of | Default |
|---|---|---|
| `SIGN_HOLISTIC_MODEL` | `holistic_landmarker.task` | `~/Projects/asl-data/holistic_landmarker.task` |
| `SIGN_MODEL_PATH` | the sign model (`gru40.onnx`) | `~/Projects/asl-data/gru40.onnx` |

```powershell
$env:SIGN_HOLISTIC_MODEL = "D:\models\holistic_landmarker.task"
$env:SIGN_MODEL_PATH = "D:\models\gru40.onnx"
python demo_ui.py
```

These values last only for the current PowerShell window.

### SHA-256 check of the sign model

When the sign model is loaded, `models/predictor.py`:

1. reads the expected hash from `models/<model name without extension>.sha256`
   (for `gru40.onnx` this is `models/gru40.sha256`, which is committed);
2. computes the SHA-256 of the model file;
3. refuses to start with `model hash mismatch for gru40.onnx` if they differ.

The word list is read from `models/<model name>.words.json`
(`models/gru40.words.json`). If you point `SIGN_MODEL_PATH` to a file with
another name, the matching `.sha256` and `.words.json` files must exist in
`models/`.

Only the sign model is hash-checked; `holistic_landmarker.task` is not.

To verify your copy by hand, compare the output with `models/gru40.sha256`:

```powershell
Get-FileHash "$HOME\Projects\asl-data\gru40.onnx" -Algorithm SHA256
Get-Content models\gru40.sha256
```

```bash
sha256sum ~/Projects/asl-data/gru40.onnx
cat models/gru40.sha256
```

(`Get-FileHash` prints the hash in upper case; the comparison is the same.)

## Run

All commands are run from the project root with the venv active.

```powershell
python demo_ui.py              # full interface (overlay + buttons + voice)
python demo.py                 # simple OpenCV window (overlay + voice)
python demo.py --keyboard      # no models needed: keys 1, 2, 3 = HELLO, THANKS, YES
```

If a model file is missing or the hash does not match, the demos print
`Cannot load the models: ...` and suggest `--keyboard`.

### Options

Options of both `demo_ui.py` and `demo.py`:

| Option | Default | Meaning |
|---|---|---|
| `--camera` | `0` | camera index |
| `--threshold` | `0.6` | minimum confidence for a prediction to count |
| `--window` | `10` | frames in the vote window |
| `--min-votes` | `7` | votes needed in the window to accept a sign |

Only in `demo.py`:

| Option | Meaning |
|---|---|
| `--keyboard` | use the keyboard instead of the model (only `--camera` is used in this mode) |

Example:

```powershell
python demo_ui.py --camera 1 --threshold 0.7
```

### Keys

| Key | `demo_ui.py` | `demo.py` |
|---|---|---|
| `c` | clear the sentence | clear the sentence |
| `q` | quit | quit |
| `Esc` | quit | - |
| `r` | reset | - |
| `v` | voice on / off | - |
| `Space` | pause / start | - |
| `1`, `2`, `3` | - | HELLO, THANKS, YES (only with `--keyboard`) |

`demo_ui.py` also has clickable buttons: PAUSE/START, RESET, CLEAR, VOICE
ON/OFF, QUIT.

## Tips for a good demo

- Keep your face and both shoulders in the image. The model measures the hand
  position from the nose and the shoulder width; without them no sign is
  predicted.
- Use good, even light and a plain background so the hands are detected.
- Hold each sign for about 1 to 2 seconds, then lower your hands. The window
  shows `Hands: L R` (in `demo.py`) when hands are detected.
- The same word twice in a row is shown only once.

### English voice

The voice uses `pyttsx3` (offline, the Windows SAPI5 voices on Windows). It
runs in a background thread and never blocks the camera.

The code does not select a voice itself: `pyttsx3` uses the system default
voice. On a Windows installed in another language (for example French), the
English words are read with that language's voice. To get English speech,
install an English voice (Settings > Time & language > Speech) and make it the
default voice.

If `pyttsx3` or the audio device is missing, one warning is printed and the
demo continues without sound. The VOICE ON/OFF button (or `v`) turns speech
off.

## Security in short

- Allowlist: only the 40 model words can be shown or spoken; any other output
  counts as "no sign".
- Model hash check: the sign model is refused if its SHA-256 does not match the
  committed hash.
- No shell: no system command is run or built from text; the voice speaks only
  allowlisted text of at most 64 letters, digits, spaces, `/` or `-`.
- Input validation: predictions, sign identifiers (capital letters, digits,
  `_`), confidences, landmark arrays and options are checked for type and
  value.
- No data stored: the demos keep landmarks in memory only and store no image or
  video.

Details: [docs/threat-model.md](docs/threat-model.md) and
[docs/security-traceability.md](docs/security-traceability.md).

## Tests and CI

```powershell
pip install -r requirements-ci.txt
ruff check .
python -m pytest
```

The tests are in `tests/` and do not need the model files, a camera or an audio
device.

CI (`.github/workflows/ci.yml`) runs on every pull request and push to
`develop` and `main`, on Ubuntu with Python 3.12:

1. `pip install -r requirements-ci.txt`
2. `ruff check .`
3. `python -m pytest -v`

Rule: CI installs `requirements-ci.txt`, not `requirements.txt`. When you add a
package to `requirements.txt` and import it at the top of a module, also add it
to `requirements-ci.txt`, or import it lazily (inside the function that needs
it, as done for `pyttsx3`, `cv2` and `mediapipe`).

## Team workflow

| Member | Role | Folders |
|---|---|---|
| 1 | AI / ML | `models/` |
| 2 | Computer Vision | `vision/` |
| 3 | Backend / Translation + Voice | `translation/`, `voice/` |
| 4 | AR / UI | `ui/` |
| 5 | Dataset / Testing / Docs | `data/`, `tests/`, `docs/` |

- One branch per task (`feature/...`, `fix/...`, `docs/...`).
- Every change goes through a pull request with at least one review by a
  teammate (see `.github/pull_request_template.md`).
- Never push directly to `develop` or `main`.
- Everyone modifies only their own folders.
- No secrets, videos, datasets or model files in the repository.

## Known limits

- Accuracy is about 68% on 4 unseen signers with the 40-word model. The final
  model is trained on all 21 people, so it has no held-out score.
- Pointing signs such as GO, HESHEIT, GIVE and MINEMY are recognized less well.
- Only single words are recognized; there is no grammar or sentence model.
- The French Windows voice problem is solved by using an English voice (see
  [English voice](#english-voice)).
- This is a prototype, not a reliable interpreter.
