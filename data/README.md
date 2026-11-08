# Dataset

The model works from **landmarks** (numbers), never from images.

## What is used

- Language: American Sign Language (ASL), 40 words
- Each frame gives 67 landmarks: left hand (21), right hand (21) and 25 face and
  upper-body points (MediaPipe Holistic)
- The model reads the last 30 frames
- The final model is trained on 21 signers; evaluation used 4 unseen signers
  (about 68% accuracy)

## The 40 words

HELLO, BYE, THANKYOU, PLEASE, FINE, YES, NO, NOT, MINEMY, YOURSELF, HESHEIT,
WEUS, MOM, DAD, BOY, GIRL, GO, LIKE, HAVE, DRINK, SEE, WAIT, SLEEP, GIVE, WATER,
FOOD, HOME, BED, HAPPY, SAD, HUNGRY, THIRSTY, SICK, WHERE, WHO, WHY, NOW, LATER,
TOMORROW, YESTERDAY

The official list is `models/gru40.words.json`.

## Source

Source, license and number of samples: to be completed by Member 1.

## Privacy rules

- No image, name or personal data is stored with the samples
- Datasets, models and videos are never committed to Git (`.gitignore`)
- `vision/record_videos.py` is a dataset tool that writes video clips locally.
  It is not used by the demos
- Signers must agree before being recorded

## Integrity

Each model has a SHA-256 file in `models/` (for example `gru40.sha256`). The model
is refused if its hash does not match.

## Owner

Member 5 (Dataset / Testing / Documentation).