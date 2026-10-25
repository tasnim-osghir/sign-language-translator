# Threat Model (STRIDE)

Project: AI + AR Sign Language Translator
Method: STRIDE (Spoofing, Tampering, Repudiation, Information disclosure,
Denial of service, Elevation of privilege)
Owner: Member 5, reviewed with the team

## 1. What we protect (assets)
 
| Asset                      | Why it matters 
|                            |
| Camera stream              | Shows a person's face and body 
| Landmarks                  | Personal data; sign language can reveal a disability (GDPR) 
| Dataset                    | If poisoned, the model learns wrong signs 
| Model file                 | If replaced, the app can output anything 
| Source code and repository | Shared by 5 people; a mistake or a secret can leak 
| Dependencies (libraries)   | A vulnerable or malicious library is a way in 

## 2. Trust boundary

The translation engine (`translation/`) sits between the AI model and the rest of
the application. It does **not** blindly trust the model's output: every
prediction is validated before it can become a word.

Camera -> Vision -> AI model -> [ Translation engine ] -> Screen / Voice
                                  ^ trust boundary

## 3. Threats and countermeasures

| STRIDE | Threat in this project | Countermeasure | Rule | Status |

| Spoofing | Model file replaced by a fake one; an involuntary gesture triggers a word | SHA-256 hash of the model checked at load; confidence threshold; stability vote | S4 | Threshold done; hash and vote to do |
| Tampering | Model, dataset (poisoning) or a dependency modified | Hashes, Pull Request reviews, pinned dependencies, pip-audit | S4, S6 | Partly |
| Repudiation | Nobody knows who changed what | Git history, mandatory Pull Requests, protected branches | - | Git in use |
| Information disclosure | Video or landmarks leaked; secret in Git; verbose logs | Local processing only; secret detection; minimal logs; consent message | S3, S7, S8, S10 | To verify |
| Denial of service | Flood of predictions; voice spam; app blocked by speech | Bounded memory (max 50 words); cooldown; separate voice thread | S9 | Memory bound tested; cooldown to do |
| Elevation of privilege | Command injection through text; loading an untrusted `.pkl` file | Strict sign format (capital letters only); no shell commands; no unknown files loaded | S5 | Format tested |

## 4. Threats specific to AI

- **Adversarial examples:** a posture designed to fool the model into a wrong sign.
- **Dataset bias:** a model trained on few people performs badly on others.
- **False positives / negatives:** a wrong word can mislead the person reading it.
  This is a prototype, not a reliable interpreter.

## 5. Known gaps (honest list)

- Stability vote and cooldown are not implemented yet (week 2).
- No full penetration test (out of scope for this project).

## 6. Evidence

Automated tests: `tests/test_prediction.py`, `tests/test_engine.py`,
`tests/test_security.py`.