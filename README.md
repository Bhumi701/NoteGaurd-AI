# NoteGuard AI — Counterfeit Currency Identification Agent

NoteGuard AI checks whether an Indian currency note is **Genuine**, **Suspicious**, or **Fake** from a single photo. It combines a custom-trained YOLO26n object detection model (hosted on Roboflow) with a lightweight, explainable rule-based verdict layer built on top of the raw detections.

No online payment app today lets a user verify note authenticity in real time — NoteGuard AI is a step toward that: upload a photo, get an instant, explainable verdict.

## How it works

1. **Detection** — The trained YOLO26n model detects 6 key security features on the note:
   - `gandhi_portrait`
   - `identification-mark`
   - `lines`
   - `security-thread`
   - `serial_panel-1`
   - `serial_panel-2`

   Each detection comes with a confidence score. (`Indian-Currency_Note` is also detected but is just the note's boundary box, not a security feature.)

2. **Verdict layer** — The model alone only localizes features; it doesn't output genuine/fake on its own. This script adds that logic:
   - **GENUINE** — 5 or 6 of the 6 features detected above a 50% confidence threshold
   - **SUSPICIOUS** — 3 or 4 of the 6 features pass
   - **FAKE** — 2 or fewer features pass

   The 50% threshold and tier cutoffs were tuned against real test photos — a stricter 70% threshold flagged genuine notes as suspicious, since real-world lighting/angle naturally pushes 1–2 features below a high bar even on authentic notes.

## Model training journey

- Started with a multi-agent pipeline (pattern, OCR, texture, thread, fusion agents) — accuracy fluctuated around 56–59%.
- Pivoted to a single YOLO26n detector. First attempt: 51 annotated images on Roboflow → only 19% accuracy, weak precision/recall/F1.
- Used Roboflow's augmentation tool to expand the dataset to 152 images and retrained → **mAP@50 72.1%, precision 64.6%, recall 71.1%, F1-score 67.7%** on the held-out validation set.
- The underlying dataset ("Indian Currency Notes — Raw & Augmented") is published on Kaggle, and was used by Kaggle Datasets Grandmaster Marília Prata in her own notebook.

## Setup

```bash
pip install -U inference-sdk
```

Set your Roboflow API key as an environment variable (never hardcode it):

```powershell
# PowerShell
$env:ROBOFLOW_API_KEY = "your-key-here"
```
```bash
# macOS / Linux
export ROBOFLOW_API_KEY="your-key-here"
```

> **Reviewers/evaluators:** you'll need your own Roboflow API key to run this
> directly (free at [roboflow.com](https://roboflow.com) — sign up, then
> find it under Settings > API Keys). If you'd rather not set one up, the
> submitted pitch video shows the script running end-to-end with real
> output, or feel free to reach out and I can share a temporary evaluation
> key.

## Usage

```bash
python main.py path/to/note_photo.jpg
```

### Example output

```
Running detection on: test_images/100-.jpeg
--------------------------------------------------

VERDICT: GENUINE
Confidence threshold used: 50%

Feature confidences:
  - gandhi_portrait: 59.1%  [PASS]
  - identification-mark: 90.9%  [PASS]
  - lines: 71.1%  [PASS]
  - security-thread: 57.4%  [PASS]
  - serial_panel-1: 75.3%  [PASS]
  - serial_panel-2: 0.0%  [fail]

Features that passed threshold: ['gandhi_portrait', 'identification-mark', 'lines', 'security-thread', 'serial_panel-1']

All raw detections:
  - Indian-Currency_Note: 97.1%
  - identification-mark: 90.9%
  - serial_panel-1: 75.3%
  - lines: 71.1%
  - gandhi_portrait: 59.1%
  - security-thread: 57.4%
```

## Tech stack

- **Roboflow** — dataset annotation, augmentation, YOLO26n training and hosted inference
- **inference-sdk** (Python) — calls the hosted model
- Rule-based verdict logic — plain Python, no extra ML dependencies

## Notes / limitations

- The confidence thresholds are tuned on a small set of test photos; more labeled data (more images, more lighting/angle variety) is the main lever for pushing accuracy higher — this was the single biggest driver of improvement during training (19% → 72% mAP@50 just from adding more images).
- This is a detection + rule-based verdict system, not a from-scratch fraud classifier — it's intentionally explainable: every verdict can be traced back to which specific security features were or weren't detected, and at what confidence.
