"""
verify_note.py -- Loads the Roboflow-trained YOLO26n currency detector,
runs it on a note photo, and applies a simple rule-based verdict on top
of the raw detections (GENUINE / SUSPICIOUS / FAKE).

USAGE:
    pip install -U inference-sdk
    python verify_note.py path/to/note_photo.jpg

The Roboflow model itself only detects and localizes security features
(gandhi_portrait, serial_panel-1, serial_panel-2, etc.) with a confidence
score each -- it does NOT output a genuine/fake label on its own. This
script adds that verdict layer using simple, explainable rules.
"""

import sys
import json
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ---------------------------------------------------------------------------
# CONFIG -- fill in / adjust as needed
# ---------------------------------------------------------------------------
import os
API_URL = "https://serverless.roboflow.com"
API_KEY = os.environ.get("ROBOFLOW_API_KEY")  # set this in your environment, never hardcode a real key here
if not API_KEY:
    raise RuntimeError(
        "Set the ROBOFLOW_API_KEY environment variable before running this script.\n"
        "PowerShell:  $env:ROBOFLOW_API_KEY = \"your-key-here\"\n"
        "Then re-run: python main.py <image_path>"
    )

# Direct model ID (not a workflow ID). Get the exact value from:
# Roboflow > your model > "Deploy Model" > "Deploy My API" tab -- it shows
# a client.infer(..., model_id="...") snippet with the correct ID filled in.
MODEL_ID = "indian-currency-5og0r/2"  # TODO: replace with the exact ID from "Deploy My API"

# The 6 security feature classes from the dataset (Indian-Currency_Note
# itself is just the note boundary box, not a security feature, so it's
# excluded from the verdict logic).
REQUIRED_FEATURES = [
    "gandhi_portrait",
    "identification-mark",
    "lines",
    "security-thread",
    "serial_panel-1",
    "serial_panel-2",
]
CONFIDENCE_THRESHOLD = 0.50  # 50% -- matches the model's real recall (~71%);
                              # a 70% bar was too strict per-feature and even
                              # flagged genuine notes as suspicious in testing.

# Verdict thresholds -- how many of the 6 required features must clear the
# confidence bar for each verdict tier. Not all 6 need to pass for GENUINE --
# real photos (angle/lighting) naturally push 1-2 features below the bar
# even on authentic notes, so we ask for a clear majority instead of 100%.
GENUINE_MIN_FEATURES = 5      # 5 or 6 of 6 features pass -> GENUINE
SUSPICIOUS_MIN_FEATURES = 3   # 3-4 of 6 pass -> SUSPICIOUS
                               # 0-2 of 6 pass -> FAKE


def get_predictions(image_path: str) -> list:
    """Calls the Roboflow model directly (not via the workflow, which had a
    server-side image-deserialization bug) and returns raw predictions."""
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY).configure(
        InferenceConfiguration(api_key_transport="header")  # header-based auth (inference v1.5.0+)
    )

    # Pass the plain file path -- the SDK reads and base64-encodes it
    # internally. (Do NOT wrap it in a dict; the SDK only accepts a
    # string path/URL/base64, a numpy array, or a PIL Image.)
    response = client.infer(image_path, model_id=MODEL_ID)

    # client.infer() on a plain detection model returns a dict with a
    # top-level "predictions" list directly -- no need for the nested
    # search that run_workflow() needed.
    predictions = response.get("predictions") if isinstance(response, dict) else None

    if predictions is None:
        print("\n[!] Could not find a 'predictions' list in the API response.")
        print("Raw response below -- check MODEL_ID and the response structure:\n")
        print(json.dumps(response, indent=2))
        return []

    return predictions


def apply_verdict(predictions: list) -> dict:
    """Applies the simple rule-based genuine/suspicious/fake logic."""
    # Best confidence seen per required feature (a feature may be detected
    # more than once, e.g. serial_panel-1 appearing twice at different spots).
    best_confidence = {feat: 0.0 for feat in REQUIRED_FEATURES}
    all_detections = []

    for pred in predictions:
        cls = pred.get("class", "unknown")
        conf = pred.get("confidence", 0.0)
        all_detections.append({"class": cls, "confidence": round(conf, 3)})
        if cls in best_confidence and conf > best_confidence[cls]:
            best_confidence[cls] = conf

    passed_features = [f for f, c in best_confidence.items() if c >= CONFIDENCE_THRESHOLD]
    num_passed = len(passed_features)

    if num_passed >= GENUINE_MIN_FEATURES:
        verdict = "GENUINE"
    elif num_passed >= SUSPICIOUS_MIN_FEATURES:
        verdict = "SUSPICIOUS"
    else:
        verdict = "FAKE"

    return {
        "verdict": verdict,
        "feature_confidences": {f: round(c, 3) for f, c in best_confidence.items()},
        "features_passed_threshold": passed_features,
        "threshold_used": CONFIDENCE_THRESHOLD,
        "all_detections": all_detections,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_note.py path/to/note_photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\nRunning detection on: {image_path}")
    print("-" * 50)

    predictions = get_predictions(image_path)
    if not predictions:
        print("No predictions returned -- nothing to verify.")
        sys.exit(1)

    result = apply_verdict(predictions)

    print(f"\nVERDICT: {result['verdict']}")
    print(f"Confidence threshold used: {result['threshold_used'] * 100:.0f}%\n")
    print("Feature confidences:")
    for feat, conf in result["feature_confidences"].items():
        status = "PASS" if conf >= result["threshold_used"] else "fail"
        print(f"  - {feat}: {conf * 100:.1f}%  [{status}]")

    print(f"\nFeatures that passed threshold: {result['features_passed_threshold']}")
    print("\nAll raw detections:")
    for d in result["all_detections"]:
        print(f"  - {d['class']}: {d['confidence'] * 100:.1f}%")
    print()


if __name__ == "__main__":
    main()