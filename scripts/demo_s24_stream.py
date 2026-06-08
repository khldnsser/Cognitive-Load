#!/usr/bin/env python
"""Live demo: stream held-out Subject 24 through the deployed EEP endpoint.

Separates S24 into its baseline (LOW load) and cognitive_load (HIGH load)
recordings, calibrates on rest signals, then replays each 60s window through
the public EEP /predict in (fast-forwarded) real time. The predicted label is
printed next to the ground-truth label so you can watch the model classify
cognitive load live in the terminal.

Because it hits the cloud EEP (AWS NLB) by default, every request flows through
the EKS pods and is scraped by the in-cluster Prometheus — so the cloud Grafana
(kubectl port-forward -n cogload svc/grafana 3000:3000) updates as this runs.

Usage:
    python scripts/demo_s24_stream.py                 # cloud EEP, default pacing
    python scripts/demo_s24_stream.py --delay 0.5     # faster fast-forward
    python scripts/demo_s24_stream.py --session post  # use the 'post' session
    python scripts/demo_s24_stream.py --url http://localhost:8080   # local stack
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogload.config import RAW_DATA_ROOT  # noqa: E402
from cogload.data.loader import load_bvp, load_eda, load_temp  # noqa: E402
from cogload.features.windowing import sliding_windows  # noqa: E402

# --- demo constants -------------------------------------------------------
DEFAULT_URL = (
    "http://a55d869eb058c442580c72b3969c5d48-8766164aef600474.elb."
    "eu-central-1.amazonaws.com"
)
HOLDOUT_SUBJECT = 24
WINDOW_S = 60   # champion was trained on 60s windows
HOP_S = 15

# --- ANSI colors ----------------------------------------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


# --- tiny stdlib HTTP helpers (no third-party deps) -----------------------
class HttpError(Exception):
    """Any network/HTTP failure — keeps call sites simple."""


def http_get(url: str, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise HttpError(str(e)) from e


def http_post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise HttpError(str(e)) from e


def label_str(v: int) -> str:
    return f"{RED}HIGH{RESET}" if v == 1 else f"{GREEN}LOW {RESET}"


def s24_dir(session: str) -> Path:
    return RAW_DATA_ROOT / "survey_gamification" / str(HOLDOUT_SUBJECT) / session


def load_condition(session: str, condition: str):
    """Return (bvp, eda, temp) DataFrames for one S24 condition recording."""
    path = s24_dir(session) / condition
    if not path.exists():
        sys.exit(f"{RED}Missing data:{RESET} {path}")
    return load_bvp(path), load_eda(path), load_temp(path)


def window_to_payload(slices) -> dict:
    """Turn a sliding-window slice dict into a SignalWindow JSON payload."""
    bvp, eda, temp = slices["bvp"], slices["eda"], slices["temp"]
    return {
        "bvp_values":  bvp["bvp"].tolist(),
        "bvp_times":   bvp["time"].tolist(),
        "eda_values":  eda["eda"].tolist(),
        "eda_times":   eda["time"].tolist(),
        "temp_values": temp["temp"].tolist(),
        "temp_times":  temp["time"].tolist(),
    }


def collect_windows(bvp, eda, temp) -> list[dict]:
    out = []
    for _t_start, slices in sliding_windows(bvp, eda, temp, WINDOW_S, HOP_S):
        if any(slices[s] is None or slices[s].empty for s in ("bvp", "eda", "temp")):
            continue
        out.append(window_to_payload(slices))
    return out


def calibrate(url: str, rest_windows: list[dict]) -> dict | None:
    """POST a few rest windows to EEP /calibrate, return CalibrationParams."""
    if not rest_windows:
        return None
    body = {"windows": rest_windows[: min(4, len(rest_windows))]}
    print(f"{CYAN}Calibrating{RESET} on {len(body['windows'])} rest windows ...", end=" ", flush=True)
    try:
        resp = http_post(f"{url}/calibrate", body, timeout=30)
    except HttpError as e:
        print(f"{YELLOW}failed ({e}); falling back to UNCALIBRATED predictions.{RESET}")
        return None
    params = resp["params"]
    print(f"{GREEN}done{RESET} ({len(params['mu'])} features).")
    return params


def stream(url: str, title: str, true_label: int, windows: list[dict],
           calib: dict | None, delay: float, tally: dict) -> None:
    print(f"\n{BOLD}── Streaming {title}  (ground truth = {label_str(true_label)}{BOLD}) "
          f"· {len(windows)} windows ──{RESET}")
    for i, w in enumerate(windows, 1):
        payload = {"window": w, "calib": calib}
        t0 = time.time()
        try:
            res = http_post(f"{url}/predict", payload, timeout=15)
        except HttpError as e:
            print(f"  win {i:2d}  {YELLOW}request failed: {e}{RESET}")
            continue
        ms = (time.time() - t0) * 1000
        pred, prob = res["prediction"], res["probability"]
        correct = pred == true_label
        tally["total"] += 1
        tally["correct"] += int(correct)
        mark = f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"
        cal = "" if res.get("calibrated") else f" {YELLOW}(uncal){RESET}"
        print(f"  win {i:2d}  pred {label_str(pred)}  p={prob:0.2f}  "
              f"thr={res['threshold']:.2f}  {mark}{cal}  {GREY}{ms:4.0f}ms{RESET}")
        time.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help="EEP base URL (default: cloud NLB)")
    ap.add_argument("--session", default="pre", choices=["pre", "post"],
                    help="S24 survey session to replay (default: pre)")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between windows — fast-forward factor (real hop is 15s)")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    print(f"{BOLD}Cognitive-Load live demo — Subject {HOLDOUT_SUBJECT} (held out){RESET}")
    print(f"Endpoint : {CYAN}{url}{RESET}")
    print(f"Session  : {args.session}   ·   window {WINDOW_S}s / hop {HOP_S}s   ·   "
          f"pacing {args.delay}s/window (real-time hop is {HOP_S}s)")

    # health check
    try:
        status = http_get(f"{url}/health", timeout=10).get("status", "?")
        col = GREEN if status == "ok" else YELLOW
        print(f"Health   : {col}{status}{RESET}")
    except HttpError as e:
        sys.exit(f"{RED}EEP unreachable: {e}{RESET}")

    # load + window both conditions
    base_b, base_e, base_t = load_condition(args.session, "baseline")
    cog_b, cog_e, cog_t = load_condition(args.session, "cognitive_load")
    baseline_windows = collect_windows(base_b, base_e, base_t)
    cognitive_windows = collect_windows(cog_b, cog_e, cog_t)
    print(f"Windows  : baseline={len(baseline_windows)}  cognitive={len(cognitive_windows)}")

    # calibrate on rest (baseline) signals, then stream both conditions
    calib = calibrate(url, baseline_windows)

    tally = {"total": 0, "correct": 0}
    stream(url, "BASELINE / REST", 0, baseline_windows, calib, args.delay, tally)
    stream(url, "COGNITIVE LOAD / STROOP", 1, cognitive_windows, calib, args.delay, tally)

    acc = tally["correct"] / tally["total"] if tally["total"] else 0.0
    print(f"\n{BOLD}Summary{RESET}: {tally['correct']}/{tally['total']} correct  "
          f"→  accuracy {acc:0.1%}")
    print(f"{GREY}Cloud Grafana now reflects these requests "
          f"(kubectl port-forward -n cogload svc/grafana 3000:3000).{RESET}")


if __name__ == "__main__":
    main()
