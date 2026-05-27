"""Synthetic sensor data simulator for AgriScan360 demos.

Generates mean-reverting random-walk readings and pushes them to ThingSpeak
(GET /update) and to the local FastAPI /sensor_data endpoint, simulating the
future ESP8266 firmware so the dashboard can be demoed end-to-end without
hardware.
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

THINGSPEAK_UPDATE_URL = "https://api.thingspeak.com/update"
IST = ZoneInfo("Asia/Kolkata")

INITIAL = {"soil": 55.0, "temp": 27.0, "hum": 70.0}
WALK_PARAMS = {
    "soil": {"mean": 55.0, "sigma": 1.5, "lo": 40.0, "hi": 70.0},
    "temp": {"mean": 27.0, "sigma": 0.4, "lo": 22.0, "hi": 32.0},
    "hum":  {"mean": 70.0, "sigma": 1.0, "lo": 55.0, "hi": 85.0},
}


def step(value: float, mean: float, sigma: float, lo: float, hi: float) -> float:
    delta = random.gauss(0.0, sigma) - 0.05 * (value - mean)
    return round(max(lo, min(hi, value + delta)), 1)


def now_ist_clock() -> str:
    return datetime.now(IST).strftime("%H:%M:%S IST")


def push_thingspeak(api_key: str, soil: float, temp: float, hum: float) -> tuple[bool, str]:
    try:
        resp = requests.get(
            THINGSPEAK_UPDATE_URL,
            params={"api_key": api_key, "field1": soil, "field2": temp, "field3": hum},
            timeout=10,
        )
    except requests.exceptions.RequestException as exc:
        return False, f"ERROR {exc.__class__.__name__}: {exc}"

    body = resp.text.strip()
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} ({body[:80]})"
    if body == "0":
        return False, "rate-limited (entry 0)"
    return True, f"200 OK (entry {body})"


def push_local(local_url: str, device_id: str, soil: float, temp: float, hum: float) -> tuple[bool, str]:
    try:
        resp = requests.post(
            local_url,
            json={
                "device_id": device_id,
                "soil_moisture": soil,
                "air_temp": temp,
                "air_humidity": hum,
            },
            timeout=5,
        )
    except requests.exceptions.RequestException as exc:
        return False, f"ERROR {exc.__class__.__name__}: {exc}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} ({resp.text[:80]})"
    try:
        record_id = resp.json().get("record_id", "") or ""
    except ValueError:
        return False, f"HTTP 200 (non-JSON: {resp.text[:80]})"
    return True, f"200 OK (record_id={record_id[:8]}...)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ThingSpeak + local sensor simulator")
    p.add_argument("--duration", type=int, default=300, help="seconds to run")
    p.add_argument("--interval", type=int, default=15, help="seconds between pushes")
    p.add_argument("--device-id", default="agro-sim-01")
    p.add_argument("--local-url", default="http://127.0.0.1:8000/sensor_data")
    p.add_argument("--no-thingspeak", action="store_true", help="skip ThingSpeak pushes")
    p.add_argument("--no-local", action="store_true", help="skip local FastAPI pushes")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    api_key = ""
    channel_id = ""
    if not args.no_thingspeak:
        api_key = os.getenv("THINGSPEAK_WRITE_API_KEY", "").strip()
        channel_id = os.getenv("THINGSPEAK_CHANNEL_ID", "").strip()
        if not api_key:
            print(
                "FATAL: THINGSPEAK_WRITE_API_KEY is not set. "
                "Set it (setx on Windows / export on POSIX) or pass --no-thingspeak."
            )
            return 2
        if not channel_id:
            print(
                "FATAL: THINGSPEAK_CHANNEL_ID is not set. "
                "Set it or pass --no-thingspeak."
            )
            return 2

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print("=== AgriScan360 sensor simulator ===")
    print(f"  device_id   : {args.device_id}")
    print(f"  duration    : {args.duration}s")
    print(f"  interval    : {args.interval}s")
    print(f"  local target: {'(skipped)' if args.no_local else args.local_url}")
    print(f"  thingspeak  : {'(skipped)' if args.no_thingspeak else f'channel {channel_id}'}")
    print(f"  seed        : {args.seed}")
    print()

    soil = INITIAL["soil"]
    temp = INITIAL["temp"]
    hum = INITIAL["hum"]

    start = time.time()
    tick = 0
    ts_ok = ts_fail = local_ok = local_fail = 0

    try:
        while True:
            if time.time() - start > args.duration:
                break

            tick += 1
            soil = step(soil, **WALK_PARAMS["soil"])
            temp = step(temp, **WALK_PARAMS["temp"])
            hum = step(hum, **WALK_PARAMS["hum"])

            print(f"[{now_ist_clock()}] tick #{tick}  soil={soil}%  temp={temp}°C  hum={hum}%")

            if not args.no_thingspeak:
                ok, msg = push_thingspeak(api_key, soil, temp, hum)
                print(f"  ↳ thingspeak: {msg}")
                if ok:
                    ts_ok += 1
                else:
                    ts_fail += 1

            if not args.no_local:
                ok, msg = push_local(args.local_url, args.device_id, soil, temp, hum)
                print(f"  ↳ local:      {msg}")
                if ok:
                    local_ok += 1
                else:
                    local_fail += 1

            remaining = args.duration - (time.time() - start)
            if remaining <= 0:
                break
            time.sleep(min(args.interval, remaining))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    elapsed = time.time() - start
    print()
    print("=== Summary ===")
    print(f"  ticks         : {tick}")
    print(f"  thingspeak    : {ts_ok} ok / {ts_fail} fail")
    print(f"  local         : {local_ok} ok / {local_fail} fail")
    print(f"  elapsed       : {elapsed:.1f}s")
    if not args.no_thingspeak and channel_id:
        print(f"  View live chart: https://thingspeak.com/channels/{channel_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
