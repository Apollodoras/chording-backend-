"""Takedown CLI — §3's "satisfiable in minutes", from a laptop.

    python scripts/admin.py block   --video dQw4w9WgXcQ --reason "DMCA 2026-08-03"
    python scripts/admin.py block   --channel UCxxxx    --reason "label request"
    python scripts/admin.py unblock --video dQw4w9WgXcQ
    python scripts/admin.py purge   --video dQw4w9WgXcQ
    python scripts/admin.py offset  --video dQw4w9WgXcQ --ms -250
    python scripts/admin.py audit
    python scripts/admin.py blocklist

Reads `CHORDS_BASE_URL` and `CHORDS_ADMIN_TOKEN` from the environment. `block`
purges as part of the same request and **prints the row counts**, because the
handoff asks you to verify the cascade actually cascaded rather than assume it —
a purge that silently matched nothing is the failure you find out about from a
lawyer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

BASE_URL = os.environ.get("CHORDS_BASE_URL", "http://127.0.0.1:8000")
TOKEN = os.environ.get("CHORDS_ADMIN_TOKEN", "")
ACTOR = os.environ.get("CHORDS_ADMIN_ACTOR") or os.environ.get("USER") or "admin"


def headers() -> dict[str, str]:
    if not TOKEN:
        sys.exit("CHORDS_ADMIN_TOKEN is not set — refusing to send an unauthenticated request.")
    return {"X-Admin-Token": TOKEN, "X-Admin-Actor": ACTOR}


def show(response: httpx.Response) -> int:
    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)
    return 0 if response.is_success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Chord-analysis takedown admin")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("block", "unblock"):
        p = sub.add_parser(name)
        p.add_argument("--video")
        p.add_argument("--channel")
        p.add_argument("--reason")

    p = sub.add_parser("purge")
    p.add_argument("--video", required=True)

    p = sub.add_parser("offset")
    p.add_argument("--video", required=True)
    p.add_argument("--ms", type=int, required=True,
                   help="positive = the chart is early and should be pushed later")

    sub.add_parser("audit").add_argument("--limit", type=int, default=50)
    sub.add_parser("blocklist")

    args = parser.parse_args()
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        if args.command == "block":
            body = {"videoId": args.video, "channelId": args.channel, "reason": args.reason}
            return show(client.post("/v1/admin/block", json=body, headers=headers()))
        if args.command == "unblock":
            body = {"videoId": args.video, "channelId": args.channel}
            return show(client.request("DELETE", "/v1/admin/block", json=body, headers=headers()))
        if args.command == "purge":
            return show(client.delete(f"/v1/admin/maps/{args.video}", headers=headers()))
        if args.command == "offset":
            return show(client.post(f"/v1/admin/maps/{args.video}/offset",
                                    json={"offsetMs": args.ms}, headers=headers()))
        if args.command == "audit":
            return show(client.get("/v1/admin/audit", params={"limit": args.limit},
                                   headers=headers()))
        if args.command == "blocklist":
            return show(client.get("/v1/admin/blocklist", headers=headers()))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
