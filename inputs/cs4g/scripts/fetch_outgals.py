#!/usr/bin/env python
"""
fetch_outgals.py - Mirror the best `.outgal` per galaxy locally.

Reads the P4 index produced by `probe_p4.py` and downloads the
`best_outgal` (already chosen by complexity rank) for each galaxy with a
P4 entry. Files are cached at `inputs/cs4g/p4/{name}/{filename}` and
skipped on re-run unless `--force` is passed.

Usage:
    python inputs/cs4g/scripts/fetch_outgals.py --limit 20      # dry run
    python inputs/cs4g/scripts/fetch_outgals.py                 # full mirror
    python inputs/cs4g/scripts/fetch_outgals.py --min-rank 2    # skip onecomp-only
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = REPO_ROOT / "inputs" / "cs4g" / "cs4g_p4_index.csv"
DEFAULT_MIRROR = REPO_ROOT / "inputs" / "cs4g" / "p4"
USER_AGENT = "isophote_test/cs4g-fetch (research; dr.guangtou@gmail.com)"
TIMEOUT_SEC = 60


def fetch_one(url: str, dest: Path, force: bool, delay: float) -> tuple[str, bool, str]:
    """Return (status_string, was_downloaded, error_message)."""
    if dest.exists() and not force:
        return "cached", False, ""
    if delay > 0:
        time.sleep(delay)
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            body = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return "ok", True, ""
    except HTTPError as e:
        return "http_error", False, f"HTTP {e.code}"
    except (URLError, TimeoutError) as e:
        return "net_error", False, str(e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                   help=f"P4 index CSV (default: {DEFAULT_INDEX})")
    p.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR,
                   help=f"Local mirror root (default: {DEFAULT_MIRROR})")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent downloads (default: 8)")
    p.add_argument("--delay", type=float, default=0.05,
                   help="Per-request delay in seconds (default: 0.05)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if file is cached")
    p.add_argument("--min-rank", type=int, default=2,
                   help="Minimum complexity rank to fetch (default: 2 = skip onecomp-only)")
    p.add_argument("--limit", type=int, default=0,
                   help="Fetch only first N galaxies (0 = all)")
    args = p.parse_args()

    idx = Table.read(args.index, format="csv")
    eligible = idx[(idx["complexity_rank"] >= args.min_rank) & (idx["best_outgal"] != "")]
    if args.limit > 0:
        eligible = eligible[:args.limit]

    print(f"Fetching {len(eligible)} galaxies "
          f"(rank >= {args.min_rank}, mirror={args.mirror})")

    def task(name: str, p4_url: str, fname: str):
        url = f"{p4_url.rstrip('/')}/{fname}"
        dest = args.mirror / str(name) / fname
        return name, fetch_one(url, dest, args.force, args.delay)

    counts = {"ok": 0, "cached": 0, "http_error": 0, "net_error": 0}
    errors = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(task, str(r["name"]), str(r["p4_url"]), str(r["best_outgal"]))
                for r in eligible]
        for i, fut in enumerate(as_completed(futs), 1):
            name, (status, _, err) = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "cached"):
                errors.append((name, status, err))
            if i % 100 == 0 or i == len(eligible):
                rate = i / (time.time() - t0)
                print(f"  {i:5d}/{len(eligible)}  ok={counts['ok']:5d}  cached={counts['cached']:5d}  "
                      f"errors={counts['http_error']+counts['net_error']:3d}  ({rate:.1f}/s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Downloaded:    {counts['ok']:5d}")
    print(f"  Already cached:{counts['cached']:5d}")
    print(f"  HTTP errors:   {counts['http_error']:5d}")
    print(f"  Net errors:    {counts['net_error']:5d}")
    if errors:
        print(f"\nFirst few errors:")
        for n, s, e in errors[:10]:
            print(f"  {n}: {s} - {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
