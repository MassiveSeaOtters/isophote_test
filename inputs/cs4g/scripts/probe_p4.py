#!/usr/bin/env python
"""
probe_p4.py - Probe IRSA P4 directories for each CS4G candidate.

For each candidate galaxy, fetches the IRSA-hosted P4 directory listing,
parses the linked `.outgal` filenames, and assigns a complexity rank based
on the Salo+2015 filename convention:

    rank 0 -> no P4 entry (HTTP 404)
    rank 1 -> only `*_onecomp.outgal` present
    rank 2 -> `*_twocomp.outgal` present (bulge + disk)
    rank 3 -> `*_twocomp.outgal` PLUS at least one extra-tag .outgal
              (e.g. `_dbar`, `_ddbar`, `_ddbarf`, `_zzn`)

The "best" file per galaxy is the highest-rank `.outgal` (preferring
extra-tag > twocomp > onecomp; among extra-tag files we keep all and pick
later in Phase 3).

Usage:
    python inputs/cs4g/scripts/probe_p4.py --limit 20    # small-scale dry run
    python inputs/cs4g/scripts/probe_p4.py               # full ~1642 probe
    python inputs/cs4g/scripts/probe_p4.py --workers 8 --delay 0.05
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "inputs" / "cs4g" / "cs4g_candidates.csv"
DEFAULT_OUTPUT = REPO_ROOT / "inputs" / "cs4g" / "cs4g_p4_index.csv"
USER_AGENT = "isophote_test/cs4g-probe (research; dr.guangtou@gmail.com)"
TIMEOUT_SEC = 30


class _LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for k, v in attrs:
            if k == "href" and v and not v.startswith("?") and v not in ("/", "./", "../"):
                self.urls.append(v)


def fetch_listing(url: str, retries: int = 2) -> tuple[int, list[str]]:
    """Return (http_status, list_of_outgal_filenames). Status 404 -> empty list."""
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(retries + 1):
        try:
            with request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                parser = _LinkExtractor()
                parser.feed(body)
                outgals = sorted(u for u in parser.urls if u.endswith(".outgal"))
                return 200, outgals
        except HTTPError as e:
            if e.code == 404:
                return 404, []
            last_err = e
        except (URLError, TimeoutError) as e:
            last_err = e
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


# Filename suffix taxonomy. Order matters: extra-tag suffixes win over twocomp,
# twocomp wins over onecomp.
_TAG_RANK_RE = [
    (re.compile(r"_onecomp\.outgal$"), 1, "onecomp"),
    (re.compile(r"_twocomp\.outgal$"), 2, "twocomp"),
]


def classify_outgals(outgals: list[str]) -> tuple[int, str | None]:
    """Map a list of outgal filenames to (complexity_rank, best_filename).

    rank 3 = any file whose suffix is neither `_onecomp` nor `_twocomp`
              (these are S4G-pipeline extra-structure tags: _dbar, _ddbar, etc.)
    """
    if not outgals:
        return 0, None
    extra = [
        f for f in outgals
        if not any(rgx.search(f) for rgx, _, _ in _TAG_RANK_RE)
    ]
    if extra:
        # Prefer the longest tag (more characters typically -> richer model)
        return 3, max(extra, key=len)
    has_two = any(f.endswith("_twocomp.outgal") for f in outgals)
    if has_two:
        return 2, next(f for f in outgals if f.endswith("_twocomp.outgal"))
    has_one = any(f.endswith("_onecomp.outgal") for f in outgals)
    if has_one:
        return 1, next(f for f in outgals if f.endswith("_onecomp.outgal"))
    return 0, None


def probe_one(name: str, url: str, delay: float) -> dict:
    """Probe a single galaxy and return a row dict."""
    if delay > 0:
        time.sleep(delay)
    try:
        status, outgals = fetch_listing(url)
    except Exception as e:
        return {
            "name": name, "p4_url": url, "http_status": -1,
            "n_outgal": 0, "complexity_rank": 0,
            "best_outgal": "", "all_outgals": "", "error": str(e),
        }
    rank, best = classify_outgals(outgals)
    return {
        "name": name, "p4_url": url, "http_status": status,
        "n_outgal": len(outgals), "complexity_rank": rank,
        "best_outgal": best or "",
        "all_outgals": json.dumps(outgals),
        "error": "",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help=f"Candidate CSV (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output index CSV (default: {DEFAULT_OUTPUT})")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent HTTP requests (default: 8)")
    p.add_argument("--delay", type=float, default=0.05,
                   help="Per-request delay in seconds (politeness, default: 0.05)")
    p.add_argument("--limit", type=int, default=0,
                   help="Probe only first N candidates (0 = all). Use for dry runs.")
    args = p.parse_args()

    cands = Table.read(args.input, format="csv")
    if args.limit > 0:
        cands = cands[:args.limit]

    print(f"Probing {len(cands)} galaxies with {args.workers} workers (delay={args.delay}s)")
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(probe_one, str(name), str(url), args.delay): name
                for name, url in zip(cands["name"], cands["p4_url"])}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 50 == 0 or i == len(cands):
                done_pct = 100 * i / len(cands)
                rate = i / (time.time() - t0)
                print(f"  {i:5d}/{len(cands)} ({done_pct:5.1f}%) - {rate:.1f}/s")

    rows.sort(key=lambda r: r["name"])
    tbl = Table(rows=rows, names=list(rows[0].keys()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tbl.write(args.output, format="csv", overwrite=True)

    # Summary
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(cands)/elapsed:.1f}/s)")
    print("Complexity rank distribution:")
    for rank in (0, 1, 2, 3):
        n = int((tbl["complexity_rank"] == rank).sum())
        labels = {0: "no P4 (404)", 1: "onecomp only", 2: "twocomp", 3: "extra-tag"}
        print(f"  rank {rank} ({labels[rank]:14s}): {n:5d}")
    n_err = int((tbl["http_status"] == -1).sum())
    if n_err:
        print(f"  fetch errors:                    {n_err:5d}")
    print(f"\nWrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
