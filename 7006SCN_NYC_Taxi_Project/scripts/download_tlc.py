"""Download *official* NYC TLC 2019 monthly Parquet files; no Kaggle mirrors.

Usage: python scripts/download_tlc.py --head-only
       python scripts/download_tlc.py --config config/config.json
A full download is >1 GiB: run it on a machine with enough disk and bandwidth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coursework.settings import SOURCE_TEMPLATE, ZONE_SOURCE, load_config, now_utc, project_path, write_json


def remote_length(session: requests.Session, url: str) -> int | None:
    reply = session.head(url, timeout=30, allow_redirects=True)
    reply.raise_for_status()
    return int(reply.headers["Content-Length"]) if reply.headers.get("Content-Length") else None


def is_valid_parquet(path: Path, expected: int | None = None) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    if expected is not None and path.stat().st_size != expected:
        return False
    with path.open("rb") as f:
        first = f.read(4)
        f.seek(-4, 2)
        last = f.read(4)
    return first == last == b"PAR1"


def stream_file(session: requests.Session, url: str, dest: Path, expected: int | None = None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".incomplete")
    if temp.exists():
        temp.unlink()  # Never treat an interrupted Parquet transfer as a complete record.
    checksum = hashlib.sha256()
    size = 0
    try:
        with session.get(url, stream=True, timeout=(20, 120)) as reply:
            reply.raise_for_status()
            with temp.open("wb") as output:
                for block in reply.iter_content(chunk_size=2**20):
                    if block:
                        output.write(block)
                        checksum.update(block)
                        size += len(block)
        if expected is not None and size != expected:
            raise ValueError(f"Truncated/changed download {dest.name}: got {size}, expected {expected}")
        if dest.suffix == ".parquet" and not is_valid_parquet(temp, expected):
            raise ValueError(f"Invalid Parquet magic bytes for {dest.name}")
        temp.replace(dest)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--head-only", action="store_true", help="Check remote sizes without downloading")
    parser.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)), help="1..12; all 12 required for final assessment")
    args = parser.parse_args()
    if any(m < 1 or m > 12 for m in args.months) or not args.months:
        parser.error("--months must consist of integers from 1 to 12")
    session = requests.Session()
    session.headers["User-Agent"] = "7006SCN academic reproducibility / TLC 2019"
    try:
        cfg = None if args.head_only else load_config(args.config)
        target = None if cfg is None else project_path(cfg, "raw_dir")
        records = []
        for month in sorted(set(args.months)):
            url = SOURCE_TEMPLATE.format(month=month)
            expected = remote_length(session, url)
            print(f"{month:02d}: {expected} bytes (HTTP HEAD) {url}", flush=True)
            if target is None:
                continue
            path = target / f"yellow_tripdata_2019-{month:02d}.parquet"
            if not is_valid_parquet(path, expected):
                digest = stream_file(session, url, path, expected)
                print(f"   saved {path}", flush=True)
            else:
                with path.open("rb") as existing:
                    digest = hashlib.file_digest(existing, "sha256").hexdigest()
                print(f"   verified existing {path}", flush=True)
            records.append({"month": month, "url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": digest})
        if target is not None:
            zones = project_path(cfg, "zone_lookup")
            if not zones.exists() or zones.stat().st_size == 0:
                stream_file(session, ZONE_SOURCE, zones)
            write_json(target / "download_manifest.json", {
                "observed_at_utc": now_utc(), "source_page": cfg["allocation"]["source_page"],
                "files": records, "total_bytes": sum(x["bytes"] for x in records),
                "zone_lookup_url": ZONE_SOURCE, "all_12_months": len(records) == 12,
            })
            print("Saved manifest. Recheck full-dataset counts in Task1; HEAD bytes are NOT row-count evidence.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
