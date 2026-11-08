"""Fetch selected, complete SDSD frame sequences from the author's public split ZIP.

HTTP byte ranges avoid downloading all 12.8 GB. Each extracted member is
validated against the ZIP central directory CRC32. Original .npy bytes are kept.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import threading
import time
import zlib

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
PARTS = [
    ("13ueV_fAPWAnY0AJ2gaKvIDBU2ggIISYJ", 4698669056),
    ("1uOez3vXhdO8NZChM8-TIwRbiu6Srdy5o", 4698669056),
    ("1hnZcLrSnSzpl063oRnwWoAe6x-XQCOJA", 3390764288),
]
local = threading.local()


def get_range(disk: int, start: int, size: int) -> bytes:
    if not hasattr(local, "session"):
        local.session = requests.Session()
        # The host's inherited proxy cannot reach Google Drive in this session.
        local.session.trust_env = False
    assert 0 <= start < PARTS[disk][1] and size > 0
    n = min(size, PARTS[disk][1] - start)
    url = "https://drive.usercontent.google.com/download"
    for attempt in range(5):
        try:
            r = local.session.get(url, params={"id": PARTS[disk][0],
                "export": "download", "confirm": "t", "range_start": start},
                headers={"Range": f"bytes={start}-{start+n-1}"}, timeout=(15, 90))
            r.raise_for_status()
            expected = f"bytes {start}-{start+n-1}/{PARTS[disk][1]}"
            if r.status_code != 206 or r.headers.get("Content-Range") != expected or len(r.content) != n:
                raise ValueError(f"Invalid range response: {r.status_code} {r.headers.get('Content-Range')}")
            result = r.content
            return result if n == size else result + get_range(disk + 1, 0, size - n)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))


def index() -> list[dict]:
    cache = ROOT / "records/sdsd_indoor_zip_index.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    tail = get_range(2, PARTS[2][1] - 131072, 131072)
    pos = tail.rfind(b"PK\x05\x06")
    _, disk, cd_disk, _, count, cd_size, cd_offset, _ = struct.unpack("<4s4H2IH", tail[pos:pos+22])
    assert disk == cd_disk == 2
    data = get_range(cd_disk, cd_offset, cd_size)
    members, p = [], 0
    while p < len(data):
        vals = struct.unpack("<4s6H3I5H2I", data[p:p+46])
        sig, _, _, flags, method, _, _, crc, csize, usize, nl, xl, cl, disk, _, _, off = vals
        assert sig == b"PK\x01\x02"
        name = data[p+46:p+46+nl].decode("utf-8" if flags & 2048 else "cp437")
        extra = data[p+46+nl:p+46+nl+xl]
        q = 0
        while q + 4 <= len(extra):
            tag, ln = struct.unpack("<HH", extra[q:q+4]); content = extra[q+4:q+4+ln]; q += 4 + ln
            if tag == 1:
                stream = io.BytesIO(content)
                if usize == 0xFFFFFFFF: usize = struct.unpack("<Q", stream.read(8))[0]
                if csize == 0xFFFFFFFF: csize = struct.unpack("<Q", stream.read(8))[0]
                if off == 0xFFFFFFFF: off = struct.unpack("<Q", stream.read(8))[0]
                if disk == 0xFFFF: disk = struct.unpack("<I", stream.read(4))[0]
        members.append(dict(name=name, disk=disk, offset=off, compressed=csize,
                            size=usize, method=method, crc32=crc))
        p += 46 + nl + xl + cl
    assert len(members) == count
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(members, indent=2), encoding="utf-8")
    return members


def extract(member: dict) -> dict:
    dest = ROOT / "data/raw" / member["name"]
    # Reject paths outside this project's dataset directory.
    assert dest.resolve().is_relative_to((ROOT / "data/raw").resolve())
    if dest.exists() and dest.stat().st_size == member["size"]:
        raw = dest.read_bytes()
        if zlib.crc32(raw) == member["crc32"]:
            return dict(path=str(dest.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), cached=True)
    # Original filenames are short; include local-header extra fields in one read.
    block = get_range(member["disk"], member["offset"], member["compressed"] + 4096)
    vals = struct.unpack("<4s5H3I2H", block[:30])
    assert vals[0] == b"PK\x03\x04" and not vals[2] & 1, "Encrypted/invalid member"
    start = 30 + vals[-2] + vals[-1]
    compressed = block[start:start+member["compressed"]]
    raw = zlib.decompress(compressed, -15) if member["method"] == 8 else compressed
    assert len(raw) == member["size"] and zlib.crc32(raw) == member["crc32"], member["name"]
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    assert arr.ndim == 3 and arr.shape[-1] == 3
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return dict(path=str(dest.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(),
                shape=list(arr.shape), dtype=str(arr.dtype), min=float(arr.min()), max=float(arr.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="pair1,pair2,pair3,pair4,pair5,pair6,pair7,pair8,pair9,pair10,pair11,pair12,pair13,pair15,pair21,pair23,pair31,pair33,pair50,pair52,pair58,pair60,pair68,pair70")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()
    members = index()
    print("Archive entries", len(members), flush=True)
    if args.index_only:
        print(json.dumps(members[:7], indent=2)); return
    selected = []
    for scene in args.scenes.split(","):
        groups = {}
        for m in members:
            parts = m["name"].split("/")
            if scene in parts and m["name"].endswith(".npy"):
                groups.setdefault("/".join(parts[:-1]), []).append(m)
        if len(groups) != 2:
            raise ValueError(f"Expected exactly 2 paired folders for {scene}, got {list(groups)}")
        for folder, group in groups.items():
            group.sort(key=lambda m: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", m["name"])])
            assert len(group) >= args.frames
            selected += group[:args.frames]
    print("Download",len(selected),"frames; compressed MB",round(sum(x['compressed'] for x in selected)/1e6),flush=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, row in enumerate(executor.map(extract, selected), 1):
            records.append(row)
            if i % 10 == 0 or i == len(selected): print(f"{i}/{len(selected)} {row['path']}",flush=True)
    target = ROOT / "records/download_manifest.json"
    previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    previous.update({r['path']: r for r in records})
    target.write_text(json.dumps(previous, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
