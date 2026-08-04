#!/usr/bin/env python3
"""Download a diverse out-of-distribution image set for classifier rejection testing.

Kept in data/negatives/ood/ rather than data/negatives/images/, which the detector
pipelines own — mixing them would change detector training as a side effect.

Source is picsum.photos (Unsplash-derived). These are real-world photographs, so
they match the *style* of a phone photo, unlike PlantVillage/AgML leaf datasets
whose plain studio backgrounds would teach the model to key on the background
instead of the subject. They are still easy-to-medium negatives: objects,
interiors, people, landscapes. They do NOT cover the near-miss case (a different
crop's leaf photographed in a field), which needs real field images.

Usage:  python fetch_ood_negatives.py [count]
"""
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # this file lives in scripts/
OUT = PROJECT_ROOT / "data" / "negatives" / "ood"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(1234)                       # reproducible set
seeds = random.sample(range(2000, 20000), COUNT)   # disjoint from the detector seeds (1-2000)

pending = [(s, OUT / f"ood_{s:05d}.jpg") for s in seeds
           if not (OUT / f"ood_{s:05d}.jpg").exists()]
print(f"  target {COUNT}, {COUNT - len(pending)} cached, {len(pending)} to fetch")


def fetch(item):
    seed, dest = item
    url = f"https://picsum.photos/seed/{seed}/640/640"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
        if len(data) < 5000:            # truncated / error page
            return seed, "too small"
        dest.write_bytes(data)
        return seed, None
    except Exception as exc:
        return seed, str(exc)[:60]


ok = err = 0
with ThreadPoolExecutor(max_workers=16) as pool:
    futs = {pool.submit(fetch, it): it for it in pending}
    for done, f in enumerate(as_completed(futs), 1):
        _, exc = f.result()
        if exc:
            err += 1
        else:
            ok += 1
        if done % 200 == 0 or done == len(pending):
            print(f"    {done}/{len(pending)}  ok={ok} err={err}")

have = sorted(OUT.glob("*.jpg"))
print(f"  done: {len(have)} images in {OUT}")
