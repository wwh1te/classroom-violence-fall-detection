"""Reproducible video preparation for issue #1. Run as python -m src.data.pipeline."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from urllib.request import Request, urlopen

LABELS = {"normal", "violence", "fall"}
SPLITS = ("train", "val", "test")
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv", ".webm"}

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    temp.replace(path)

def inside(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path outside root: {name}")
    return path

def checksum(path, git=False):
    path = Path(path)
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def download(row, root, limit):
    path = inside(root, row["raw_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid(p):
        return (p.is_file() and p.stat().st_size > 0
                and (not row.get("source_bytes") or p.stat().st_size == row["source_bytes"])
                and (not row.get("git_blob_sha1") or checksum(p, True) == row["git_blob_sha1"]))
    if valid(path):
        return dict(row, sha256=checksum(path), download_status="cached")
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urlopen(Request(row["url"], headers={"User-Agent": "ClassroomSafetyAI/1"}), timeout=60) as response, temp.open("wb") as stream:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > limit:
                        raise ValueError("Download exceeds per-file limit")
                    stream.write(block)
            if not valid(temp):
                raise ValueError("Empty/incomplete download or source checksum mismatch")
            temp.replace(path)
            return dict(row, sha256=checksum(path), download_status="downloaded")
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)

def inspect(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps, expected = cap.get(cv2.CAP_PROP_FPS), round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Unreadable video/fps: {path}")
        count, shape = 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if shape is not None and shape != frame.shape:
                raise ValueError("Frame dimensions change")
            shape, count = frame.shape, count + 1
        if not count or (expected > 0 and count != expected):
            raise ValueError(f"Truncated video: {count}/{expected} frames: {path}")
        return dict(fps=fps, frames=count, width=shape[1], height=shape[0], duration_s=count/fps)
    finally:
        cap.release()

def letterbox(frame, width, height):
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    scale = min(width/w, height/h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width-nw)//2, (height-nh)//2
    canvas[y:y+nh, x:x+nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas

def dhash(frame):
    import cv2
    tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8))
    return f"{sum(int(b) << i for i,b in enumerate((tiny[:,1:] > tiny[:,:-1]).flatten())):016x}"

def _normalize_video(source, path, cfg, crop=None):
    import cv2
    original = inspect(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.mp4")
    width, height, fps = cfg["width"], cfg["height"], cfg["fps"]
    count = max(1, math.ceil(original["duration_s"] * fps - 1e-6))
    positions = {round(i*(count-1)/4) for i in range(5)}
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    hashes, source_index = [], -1
    try:
        if not writer.isOpened():
            raise ValueError("Cannot open MP4 encoder")
        for i in range(count):
            wanted = min(original["frames"]-1, math.floor(i*original["fps"]/fps + 1e-6))
            while source_index < wanted:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Decode failed during normalization")
                source_index += 1
            image = frame[:,frame.shape[1]//2:] if crop == "right_half_rgb" else frame
            image = letterbox(image, width, height)
            writer.write(image)
            if i in positions:
                hashes.append(dhash(image))
        writer.release()
        video = inspect(temp)
        if video["frames"] != count or (video["width"],video["height"]) != (width,height):
            raise ValueError("Normalized output failed verification")
        temp.replace(path)
    except Exception:
        writer.release()
        temp.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
    return dict(original=original, video=video, dhashes=hashes)

def normalize(row, root, cfg):
    source = inside(root, row["raw_path"])
    if checksum(source) != row["sha256"]:
        raise ValueError("Raw checksum changed")
    name = f"data/processed/{row['label']}/{row['sample_id']}.mp4"
    path = inside(root, name)
    normalized = _normalize_video(source, path, cfg, row.get("crop"))
    original, video, hashes = (
        normalized["original"], normalized["video"], normalized["dhashes"]
    )
    return dict(row, processed_path=name, processed_sha256=checksum(path),
                original=original, video=video, dhashes=hashes,
                label_status="source_label_pending_review", subject_ids=[],
                scene_id=row["dataset"] + "_scene_unknown")

def near_candidates(rows, threshold=4):
    candidates=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            ah,bh=a.get("dhashes",[]),b.get("dhashes",[])
            if not ah or len(ah)!=len(bh) or a["group_id"]==b["group_id"]:
                continue
            if abs(a["video"]["duration_s"]-b["video"]["duration_s"])>.25:
                continue
            distance=sum((int(x,16)^int(y,16)).bit_count() for x,y in zip(ah,bh))/len(ah)
            if distance<=threshold:
                candidates.append(dict(left=a["sample_id"],right=b["sample_id"],mean_hamming=distance,
                                       cross_split=a.get("split")!=b.get("split")))
    return candidates

def split_rows(rows, ratios=(.7,.15,.15), seed=42, near_threshold=4):
    """Connected components keep same recordings and exact duplicates together."""
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or not math.isclose(sum(ratios),1):
        raise ValueError("Need three positive ratios summing to 1")
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i,row in enumerate(rows):
        for field in ("group_id","sha256","processed_sha256","parent_video_id"):
            if row.get(field):
                token = (field,row[field])
                if token in seen:
                    parent[find(i)] = find(seen[token])
                seen[token] = i
    by_id={r['sample_id']:i for i,r in enumerate(rows)}
    for pair in near_candidates(rows,near_threshold):
        parent[find(by_id[pair['left']])]=find(by_id[pair['right']])
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        groups[find(i)].append(row)
    strata = defaultdict(list)
    for group in groups.values():
        strata[tuple(sorted({(r["dataset"],r["label"]) for r in group}))].append(group)
    rng, output = random.Random(seed), []
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda g:min(r["sample_id"] for r in g))
        if len(items) < 3:
            raise ValueError(f"Need >=3 independent groups per stratum: {key}")
        rng.shuffle(items)
        counts = [max(1,math.floor(len(items)*x)) for x in ratios]
        while sum(counts) > len(items):
            j = max((j for j in range(3) if counts[j]>1),key=lambda j:counts[j]-len(items)*ratios[j])
            counts[j] -= 1
        while sum(counts) < len(items):
            j = max(range(3),key=lambda j:len(items)*ratios[j]-counts[j])
            counts[j] += 1
        start = 0
        for name,n in zip(SPLITS,counts):
            for group in items[start:start+n]:
                component=hashlib.sha256("|".join(sorted(r["sample_id"] for r in group)).encode()).hexdigest()[:16]
                output.extend(dict(r,split=name,split_component_id=component) for r in group)
            start += n
    return sorted(output,key=lambda r:r["sample_id"])

def audit(rows, threshold=4):
    errors, warnings, near = [], [], []
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        errors.append("empty_manifest_or_duplicate_ids")
    for row in rows:
        if row.get("split") not in SPLITS or row["label"] not in LABELS:
            errors.append("invalid_split_or_label:" + row["sample_id"])
    for field in ("group_id","sha256","processed_sha256","parent_video_id"):
        buckets = defaultdict(list)
        for row in rows:
            if row.get(field):
                buckets[row[field]].append(row)
        for value,group in buckets.items():
            if len({r["split"] for r in group}) > 1:
                errors.append(f"cross_split_{field}:{value}")
            if "sha256" in field and len(group)>1:
                warnings.append(dict(type="exact_duplicate",field=field,samples=[r["sample_id"] for r in group]))
    for name in SPLITS:
        if {r["label"] for r in rows if r.get("split")==name} != LABELS:
            errors.append("missing_class:" + name)
    near=near_candidates(rows,threshold)
    pending = sum(r.get("label_status")!="reviewed" for r in rows)
    if pending:
        warnings.append(dict(type="pending_human_label_review",count=pending))
    warnings.append(dict(type="subject_scene_independence_not_established"))
    return dict(sample_count=len(rows), errors=errors, warnings=warnings,
                near_duplicate_candidates=near,
                counts=dict(sorted(Counter(r["split"]+"/"+r["label"] for r in rows).items())),
                ready_for_final_evaluation=not errors and not pending and not any(x["cross_split"] for x in near))

def annotations(path, rows):
    if path.exists():
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as stream:
        fields = ["sample_id","start_s","end_s","label","person_ids","status","annotator","reviewer","notes"]
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(sample_id=row["sample_id"],label=row["label"],status="pending",
                                 notes="Source clip label only; review event boundaries."))

def normalize_raw(source_dir, output_dir, root, cfg):
    source_dir = inside(root, source_dir)
    output_dir = inside(root, output_dir)
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("Normalized output directory must be outside the raw source folder")
    sources = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not sources:
        raise ValueError(f"No supported video files found in folder: {source_dir}")

    planned = [
        (source, output_dir / source.relative_to(source_dir).with_suffix(".mp4"))
        for source in sources
    ]
    destinations = [destination.resolve() for _, destination in planned]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Multiple raw videos map to the same normalized output path")
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Normalized output already exists; move it or choose another output directory: "
            + existing[0]
        )

    results = []
    errors = []
    for source, destination in planned:
        try:
            normalized = _normalize_video(source, destination, cfg)
            results.append(
                {
                    "raw_path": source.relative_to(root).as_posix(),
                    "processed_path": destination.relative_to(root).as_posix(),
                    "sha256": checksum(source),
                    "processed_sha256": checksum(destination),
                    **normalized,
                }
            )
        except (OSError, ValueError) as exc:
            errors.append({"raw_path": source.relative_to(root).as_posix(), "error": str(exc)})
    return results, errors

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["download","prepare","audit","normalize-raw"])
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--catalog",default="data/metadata/catalog.jsonl")
    parser.add_argument("--config",default="configs/data.json")
    parser.add_argument("--source-dir",default="data/raw")
    parser.add_argument("--output-dir",default="data/processed/raw")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--max-file-mb",type=int,default=64)
    parser.add_argument("--dataset",choices=["airtlab","urfall"])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    local = root/"data/metadata/local"
    cfg = json.loads((root/args.config).read_text(encoding="utf-8-sig"))
    try:
        if args.command=="normalize-raw":
            results, errors = normalize_raw(args.source_dir,args.output_dir,root,cfg)
            write_rows(local/"raw_normalization.jsonl",results)
            write_rows(local/"raw_normalization_errors.jsonl",errors)
            print(json.dumps({"normalized":len(results),"errors":len(errors)},indent=2))
            return int(bool(errors))
        if args.command=="download":
            rows = read_rows(root/args.catalog)
            if len({r["sample_id"] for r in rows})!=len(rows):
                raise ValueError("Duplicate sample IDs")
            for r in rows:
                if r["label"] not in LABELS or not r.get("group_id") or not r.get("license_id") or not r["url"].startswith("https://"):
                    raise ValueError("Invalid catalog row")
                inside(root,r["raw_path"])
            for r in rows:
                inside(root,r["raw_path"]).parent.mkdir(parents=True,exist_ok=True)
            if args.dataset:
                rows = [r for r in rows if r["dataset"]==args.dataset]
            def task(row):
                try:
                    return download(row,root,args.max_file_mb*1024*1024)
                except Exception as exc:
                    return dict(row,download_status="error",error=str(exc))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results=[]
                for i,row in enumerate(pool.map(task,rows),1):
                    results.append(row)
                    if i%20==0 or i==len(rows):
                        print(f"Downloaded/checked {i}/{len(rows)}",flush=True)
            target=local/"downloads.jsonl"
            old=read_rows(target) if target.exists() else []
            merged={r["sample_id"]:r for r in old}
            merged.update({r["sample_id"]:r for r in results})
            write_rows(target,sorted(merged.values(),key=lambda r:r["sample_id"]))
            failed=sum(r["download_status"]=="error" for r in results)
            print(json.dumps(dict(requested=len(rows),errors=failed)))
            return int(bool(failed))
        if args.command=="prepare":
            rows=read_rows(local/"downloads.jsonl")
            catalog=read_rows(root/args.catalog)
            if {r["sample_id"] for r in rows}!={r["sample_id"] for r in catalog}:
                raise ValueError("Downloads must cover the complete selected catalog")
            if not rows or any(r["download_status"]=="error" for r in rows):
                raise ValueError("Fix missing/failed downloads first")
            def task(row):
                try:
                    return normalize(row,root,cfg)
                except Exception as exc:
                    return dict(sample_id=row["sample_id"],error=str(exc))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results=[]
                for i,row in enumerate(pool.map(task,rows),1):
                    results.append(row)
                    if i%20==0 or i==len(rows):
                        print(f"Prepared {i}/{len(rows)}",flush=True)
            failed=[r for r in results if "error" in r]
            write_rows(local/"preparation_errors.jsonl",failed)
            if failed:
                raise ValueError(f"{len(failed)} video errors; see preparation_errors.jsonl")
            rows=split_rows(results,cfg["split_ratios"],cfg["seed"],cfg["near_duplicate_hamming"])
            write_rows(local/"manifest.jsonl",rows)
            annotations(root/"data/annotations/local/events.csv",rows)
        else:
            rows=read_rows(local/"manifest.jsonl")
            catalog=read_rows(root/args.catalog)
            if {r["sample_id"] for r in rows}!={r["sample_id"] for r in catalog}:
                raise ValueError("Manifest does not cover the catalog")
            for row in rows:
                for key,hashkey in (("raw_path","sha256"),("processed_path","processed_sha256")):
                    if checksum(inside(root,row[key]))!=row[hashkey]:
                        raise ValueError("Checksum changed: "+row["sample_id"]+":"+key)
                inspect(inside(root,row["processed_path"]))
        report=audit(rows,cfg["near_duplicate_hamming"])
        local.mkdir(parents=True,exist_ok=True)
        (local/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({k:report[k] for k in ("sample_count","counts","errors","ready_for_final_evaluation")},indent=2))
        return int(bool(report["errors"]))
    except (ValueError,OSError,KeyError) as exc:
        print(f"Data pipeline error: {exc}",file=sys.stderr)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
