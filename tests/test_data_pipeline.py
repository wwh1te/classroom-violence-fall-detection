import hashlib
from pathlib import Path
import cv2
import numpy as np
import pytest
from src.data.pipeline import (inside, checksum, split_rows, audit, normalize,
                               inspect, letterbox, annotations, download)

def rows():
    return [dict(sample_id=f"{label}-{i}-{cam}", dataset="demo", label=label,
                 group_id=f"{label}-{i}", parent_video_id=f"{label}-{i}",
                 sha256=f"{label}-{i}-{cam}", processed_sha256=f"p-{label}-{i}-{cam}",
                 video={"duration_s":2}, dhashes=[])
            for label in ("normal","violence","fall") for i in range(10) for cam in range(2)]

def test_recordings_and_camera_views_stay_together():
    result=split_rows(rows())
    assert not audit(result)["errors"]
    assert {r["split"] for r in result} == {"train","val","test"}
    for group in {r["group_id"] for r in result}:
        assert len({r["split"] for r in result if r["group_id"]==group})==1

def test_split_is_order_independent_and_seeded():
    assert split_rows(rows()) == split_rows(list(reversed(rows())))
    assert split_rows(rows(),seed=42) != split_rows(rows(),seed=43)

def test_duplicates_join_recording_components():
    data=rows()
    data[0]["sha256"]=data[3]["sha256"]
    result=split_rows(data)
    assert result[0]
    assert not [e for e in audit(result)["errors"] if "cross_split" in e]

def test_leakage_is_detected_even_without_same_group():
    data=split_rows(rows())
    a=next(r for r in data if r["split"]=="train")
    b=next(r for r in data if r["split"]=="test")
    b["sha256"]=a["sha256"]
    assert any("cross_split_sha256" in e for e in audit(data)["errors"])

@pytest.mark.parametrize("ratios",[(1,0,0),(.5,.2,.2),(-.1,.5,.6)])
def test_invalid_ratios_fail(ratios):
    with pytest.raises(ValueError):
        split_rows(rows(),ratios)

def test_small_strata_fail_instead_of_silent_empty_test():
    with pytest.raises(ValueError):
        split_rows(rows()[:4])

@pytest.mark.parametrize("name",["../escape.mp4","C:/outside/video.mp4"])
def test_paths_cannot_escape(tmp_path,name):
    with pytest.raises(ValueError):
        inside(tmp_path,name)

def test_letterbox_preserves_geometry():
    image=np.full((100,100,3),255,dtype=np.uint8)
    out=letterbox(image,640,360)
    assert out.shape==(360,640,3)
    assert np.all(out[:,:140]==0)
    assert np.all(out[:,140:500]==255)
    assert np.all(out[:,500:]==0)

def make_video(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),30,(640,240))
    assert writer.isOpened()
    for i in range(60):
        frame=np.zeros((240,640,3),dtype=np.uint8)
        frame[:,:320]=(255,0,0)
        frame[:,320:]=(0,0,200+i%40)
        writer.write(frame)
    writer.release()

def test_real_video_normalizes_rgb_crop_fps_and_duration(tmp_path):
    raw=tmp_path/"data/raw/fall/a.mp4"
    make_video(raw)
    row=dict(sample_id="a",dataset="urfall",label="fall",raw_path="data/raw/fall/a.mp4",
             sha256=checksum(raw),crop="right_half_rgb")
    result=normalize(row,tmp_path,dict(width=640,height=360,fps=15))
    assert result["video"]["frames"]==30
    assert result["video"]["fps"]==15
    assert result["video"]["duration_s"]==2
    cap=cv2.VideoCapture(str(tmp_path/result["processed_path"]))
    ok,frame=cap.read()
    cap.release()
    assert ok and frame[180,320,2]>150 and frame[180,320,0]<20
    assert inspect(raw)["frames"]==60

def test_normalize_raw_folder_keeps_originals_and_relative_layout(tmp_path):
    import json
    from src.data.pipeline import main

    raw = tmp_path / "data/raw/fall"
    source = raw / "nested/input.mp4"
    make_video(source)
    original = source.read_bytes()
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "data.json").write_text(
        json.dumps(dict(width=320,height=180,fps=15)),
        encoding="utf-8",
    )

    assert main(["normalize-raw", "--root", str(tmp_path)]) == 0

    output = tmp_path / "data/processed/raw/fall/nested/input.mp4"
    assert output.is_file()
    assert source.read_bytes() == original
    assert inspect(output)["frames"] == 30
    records = (tmp_path / "data/metadata/local/raw_normalization.jsonl")
    row = json.loads(records.read_text(encoding="utf-8").splitlines()[0])
    assert row["raw_path"] == "data/raw/fall/nested/input.mp4"
    assert row["processed_path"] == "data/processed/raw/fall/nested/input.mp4"

def test_normalize_raw_does_not_overwrite_existing_output(tmp_path):
    import json
    from src.data.pipeline import main

    source = tmp_path / "data/raw/input.mp4"
    make_video(source)
    output = tmp_path / "data/processed/raw/input.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"keep")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "data.json").write_text(json.dumps({}), encoding="utf-8")

    assert main(["normalize-raw", "--root", str(tmp_path)]) == 1
    assert output.read_bytes() == b"keep"

def test_corrupt_video_rejected(tmp_path):
    bad=tmp_path/"bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        inspect(bad)

def test_raw_tampering_rejected(tmp_path):
    raw=tmp_path/"raw.mp4"
    make_video(raw)
    with pytest.raises(ValueError,match="checksum"):
        normalize(dict(sample_id="a",dataset="x",label="fall",raw_path="raw.mp4",sha256="wrong"),
                  tmp_path,dict(width=640,height=360,fps=15))

def test_annotation_template_never_overwrites_human_work(tmp_path):
    path=tmp_path/"events.csv"
    annotations(path,[dict(sample_id="a",label="fall")])
    path.write_text("human-reviewed",encoding="utf-8")
    annotations(path,[])
    assert path.read_text(encoding="utf-8")=="human-reviewed"

def test_near_duplicates_are_candidates_and_block_evaluation():
    data=split_rows(rows())
    a=next(r for r in data if r["split"]=="train")
    b=next(r for r in data if r["split"]=="test")
    a["dhashes"]=b["dhashes"]=["0000000000000000"]*5
    result=audit(data)
    assert result["near_duplicate_candidates"][0]["cross_split"]
    assert not result["ready_for_final_evaluation"]

def test_git_blob_hash_matches_git_definition(tmp_path):
    path=tmp_path/"blob"
    path.write_bytes(b"abc")
    assert checksum(path,True)==hashlib.sha1(b"blob 3\0abc").hexdigest()

def test_empty_audit_fails():
    assert audit([])["errors"]


def test_prepare_rejects_incomplete_catalog(tmp_path):
    import json
    from src.data.pipeline import main,write_rows
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs/data.json").write_text(json.dumps(dict(width=640,height=360,fps=15,seed=42,split_ratios=[.7,.15,.15])))
    write_rows(tmp_path/"data/metadata/catalog.jsonl",[{"sample_id":"a"},{"sample_id":"b"}])
    write_rows(tmp_path/"data/metadata/local/downloads.jsonl",[{"sample_id":"a","download_status":"cached"}])
    assert main(["prepare","--root",str(tmp_path)])==1

def test_audit_rejects_incomplete_manifest(tmp_path):
    import json
    from src.data.pipeline import main,write_rows
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs/data.json").write_text(json.dumps({}))
    write_rows(tmp_path/"data/metadata/catalog.jsonl",[{"sample_id":"a"},{"sample_id":"b"}])
    write_rows(tmp_path/"data/metadata/local/manifest.jsonl",[{"sample_id":"a"}])
    assert main(["audit","--root",str(tmp_path)])==1


def test_near_duplicate_candidates_stay_in_one_split():
    data=rows()
    data[0]["dhashes"]=data[7]["dhashes"]=["0000000000000000"]*5
    result=split_rows(data)
    candidates=audit(result)["near_duplicate_candidates"]
    assert candidates
    assert all(not pair["cross_split"] for pair in candidates)
    assert not audit(result)["errors"]
