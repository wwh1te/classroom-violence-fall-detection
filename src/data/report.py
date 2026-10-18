"""Generate measured dataset statistics, split evidence, and a Vietnamese report."""
import csv
import json
from collections import Counter
from pathlib import Path
import argparse
from .pipeline import read_rows, checksum

def generate(root):
    root=Path(root)
    local=root/"data/metadata/local"
    rows=read_rows(local/"manifest.jsonl")
    audit=json.loads((local/"audit.json").read_text(encoding="utf-8"))
    reportdir=root/"docs/reports"
    reportdir.mkdir(parents=True,exist_ok=True)
    (reportdir/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    stats={
        "date":"2026-10-17", "videos":len(rows),
        "source_counts":dict(Counter(r["dataset"]+"/"+r["label"] for r in rows)),
        "recording_groups":len({r["group_id"] for r in rows}),
        "split_components":len({r["split_component_id"] for r in rows}),
        "split_counts":audit["counts"],
        "raw_bytes":sum((root/r["raw_path"]).stat().st_size for r in rows),
        "processed_bytes":sum((root/r["processed_path"]).stat().st_size for r in rows),
        "source_duration_seconds":sum(r["original"]["duration_s"] for r in rows),
        "processed_duration_seconds":sum(r["video"]["duration_s"] for r in rows),
        "processed_frames":sum(r["video"]["frames"] for r in rows),
        "audit_errors":audit["errors"],
        "near_duplicate_candidates":len(audit["near_duplicate_candidates"]),
        "cross_split_near_duplicate_candidates":sum(x["cross_split"] for x in audit["near_duplicate_candidates"]),
        "pending_clip_review":sum(r["label_status"]!="reviewed" for r in rows),
        "ready_for_final_evaluation":audit["ready_for_final_evaluation"],
        "manifest_sha256":checksum(local/"manifest.jsonl"),
        "catalog_sha256":checksum(root/"data/metadata/catalog.jsonl"),
        "config":json.loads((root/"configs/data.json").read_text(encoding="utf-8-sig")),
    }
    (reportdir/"dataset_statistics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    with (root/"data/metadata/splits.csv").open("w",encoding="utf-8",newline="") as stream:
        fields=["sample_id","dataset","label","group_id","camera_id","split_component_id","split","sha256","processed_sha256"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines=[
        "# Mô tả dữ liệu và kết quả chuẩn bị",
        "",
        "Ngày chạy: 17/10/2026. Số liệu dưới đây đo trên các file đã tải và giải mã thành công, không phải số lượng mục tiêu.",
        "",
        f"Bộ dữ liệu khởi đầu gồm **{len(rows)} video**, thuộc **{stats['recording_groups']} nhóm bản ghi** từ AIRTLab và UR Fall. Video được ánh xạ sang ba nhãn normal, violence, fall theo nhãn cấp clip do nguồn cung cấp. Nguồn và điều khoản được ghi tại [dataset_sources.md](../dataset_sources.md).",
        "",
        "| Nguồn / lớp | Video |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {v} |" for k,v in sorted(stats["source_counts"].items())]
    lines += [
        "",
        f"Dung lượng gốc: {stats['raw_bytes']/1024**3:.3f} GiB; dung lượng chuẩn hóa: {stats['processed_bytes']/1024**3:.3f} GiB. Tổng thời lượng bản xử lý: {stats['processed_duration_seconds']/60:.2f} phút, {stats['processed_frames']:,} frame. Thời lượng tính theo từng video/góc quay, không phải thời gian của các sự kiện duy nhất.",
        "",
        "Video được giải mã tuần tự, kiểm số frame, rồi chuẩn hóa thành RGB 640×360 ở 15 fps bằng letterbox. Với UR Fall MP4 preview, chỉ lấy nửa RGB bên phải. Không đưa ảnh depth hoặc âm thanh vào đặc trưng. Bản MP4 đầu ra được đọc lại để kiểm số frame và kích thước.",
        "",
        "## Chia tập và kiểm tra",
        "",
        "Chia theo nhóm sự kiện, kết nối cả video gốc, bản trùng byte và toàn bộ ứng viên gần trùng bằng connected components, phân tầng theo nguồn/lớp, seed 42 và tỷ lệ mục tiêu 70/15/15. Hai camera AIRTLab cùng sự kiện luôn đi chung. Tỷ lệ thực tế có làm tròn theo nhóm.",
        "",
        "| Tập | normal | violence | fall | Tổng |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train","val","test"):
        values=[audit["counts"].get(split+"/"+label,0) for label in ("normal","violence","fall")]
        lines.append(f"| {split} | "+" | ".join(map(str,values+[sum(values)]))+" |")
    lines += [
        "",
        f"Audit ghi nhận {len(audit['errors'])} lỗi xác định; {len(audit['near_duplicate_candidates'])} cặp ứng viên gần trùng, trong đó {stats['cross_split_near_duplicate_candidates']} cặp xuyên tập. Chi tiết nằm trong data/metadata/local/audit.json. Ứng viên dHash cần xem lại; không xem số lỗi bằng 0 là bằng chứng độc lập người/bối cảnh.",
        "",
        "## Giới hạn và trạng thái nghiệm thu",
        "",
        f"- Có {stats['pending_clip_review']} video chưa được người gán nhãn duyệt lại. Nhãn hiện tại là nhãn cấp clip của nguồn, không phải ground truth từng frame/người.",
        "- events.csv là template pending. Chưa đo thời điểm sự kiện, đồng thuận giữa người gán nhãn hoặc độ chính xác nhãn thời gian. Chỉ có hướng dẫn và quy trình duyệt đã xây dựng.",
        "- Chưa xác lập split độc lập người tham gia hoặc bối cảnh vì thiếu metadata phù hợp. Không dùng kết quả từ split này để tuyên bố tổng quát sang người hoặc lớp học mới.",
        "- Dữ liệu không đại diện riêng lớp học. Nguồn và lớp có tương quan: violence đến từ AIRTLab, fall từ UR Fall; mô hình có thể học nền/nguồn thay vì hành động. Cần bổ sung dữ liệu cùng bối cảnh cho cả ba lớp và đánh giá trên lớp học được đồng thuận.",
        "- UR Fall dùng video preview có nén. Không so trực tiếp với nghiên cứu dùng PNG RGB gốc hoặc toàn bộ camera mà không ghi khác biệt.",
        f"- ready_for_final_evaluation = {str(audit['ready_for_final_evaluation']).lower()}. Chưa thể coi bộ sự kiện là nhãn đánh giá cuối.",
        "",
        "Mã nguồn, nguồn dữ liệu, tải/chuẩn hóa, hướng dẫn nhãn và kiểm tra tự động đã có. Trước khi đóng issue, nhóm phải xác nhận phạm vi nghiệm thu nhãn cấp clip, xem lại ứng viên gần trùng xuyên tập; nếu dùng nhãn thời gian trong báo cáo, hoàn tất gán nhãn và kiểm tra độc lập.",
        "",
        "## Tái lập",
        "",
        "Chạy download → prepare → audit theo docs/data_workflow.md, sau đó python -m src.data.report. data/metadata/splits.csv lưu split và checksum để đối chiếu phiên bản. dataset_statistics.json lưu số liệu và checksum manifest/catalog. Mã kiểm thử không dùng video thật làm fixture.",
        "",
        "Trích dẫn dataset và điều khoản được ghi trong [dataset_sources.md](../dataset_sources.md).",
    ]
    (reportdir/"dataset_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    return stats

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    generate(parser.parse_args().root)
