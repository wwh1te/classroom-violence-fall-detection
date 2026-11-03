
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_errors(
    predictions: list[dict[str, Any]] | list[tuple[str, int | str, str, str, float]],
    *,
    output_dir: str | Path | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    if not predictions:
        summary = {
            "total_errors": 0,
            "class_error_counts": {},
            "confusions": {},
        }

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            (output_path / "error_analysis.csv").write_text(
                "video_id,frame_index,ground_truth,predicted,confidence\n",
                encoding="utf-8",
            )
            (output_path / "error_summary.md").write_text(
                "# Error analysis\n\nNo prediction errors were recorded.\n",
                encoding="utf-8",
            )

        return summary

    rows: list[dict[str, Any]] = []

    for index, item in enumerate(predictions):
        if isinstance(item, tuple):
            if len(item) != 5:
                raise ValueError(
                    f"Prediction tuple at index {index} must contain 5 values."
                )

            video_id, frame_index, truth, predicted, confidence = item

        elif isinstance(item, dict):
            video_id = item.get("video_id", "unknown")
            frame_index = item.get("frame_index", "-")
            truth = item.get("ground_truth", "unknown")
            predicted = item.get("predicted", "unknown")
            confidence = item.get("confidence", 0.0)

        else:
            raise TypeError(
                f"Prediction at index {index} must be a dictionary or tuple."
            )

        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid confidence at prediction index {index}."
            ) from exc

        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Confidence at prediction index {index} must be between 0 and 1."
            )

        truth = str(truth)
        predicted = str(predicted)

        if label_names is not None:
            if truth not in label_names or predicted not in label_names:
                raise ValueError(
                    f"Unknown label at prediction index {index}: "
                    f"{truth} -> {predicted}"
                )

        rows.append({
            "video_id": str(video_id),
            "frame_index": frame_index,
            "ground_truth": truth,
            "predicted": predicted,
            "confidence": confidence,
        })

    error_rows = [
        row for row in rows
        if row["ground_truth"] != row["predicted"]
    ]

    class_error_counts = Counter()
    confusions = Counter()

    for row in error_rows:
        class_error_counts[row["ground_truth"]] += 1
        confusions[
            (row["ground_truth"], row["predicted"])
        ] += 1

    summary = {
        "total_errors": len(error_rows),
        "class_error_counts": dict(class_error_counts),
        "confusions": {
            f"{left}->{right}": count
            for (left, right), count in confusions.items()
        },
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(error_rows)

        if frame.empty:
            frame = pd.DataFrame(columns=[
                "video_id",
                "frame_index",
                "ground_truth",
                "predicted",
                "confidence",
            ])

        frame.to_csv(
            output_path / "error_analysis.csv",
            index=False,
        )

        markdown_lines = [
            "# Error analysis",
            "",
            f"Total misclassifications: {len(error_rows)}",
            "",
        ]

        if class_error_counts:
            markdown_lines.append("## Class-wise error counts")
            markdown_lines.extend(
                f"- {label}: {count}"
                for label, count in class_error_counts.most_common()
            )

            markdown_lines.append("")
            markdown_lines.append("## Most frequent confusion pairs")

            for (left, right), count in confusions.most_common(10):
                markdown_lines.append(
                    f"- {left} -> {right}: {count}"
                )
        else:
            markdown_lines.append("No prediction errors were recorded.")

        (output_path / "error_summary.md").write_text(
            "\n".join(markdown_lines) + "\n",
            encoding="utf-8",
        )

    return summary


__all__ = ["analyze_errors"]
