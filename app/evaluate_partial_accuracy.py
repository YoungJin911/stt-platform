from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =========================================================
# NORMALIZATION
# =========================================================

def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )


def normalize_text(text: str) -> str:
    text = str(text or "")

    text = re.sub(
        r"[^\w가-힣\s]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def normalize_chars_no_spaces(text: str) -> list[str]:
    return list(
        normalize_text(text).replace(
            " ",
            "",
        )
    )


# =========================================================
# EDIT DISTANCE
# =========================================================

@dataclass
class Counts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return (
            self.substitutions
            + self.deletions
            + self.insertions
        )

    @property
    def rate(self) -> float:
        if self.reference_units == 0:
            return 0.0 if self.errors == 0 else 1.0

        return self.errors / self.reference_units

    @property
    def accuracy(self) -> float:
        return max(
            0.0,
            1.0 - self.rate,
        )


def edit_counts(
    reference: list[str],
    hypothesis: list[str],
) -> Counts:
    n = len(reference)
    m = len(hypothesis)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    back = [
        [""] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "M"
                continue

            substitution = dp[i - 1][j - 1] + 1
            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1

            best = min(
                substitution,
                deletion,
                insertion,
            )

            dp[i][j] = best

            if substitution == best:
                back[i][j] = "S"
            elif deletion == best:
                back[i][j] = "D"
            else:
                back[i][j] = "I"

    substitutions = 0
    deletions = 0
    insertions = 0

    i = n
    j = m

    while i > 0 or j > 0:
        op = back[i][j]

        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1
        else:
            if i > 0:
                deletions += 1
                i -= 1
            elif j > 0:
                insertions += 1
                j -= 1

    return Counts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=n,
    )


# =========================================================
# TRANSCRIPT STRUCTURE
# =========================================================

def extract_segments(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [
            item
            for item in data
            if (
                isinstance(item, dict)
                and "start" in item
                and "end" in item
                and "text" in item
            )
        ]

    if isinstance(data, dict):
        for key in (
            "segments",
            "items",
            "results",
        ):
            value = data.get(key)

            if isinstance(value, list):
                result = extract_segments(value)

                if result:
                    return result

    return []


def overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(
        0.0,
        min(end_a, end_b)
        - max(start_a, start_b),
    )


def usable_words(segment: dict) -> list[dict]:
    words = segment.get(
        "words",
        [],
    )

    if not isinstance(words, list):
        return []

    result = []

    for word in words:
        if not isinstance(word, dict):
            continue

        word_text = str(
            word.get(
                "word",
                "",
            )
        ).strip()

        if not word_text:
            continue

        try:
            start = float(
                word.get(
                    "start"
                )
            )

            end = float(
                word.get(
                    "end"
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if end < start:
            continue

        result.append(
            {
                "start": start,
                "end": end,
                "word": word_text,
            }
        )

    return result


@dataclass
class WindowExtraction:
    text: str
    mode_counts: dict[str, int]


def transcript_text_for_window(
    segments: list[dict],
    start: float,
    end: float,
) -> WindowExtraction:
    """
    Hybrid extraction strategy.

    1. If a segment is fully contained in the benchmark window:
       use segment["text"] exactly. This preserves corrected text.

    2. If a segment crosses a benchmark boundary:
       use word timestamps and include only words whose midpoint lies
       inside the benchmark window.

       This prevents the old false-error case where a tiny overlap at
       the end of a segment pulled the next sentence into the benchmark.

    3. If a boundary-crossing segment has no usable word timestamps:
       use the whole segment only when its midpoint is inside the window.

    This gives precise clipping without losing corrected text for the
    normal fully-contained case.
    """

    pieces: list[
        tuple[
            float,
            str,
            str,
        ]
    ] = []

    mode_counts = {
        "full_segment": 0,
        "word_clip": 0,
        "midpoint_fallback": 0,
    }

    for segment in segments:
        try:
            seg_start = float(
                segment.get(
                    "start",
                    0.0,
                )
            )

            seg_end = float(
                segment.get(
                    "end",
                    seg_start,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if overlap(
            seg_start,
            seg_end,
            start,
            end,
        ) <= 0.0:
            continue

        segment_text = str(
            segment.get(
                "text",
                "",
            )
        ).strip()

        if not segment_text:
            continue

        fully_contained = (
            seg_start >= start
            and
            seg_end <= end
        )

        if fully_contained:
            pieces.append(
                (
                    seg_start,
                    segment_text,
                    "full_segment",
                )
            )

            mode_counts[
                "full_segment"
            ] += 1

            continue

        words = usable_words(
            segment
        )

        clipped_words = []

        for word in words:
            midpoint = (
                float(
                    word["start"]
                )
                +
                float(
                    word["end"]
                )
            ) / 2.0

            if (
                start
                <=
                midpoint
                <=
                end
            ):
                clipped_words.append(
                    word
                )

        if clipped_words:
            clipped_text = " ".join(
                str(
                    word[
                        "word"
                    ]
                ).strip()
                for word in clipped_words
                if str(
                    word[
                        "word"
                    ]
                ).strip()
            )

            if clipped_text:
                pieces.append(
                    (
                        float(
                            clipped_words[0][
                                "start"
                            ]
                        ),
                        clipped_text,
                        "word_clip",
                    )
                )

                mode_counts[
                    "word_clip"
                ] += 1

                continue

        segment_midpoint = (
            seg_start
            +
            seg_end
        ) / 2.0

        if (
            start
            <=
            segment_midpoint
            <=
            end
        ):
            pieces.append(
                (
                    seg_start,
                    segment_text,
                    "midpoint_fallback",
                )
            )

            mode_counts[
                "midpoint_fallback"
            ] += 1

    pieces.sort(
        key=lambda item: (
            item[0],
            item[2],
        )
    )

    return WindowExtraction(
        text=" ".join(
            text
            for _, text, _ in pieces
        ),
        mode_counts=mode_counts,
    )


# =========================================================
# REPORT HELPERS
# =========================================================

def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_counts(
    label: str,
    counts: Counts,
) -> None:
    print(
        f"{label:<18}: "
        f"{pct(counts.rate):>8} "
        f"| accuracy {pct(counts.accuracy):>8} "
        f"| S/D/I "
        f"{counts.substitutions}/"
        f"{counts.deletions}/"
        f"{counts.insertions} "
        f"| N={counts.reference_units}"
    )


@dataclass
class Aggregate:
    ref_words: list[str]
    hyp_words: list[str]
    ref_chars: list[str]
    hyp_chars: list[str]

    @classmethod
    def empty(cls) -> "Aggregate":
        return cls(
            ref_words=[],
            hyp_words=[],
            ref_chars=[],
            hyp_chars=[],
        )

    def add(
        self,
        reference: str,
        hypothesis: str,
    ) -> None:
        self.ref_words.extend(
            normalize_words(
                reference
            )
        )

        self.hyp_words.extend(
            normalize_words(
                hypothesis
            )
        )

        self.ref_chars.extend(
            normalize_chars_no_spaces(
                reference
            )
        )

        self.hyp_chars.extend(
            normalize_chars_no_spaces(
                hypothesis
            )
        )

    def metrics(
        self,
    ) -> tuple[
        Counts,
        Counts,
    ]:
        return (
            edit_counts(
                self.ref_words,
                self.hyp_words,
            ),
            edit_counts(
                self.ref_chars,
                self.hyp_chars,
            ),
        )


def print_aggregate(
    title: str,
    aggregate: Aggregate,
) -> tuple[
    Counts,
    Counts,
]:
    wer, cer = aggregate.metrics()

    print()
    print(
        "----------------------------------------"
    )

    print(
        title
    )

    print_counts(
        "WER",
        wer,
    )

    print_counts(
        "CER(no spaces)",
        cer,
    )

    return (
        wer,
        cer,
    )


# =========================================================
# MAIN
# =========================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4-D.2 partial STT benchmark with precise "
            "window clipping and confidence-separated totals."
        )
    )

    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--raw",
        type=Path,
    )

    parser.add_argument(
        "--corrected",
        type=Path,
    )

    args = parser.parse_args()

    if (
        args.raw is None
        and
        args.corrected is None
    ):
        print(
            "[ERROR] provide --raw and/or --corrected",
            file=sys.stderr,
        )

        return 2

    benchmark = load_json(
        args.benchmark
    )

    benchmark_segments = benchmark.get(
        "segments",
        [],
    )

    if not benchmark_segments:
        print(
            "[ERROR] benchmark has no segments",
            file=sys.stderr,
        )

        return 2

    sources = []

    if args.raw is not None:
        sources.append(
            (
                "RAW",
                extract_segments(
                    load_json(
                        args.raw
                    )
                ),
            )
        )

    if args.corrected is not None:
        sources.append(
            (
                "CORRECTED",
                extract_segments(
                    load_json(
                        args.corrected
                    )
                ),
            )
        )

    source_summaries: dict[
        str,
        dict[
            str,
            tuple[
                Counts,
                Counts,
            ],
        ],
    ] = {}

    for source_name, source_segments in sources:
        aggregates = {
            "high": Aggregate.empty(),
            "all": Aggregate.empty(),
        }

        print()
        print(
            "========================================"
        )

        print(
            f"  PARTIAL BENCHMARK D.2 - {source_name}"
        )

        print(
            "========================================"
        )

        for item in benchmark_segments:
            seg_id = str(
                item.get(
                    "id",
                    "segment",
                )
            )

            confidence = str(
                item.get(
                    "confidence",
                    "medium",
                )
            ).lower()

            start = float(
                item[
                    "start"
                ]
            )

            end = float(
                item[
                    "end"
                ]
            )

            reference = str(
                item[
                    "reference"
                ]
            ).strip()

            extraction = (
                transcript_text_for_window(
                    source_segments,
                    start,
                    end,
                )
            )

            hypothesis = (
                extraction.text
            )

            ref_words = normalize_words(
                reference
            )

            hyp_words = normalize_words(
                hypothesis
            )

            ref_chars = normalize_chars_no_spaces(
                reference
            )

            hyp_chars = normalize_chars_no_spaces(
                hypothesis
            )

            wer = edit_counts(
                ref_words,
                hyp_words,
            )

            cer = edit_counts(
                ref_chars,
                hyp_chars,
            )

            aggregates[
                "all"
            ].add(
                reference,
                hypothesis,
            )

            if confidence == "high":
                aggregates[
                    "high"
                ].add(
                    reference,
                    hypothesis,
                )

            print()
            print(
                f"[{seg_id}] "
                f"{start:.2f} ~ {end:.2f} "
                f"| confidence={confidence}"
            )

            print_counts(
                "WER",
                wer,
            )

            print_counts(
                "CER(no spaces)",
                cer,
            )

            print(
                "EXTRACT:",
                extraction.mode_counts,
            )

            print(
                "REF:",
                normalize_text(
                    reference
                ),
            )

            print(
                "HYP:",
                normalize_text(
                    hypothesis
                ),
            )

        high_metrics = print_aggregate(
            f"{source_name} HIGH-CONFIDENCE TOTAL",
            aggregates[
                "high"
            ],
        )

        all_metrics = print_aggregate(
            f"{source_name} ALL PSEUDO-GT TOTAL",
            aggregates[
                "all"
            ],
        )

        source_summaries[
            source_name
        ] = {
            "high": high_metrics,
            "all": all_metrics,
        }

    if (
        "RAW" in source_summaries
        and
        "CORRECTED" in source_summaries
    ):
        print()
        print(
            "========================================"
        )

        print(
            "  PHASE 4-D.2 IMPROVEMENT"
        )

        print(
            "========================================"
        )

        for group_name, label in (
            (
                "high",
                "HIGH-CONFIDENCE",
            ),
            (
                "all",
                "ALL PSEUDO-GT",
            ),
        ):
            raw_wer, raw_cer = (
                source_summaries[
                    "RAW"
                ][
                    group_name
                ]
            )

            corrected_wer, corrected_cer = (
                source_summaries[
                    "CORRECTED"
                ][
                    group_name
                ]
            )

            print()
            print(
                label
            )

            print(
                "WER improvement : "
                f"{(raw_wer.rate - corrected_wer.rate) * 100:+.2f} %p"
            )

            print(
                "CER improvement : "
                f"{(raw_cer.rate - corrected_cer.rate) * 100:+.2f} %p"
            )

            print(
                "Final WER        : "
                f"{pct(corrected_wer.rate)}"
            )

            print(
                "Final CER        : "
                f"{pct(corrected_cer.rate)}"
            )

            print(
                "Final char acc.  : "
                f"{pct(corrected_cer.accuracy)}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
