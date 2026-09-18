from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =========================================================
# TEXT LOADING
# =========================================================

SRT_TIMESTAMP_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*"
    r"\d{2}:\d{2}:\d{2},\d{3}\s*$"
)

PUNCT_RE = re.compile(
    r"[^\w가-힣\s]",
    flags=re.UNICODE,
)


def read_text_file(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def extract_text_from_srt(
    text: str,
) -> str:
    lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.isdigit():
            continue

        if SRT_TIMESTAMP_RE.match(line):
            continue

        lines.append(line)

    return " ".join(lines)


def extract_segments_from_json(
    data: Any,
) -> list[str]:
    """
    Extract transcript text from common STT JSON shapes.

    Preferred shapes:
      [
        {"start": ..., "end": ..., "text": "..."},
        ...
      ]

      {
        "segments": [
          {"text": "..."},
          ...
        ]
      }

    We intentionally do NOT recursively concatenate every string in the
    JSON because logs may contain raw_text, corrected_text, rules, etc.
    That would double-count transcript content.
    """

    if isinstance(data, list):
        texts: list[str] = []

        for item in data:
            if isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())

            elif isinstance(item, str) and item.strip():
                texts.append(item.strip())

        return texts

    if isinstance(data, dict):
        for key in (
            "segments",
            "transcript",
            "results",
            "items",
        ):
            value = data.get(key)

            if isinstance(value, list):
                texts = extract_segments_from_json(value)

                if texts:
                    return texts

        text = data.get("text")

        if isinstance(text, str) and text.strip():
            return [text.strip()]

    return []


def load_transcript(
    path: Path,
) -> str:
    suffix = path.suffix.lower()

    if suffix == ".srt":
        return extract_text_from_srt(
            read_text_file(path)
        )

    if suffix == ".json":
        try:
            data = json.loads(
                read_text_file(path)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON: {path}: {exc}"
            ) from exc

        texts = extract_segments_from_json(data)

        if not texts:
            raise ValueError(
                f"Could not find transcript text in JSON: {path}"
            )

        return " ".join(texts)

    return read_text_file(path)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_common(
    text: str,
    *,
    remove_punctuation: bool,
) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    )

    text = text.replace(
        "\u200b",
        "",
    )

    text = text.replace(
        "\ufeff",
        "",
    )

    if remove_punctuation:
        text = PUNCT_RE.sub(
            " ",
            text,
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalize_for_wer(
    text: str,
    *,
    remove_punctuation: bool,
) -> list[str]:
    text = normalize_common(
        text,
        remove_punctuation=remove_punctuation,
    )

    if not text:
        return []

    return text.split()


def normalize_for_cer(
    text: str,
    *,
    remove_punctuation: bool,
    keep_spaces: bool,
) -> list[str]:
    text = normalize_common(
        text,
        remove_punctuation=remove_punctuation,
    )

    if not keep_spaces:
        text = text.replace(
            " ",
            "",
        )

    return list(text)


# =========================================================
# EDIT DISTANCE WITH S / D / I COUNTS
# =========================================================

@dataclass
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return (
            self.substitutions
            +
            self.deletions
            +
            self.insertions
        )

    @property
    def error_rate(self) -> float:
        if self.reference_units == 0:
            return 0.0 if self.errors == 0 else 1.0

        return (
            self.errors
            /
            self.reference_units
        )

    @property
    def accuracy(self) -> float:
        return max(
            0.0,
            1.0 - self.error_rate,
        )


def edit_counts(
    reference: list[str],
    hypothesis: list[str],
) -> EditCounts:
    """
    Levenshtein distance with backtrace counts.

    Costs:
      match        0
      substitution 1
      deletion     1
      insertion    1
    """

    n = len(reference)
    m = len(hypothesis)

    # dp[i][j] = minimum edit distance for ref[:i], hyp[:j]
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
        ref_unit = reference[i - 1]

        for j in range(1, m + 1):
            hyp_unit = hypothesis[j - 1]

            if ref_unit == hyp_unit:
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

            # Deterministic tie-breaking:
            # prefer substitution, then deletion, then insertion.
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
            # Defensive fallback for unexpected empty backtrace cells.
            if i > 0:
                deletions += 1
                i -= 1

            elif j > 0:
                insertions += 1
                j -= 1

    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=n,
    )


# =========================================================
# METRICS
# =========================================================

@dataclass
class EvaluationResult:
    label: str
    wer: EditCounts
    cer_no_spaces: EditCounts
    cer_with_spaces: EditCounts

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "wer": {
                "rate": self.wer.error_rate,
                "accuracy": self.wer.accuracy,
                "substitutions": self.wer.substitutions,
                "deletions": self.wer.deletions,
                "insertions": self.wer.insertions,
                "reference_units": self.wer.reference_units,
            },
            "cer_no_spaces": {
                "rate": self.cer_no_spaces.error_rate,
                "accuracy": self.cer_no_spaces.accuracy,
                "substitutions": self.cer_no_spaces.substitutions,
                "deletions": self.cer_no_spaces.deletions,
                "insertions": self.cer_no_spaces.insertions,
                "reference_units": self.cer_no_spaces.reference_units,
            },
            "cer_with_spaces": {
                "rate": self.cer_with_spaces.error_rate,
                "accuracy": self.cer_with_spaces.accuracy,
                "substitutions": self.cer_with_spaces.substitutions,
                "deletions": self.cer_with_spaces.deletions,
                "insertions": self.cer_with_spaces.insertions,
                "reference_units": self.cer_with_spaces.reference_units,
            },
        }


def evaluate(
    *,
    label: str,
    reference_text: str,
    hypothesis_text: str,
    remove_punctuation: bool,
) -> EvaluationResult:

    ref_words = normalize_for_wer(
        reference_text,
        remove_punctuation=remove_punctuation,
    )

    hyp_words = normalize_for_wer(
        hypothesis_text,
        remove_punctuation=remove_punctuation,
    )

    ref_chars_no_spaces = normalize_for_cer(
        reference_text,
        remove_punctuation=remove_punctuation,
        keep_spaces=False,
    )

    hyp_chars_no_spaces = normalize_for_cer(
        hypothesis_text,
        remove_punctuation=remove_punctuation,
        keep_spaces=False,
    )

    ref_chars_with_spaces = normalize_for_cer(
        reference_text,
        remove_punctuation=remove_punctuation,
        keep_spaces=True,
    )

    hyp_chars_with_spaces = normalize_for_cer(
        hypothesis_text,
        remove_punctuation=remove_punctuation,
        keep_spaces=True,
    )

    return EvaluationResult(
        label=label,
        wer=edit_counts(
            ref_words,
            hyp_words,
        ),
        cer_no_spaces=edit_counts(
            ref_chars_no_spaces,
            hyp_chars_no_spaces,
        ),
        cer_with_spaces=edit_counts(
            ref_chars_with_spaces,
            hyp_chars_with_spaces,
        ),
    )


# =========================================================
# REPORTING
# =========================================================

def pct(
    value: float,
) -> str:
    return f"{value * 100:.2f}%"


def print_metric(
    name: str,
    counts: EditCounts,
) -> None:
    print(
        f"{name:<22}: "
        f"{pct(counts.error_rate):>8} "
        f"| accuracy {pct(counts.accuracy):>8} "
        f"| S/D/I "
        f"{counts.substitutions}/"
        f"{counts.deletions}/"
        f"{counts.insertions} "
        f"| N={counts.reference_units}"
    )


def print_result(
    result: EvaluationResult,
) -> None:
    print()
    print(
        "========================================"
    )
    print(
        f"  {result.label}"
    )
    print(
        "========================================"
    )

    print_metric(
        "WER",
        result.wer,
    )

    print_metric(
        "CER (no spaces)",
        result.cer_no_spaces,
    )

    print_metric(
        "CER (with spaces)",
        result.cer_with_spaces,
    )


def print_improvement(
    before: EvaluationResult,
    after: EvaluationResult,
) -> None:
    def delta(
        before_rate: float,
        after_rate: float,
    ) -> float:
        return (
            before_rate
            -
            after_rate
        )

    print()
    print(
        "========================================"
    )
    print(
        "  PIPELINE IMPROVEMENT"
    )
    print(
        "========================================"
    )

    print(
        "WER improvement         : "
        f"{delta(before.wer.error_rate, after.wer.error_rate) * 100:+.2f} %p"
    )

    print(
        "CER(no spaces) improve  : "
        f"{delta(before.cer_no_spaces.error_rate, after.cer_no_spaces.error_rate) * 100:+.2f} %p"
    )

    print(
        "CER(with spaces) improve: "
        f"{delta(before.cer_with_spaces.error_rate, after.cer_with_spaces.error_rate) * 100:+.2f} %p"
    )


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Korean STT transcript accuracy using "
            "WER and CER against a ground-truth reference."
        )
    )

    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help=(
            "Ground-truth transcript (.txt, .srt, or .json)."
        ),
    )

    parser.add_argument(
        "--hypothesis",
        type=Path,
        help=(
            "Single hypothesis transcript. "
            "Use this when comparing only one output."
        ),
    )

    parser.add_argument(
        "--raw",
        type=Path,
        help=(
            "Raw STT transcript for before/after comparison."
        ),
    )

    parser.add_argument(
        "--corrected",
        type=Path,
        help=(
            "Corrected/final transcript for before/after comparison."
        ),
    )

    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help=(
            "Include punctuation in CER/WER normalization. "
            "Default removes punctuation."
        ),
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help=(
            "Optional path to write machine-readable evaluation JSON."
        ),
    )

    return parser.parse_args()


def validate_paths(
    args: argparse.Namespace,
) -> None:
    if not args.reference.exists():
        raise FileNotFoundError(
            f"Reference file not found: {args.reference}"
        )

    supplied = [
        args.hypothesis,
        args.raw,
        args.corrected,
    ]

    if not any(
        path is not None
        for path in supplied
    ):
        raise ValueError(
            "Provide --hypothesis, or --raw/--corrected."
        )

    for path in supplied:
        if path is None:
            continue

        if not path.exists():
            raise FileNotFoundError(
                f"Transcript file not found: {path}"
            )


def main() -> int:
    args = parse_args()

    try:
        validate_paths(args)

        reference_text = load_transcript(
            args.reference
        )

        remove_punctuation = (
            not args.keep_punctuation
        )

        results: list[EvaluationResult] = []

        if args.hypothesis is not None:
            hypothesis_text = load_transcript(
                args.hypothesis
            )

            results.append(
                evaluate(
                    label=(
                        f"HYPOTHESIS: {args.hypothesis}"
                    ),
                    reference_text=reference_text,
                    hypothesis_text=hypothesis_text,
                    remove_punctuation=remove_punctuation,
                )
            )

        raw_result: EvaluationResult | None = None
        corrected_result: EvaluationResult | None = None

        if args.raw is not None:
            raw_text = load_transcript(
                args.raw
            )

            raw_result = evaluate(
                label=(
                    f"RAW: {args.raw}"
                ),
                reference_text=reference_text,
                hypothesis_text=raw_text,
                remove_punctuation=remove_punctuation,
            )

            results.append(
                raw_result
            )

        if args.corrected is not None:
            corrected_text = load_transcript(
                args.corrected
            )

            corrected_result = evaluate(
                label=(
                    f"CORRECTED: {args.corrected}"
                ),
                reference_text=reference_text,
                hypothesis_text=corrected_text,
                remove_punctuation=remove_punctuation,
            )

            results.append(
                corrected_result
            )

        print()
        print(
            "STT ACCURACY EVALUATION"
        )
        print(
            f"Reference : {args.reference}"
        )
        print(
            "Punctuation: "
            +
            (
                "kept"
                if args.keep_punctuation
                else
                "removed"
            )
        )

        for result in results:
            print_result(
                result
            )

        if (
            raw_result is not None
            and
            corrected_result is not None
        ):
            print_improvement(
                raw_result,
                corrected_result,
            )

        if args.json_output is not None:
            payload = {
                "reference": str(
                    args.reference
                ),
                "remove_punctuation": (
                    remove_punctuation
                ),
                "results": [
                    result.as_dict()
                    for result in results
                ],
            }

            args.json_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            args.json_output.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )
                +
                "\n",
                encoding="utf-8",
            )

            print()
            print(
                f"JSON written: {args.json_output}"
            )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    sys.exit(
        main()
    )
