from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
BASELINE_PATH = Path(__file__).with_name("regression_baseline.json")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return default


def read_text(path: Path) -> str:
    if not path.exists():
        return ""

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except OSError:
        return ""


def flatten_strings(
    value: Any,
) -> list[str]:

    result: list[str] = []

    if isinstance(
        value,
        str,
    ):

        result.append(
            value
        )

    elif isinstance(
        value,
        dict,
    ):

        for item in value.values():

            result.extend(
                flatten_strings(
                    item
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for item in value:

            result.extend(
                flatten_strings(
                    item
                )
            )

    return result


def combined_output_text() -> str:

    chunks: list[str] = []

    text_files = [
        OUTPUT_DIR / "result_corrected.srt",
        OUTPUT_DIR / "result_raw.srt",
        OUTPUT_DIR / "run.log",
    ]

    for path in text_files:

        text = (
            read_text(
                path
            )
        )

        if text:

            chunks.append(
                text
            )

    json_files = [
        OUTPUT_DIR / "corrected_transcript.json",
        OUTPUT_DIR / "raw_transcript.json",
        OUTPUT_DIR / "correction_log.json",
        OUTPUT_DIR / "review_candidates.json",
        OUTPUT_DIR / "omission_candidates.json",
        OUTPUT_DIR / "omission_log.json",
        OUTPUT_DIR / "omission_review.json",
        OUTPUT_DIR / "omission_realign_log.json",
        OUTPUT_DIR / "secondary_passes.json",
        OUTPUT_DIR / "omission_probe_passes.json",
    ]

    for path in json_files:

        data = (
            load_json(
                path
            )
        )

        if data is None:
            continue

        chunks.extend(
            flatten_strings(
                data
            )
        )

    return "\n".join(
        chunks
    )


def normalize_whitespace(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            text or ""
        ),
    ).strip()


def srt_plain_text(
    srt_text: str,
) -> str:
    """
    Convert an SRT file into one continuous searchable text stream.

    This intentionally ignores:
    - cue numbers
    - timestamp lines
    - blank lines

    Therefore a phrase split across two adjacent subtitle cues can still
    be regression-checked as a normal phrase.
    """

    content_lines: list[str] = []

    timestamp_pattern = re.compile(
        r"^\s*\d{2}:\d{2}:\d{2},\d{3}"
        r"\s*-->\s*"
        r"\d{2}:\d{2}:\d{2},\d{3}\s*$"
    )

    for raw_line in str(
        srt_text or ""
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.isdigit():
            continue

        if timestamp_pattern.match(
            line
        ):
            continue

        content_lines.append(
            line
        )

    return normalize_whitespace(
        " ".join(
            content_lines
        )
    )


def contains_phrase(
    haystack: str,
    needle: str,
) -> bool:
    """
    Whitespace-tolerant phrase comparison.

    We normalize runs of spaces/newlines/tabs to a single space so a
    phrase split by cue boundaries or formatting whitespace does not
    falsely fail.
    """

    return (
        normalize_whitespace(
            needle
        )
        in
        normalize_whitespace(
            haystack
        )
    )


def parse_number(
    text: str,
    label: str,
    *,
    integer: bool = False,
) -> float | int | None:

    pattern = (
        rf"(?mi)^\s*"
        rf"{re.escape(label)}"
        rf"\s*:\s*"
        rf"([0-9]+(?:\.[0-9]+)?)"
    )

    match = re.search(
        pattern,
        text,
    )

    if not match:
        return None

    value = float(
        match.group(
            1
        )
    )

    if integer:

        return int(
            value
        )

    return value


def parse_percent(
    text: str,
    label: str,
) -> float | None:

    pattern = (
        rf"(?mi)^\s*"
        rf"{re.escape(label)}"
        rf"\s*:\s*"
        rf"([0-9]+(?:\.[0-9]+)?)%"
    )

    match = re.search(
        pattern,
        text,
    )

    if not match:
        return None

    return float(
        match.group(
            1
        )
    )


def extract_section(
    text: str,
    start_heading: str,
    end_heading: str | None = None,
) -> str:

    start_index = text.find(
        start_heading
    )

    if start_index < 0:
        return ""

    section = text[
        start_index:
    ]

    if end_heading:

        relative_end = section.find(
            end_heading,
            len(
                start_heading
            ),
        )

        if relative_end >= 0:

            section = section[
                :relative_end
            ]

    return section


def parse_realign_similarity(
    text: str,
) -> float | None:

    section = extract_section(
        text=text,
        start_heading=(
            "========== OMISSION REALIGN =========="
        ),
        end_heading=(
            "========== CORRECTIONS =========="
        ),
    )

    if not section:
        return None

    matches = re.findall(
        r"(?m)^\s*SIM\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*$",
        section,
    )

    if not matches:
        return None

    return max(
        float(
            value
        )
        for value in matches
    )


def parse_verification(
    text: str,
) -> tuple[int, int] | None:

    matches = re.findall(
        r"(?mi)"
        r"(?:verification|VERIFY)"
        r"\s*:\s*"
        r"(\d+)\s*/\s*(\d+)",
        text,
    )

    if not matches:
        return None

    parsed = [
        (
            int(
                left
            ),
            int(
                right
            ),
        )
        for left, right in matches
    ]

    return max(
        parsed,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )


def parse_alignment_unresolved(
    text: str,
) -> int | None:

    matches = re.findall(
        r"(?mi)^\s*"
        r"unresolved"
        r"\s*:\s*"
        r"(\d+)\s*$",
        text,
    )

    if not matches:
        return None

    return int(
        matches[
            -1
        ]
    )


def parse_realign_words(
    text: str,
) -> int | None:
    """
    Read WORDS only from the printed OMISSION REALIGN report.

    Previous implementation searched the whole run.log with
    case-insensitive /^WORDS/, which accidentally matched the
    alignment-secondary line 'words : 696'.
    """

    section = extract_section(
        text=text,
        start_heading=(
            "========== OMISSION REALIGN =========="
        ),
        end_heading=(
            "========== CORRECTIONS =========="
        ),
    )

    if not section:
        return None

    matches = re.findall(
        r"(?m)^\s*WORDS\s*:\s*(\d+)\s*$",
        section,
    )

    if not matches:
        return None

    return max(
        int(
            value
        )
        for value in matches
    )


class CheckResult:

    def __init__(
        self,
        name: str,
        passed: bool,
        detail: str,
        required: bool = True,
    ) -> None:

        self.name = (
            name
        )

        self.passed = (
            passed
        )

        self.detail = (
            detail
        )

        self.required = (
            required
        )


def main() -> int:

    baseline = (
        load_json(
            BASELINE_PATH
        )
    )

    if not isinstance(
        baseline,
        dict,
    ):

        print(
            f"[ERROR] baseline not found or invalid: "
            f"{BASELINE_PATH}"
        )

        return 2

    run_log_path = (
        OUTPUT_DIR
        /
        "run.log"
    )

    corrected_srt_path = (
        OUTPUT_DIR
        /
        "result_corrected.srt"
    )

    missing = [
        path
        for path in (
            run_log_path,
            corrected_srt_path,
        )
        if not path.exists()
    ]

    if missing:

        print(
            "[ERROR] required output files are missing:"
        )

        for path in missing:

            print(
                "  - "
                +
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                )
            )

        print(
            "\nRun the STT pipeline first:"
        )

        print(
            "  python -m app.main 2>&1 | tee output/run.log"
        )

        return 2

    run_log = (
        read_text(
            run_log_path
        )
    )

    all_text = (
        combined_output_text()
    )

    srt_text = (
        read_text(
            corrected_srt_path
        )
    )

    searchable_srt_text = (
        srt_plain_text(
            srt_text
        )
    )

    checks: list[
        CheckResult
    ] = []

    def add(
        name: str,
        passed: bool,
        detail: str,
        required: bool = True,
    ) -> None:

        checks.append(
            CheckResult(
                name=(
                    name
                ),
                passed=(
                    passed
                ),
                detail=(
                    detail
                ),
                required=(
                    required
                ),
            )
        )

    # =====================================================
    # CORE NUMERIC INVARIANTS
    # =====================================================

    expected_raw_segments = (
        baseline.get(
            "raw_segments"
        )
    )

    actual_raw_segments = (
        parse_number(
            run_log,
            "Raw segments",
            integer=True,
        )
    )

    if (
        expected_raw_segments
        is not None
    ):

        add(
            "raw segments",
            (
                actual_raw_segments
                ==
                int(
                    expected_raw_segments
                )
            ),
            (
                f"{actual_raw_segments} "
                f"(expected {expected_raw_segments})"
            ),
        )

    expected_alignment_passes = (
        baseline.get(
            "alignment_passes"
        )
    )

    actual_alignment_passes = (
        parse_number(
            run_log,
            "Alignment passes",
            integer=True,
        )
    )

    if (
        expected_alignment_passes
        is not None
    ):

        add(
            "alignment passes",
            (
                actual_alignment_passes
                ==
                int(
                    expected_alignment_passes
                )
            ),
            (
                f"{actual_alignment_passes} "
                f"(expected {expected_alignment_passes})"
            ),
        )

    expected_omission_probes = (
        baseline.get(
            "omission_probes"
        )
    )

    actual_omission_probes = (
        parse_number(
            run_log,
            "Omission probes",
            integer=True,
        )
    )

    if (
        expected_omission_probes
        is not None
    ):

        add(
            "omission probes",
            (
                actual_omission_probes
                ==
                int(
                    expected_omission_probes
                )
            ),
            (
                f"{actual_omission_probes} "
                f"(expected {expected_omission_probes})"
            ),
        )

    max_unresolved = int(
        baseline.get(
            "max_alignment_unresolved",
            0,
        )
    )

    actual_unresolved = (
        parse_alignment_unresolved(
            run_log
        )
    )

    add(
        "alignment unresolved",
        (
            actual_unresolved
            is not None
            and
            actual_unresolved
            <=
            max_unresolved
        ),
        (
            f"{actual_unresolved} "
            f"(max {max_unresolved})"
        ),
    )

    min_token_match = float(
        baseline.get(
            "min_token_match_ratio_percent",
            90.0,
        )
    )

    actual_token_match = (
        parse_percent(
            run_log,
            "token match ratio",
        )
    )

    add(
        "token match ratio",
        (
            actual_token_match
            is not None
            and
            actual_token_match
            >=
            min_token_match
        ),
        (
            (
                f"{actual_token_match:.1f}% "
                f"(min {min_token_match:.1f}%)"
            )
            if (
                actual_token_match
                is not None
            )
            else
            "not found"
        ),
    )

    cue_min = int(
        baseline.get(
            "cue_count_min",
            120,
        )
    )

    cue_max = int(
        baseline.get(
            "cue_count_max",
            126,
        )
    )

    actual_cues = (
        parse_number(
            run_log,
            "Corrected cues",
            integer=True,
        )
    )

    add(
        "corrected cue count",
        (
            actual_cues
            is not None
            and
            cue_min
            <=
            actual_cues
            <=
            cue_max
        ),
        (
            f"{actual_cues} "
            f"(allowed {cue_min}..{cue_max})"
        ),
    )

    min_omissions = int(
        baseline.get(
            "min_omissions_applied",
            1,
        )
    )

    actual_omissions = (
        parse_number(
            run_log,
            "Omissions applied",
            integer=True,
        )
    )

    add(
        "omissions applied",
        (
            actual_omissions
            is not None
            and
            actual_omissions
            >=
            min_omissions
        ),
        (
            f"{actual_omissions} "
            f"(min {min_omissions})"
        ),
    )

    min_corrections = int(
        baseline.get(
            "min_corrections",
            9,
        )
    )

    actual_corrections = (
        parse_number(
            run_log,
            "Corrections",
            integer=True,
        )
    )

    add(
        "corrections",
        (
            actual_corrections
            is not None
            and
            actual_corrections
            >=
            min_corrections
        ),
        (
            f"{actual_corrections} "
            f"(min {min_corrections})"
        ),
    )

    max_reviews = int(
        baseline.get(
            "max_review_candidates",
            0,
        )
    )

    actual_reviews = (
        parse_number(
            run_log,
            "Review candidates",
            integer=True,
        )
    )

    add(
        "review candidates",
        (
            actual_reviews
            is not None
            and
            actual_reviews
            <=
            max_reviews
        ),
        (
            f"{actual_reviews} "
            f"(max {max_reviews})"
        ),
    )

    # =====================================================
    # OMISSION QUALITY INVARIANTS
    # =====================================================

    expected_verification = (
        baseline.get(
            "omission_verification",
            [
                2,
                2,
            ],
        )
    )

    actual_verification = (
        parse_verification(
            run_log
        )
    )

    if (
        isinstance(
            expected_verification,
            list,
        )
        and
        len(
            expected_verification
        )
        ==
        2
    ):

        expected_pair = (
            int(
                expected_verification[
                    0
                ]
            ),
            int(
                expected_verification[
                    1
                ]
            ),
        )

        add(
            "omission verification",
            (
                actual_verification
                ==
                expected_pair
            ),
            (
                f"{actual_verification} "
                f"(expected {expected_pair})"
            ),
        )

    min_realign_similarity = float(
        baseline.get(
            "min_omission_realign_similarity",
            0.99,
        )
    )

    actual_similarity = (
        parse_realign_similarity(
            run_log
        )
    )

    add(
        "omission realign similarity",
        (
            actual_similarity
            is not None
            and
            actual_similarity
            >=
            min_realign_similarity
        ),
        (
            (
                f"{actual_similarity:.3f} "
                f"(min {min_realign_similarity:.3f})"
            )
            if (
                actual_similarity
                is not None
            )
            else
            "not found"
        ),
    )

    min_realign_words = int(
        baseline.get(
            "min_omission_realign_words",
            12,
        )
    )

    actual_realign_words = (
        parse_realign_words(
            run_log
        )
    )

    add(
        "omission realign words",
        (
            actual_realign_words
            is not None
            and
            actual_realign_words
            >=
            min_realign_words
        ),
        (
            f"{actual_realign_words} "
            f"(min {min_realign_words})"
        ),
    )

    # =====================================================
    # REQUIRED / FORBIDDEN PHRASES
    # =====================================================

    required_phrases = (
        baseline.get(
            "required_phrases",
            [],
        )
    )

    for phrase in (
        required_phrases
    ):

        phrase = str(
            phrase
        )

        found = (
            contains_phrase(
                all_text,
                phrase,
            )
        )

        add(
            f'phrase "{phrase}"',
            found,
            (
                "found"
                if found
                else
                "missing"
            ),
        )

    required_srt_phrases = (
        baseline.get(
            "required_srt_phrases",
            [],
        )
    )

    for phrase in (
        required_srt_phrases
    ):

        phrase = str(
            phrase
        )

        found = (
            contains_phrase(
                searchable_srt_text,
                phrase,
            )
        )

        add(
            f'SRT phrase "{phrase}"',
            found,
            (
                "found"
                if found
                else
                "missing"
            ),
        )

    forbidden_srt_phrases = (
        baseline.get(
            "forbidden_srt_phrases",
            [],
        )
    )

    for phrase in (
        forbidden_srt_phrases
    ):

        phrase = str(
            phrase
        )

        found = (
            contains_phrase(
                searchable_srt_text,
                phrase,
            )
        )

        add(
            f'forbidden SRT "{phrase}"',
            not found,
            (
                "absent"
                if not found
                else
                "FOUND"
            ),
        )

    # =====================================================
    # PERFORMANCE
    # =====================================================

    max_total_seconds = (
        baseline.get(
            "max_total_processing_seconds"
        )
    )

    if (
        max_total_seconds
        is not None
    ):

        actual_total = (
            parse_number(
                run_log,
                "Total processing",
            )
        )

        strict_performance = bool(
            baseline.get(
                "strict_performance",
                False,
            )
        )

        add(
            "total processing time",
            (
                actual_total
                is not None
                and
                actual_total
                <=
                float(
                    max_total_seconds
                )
            ),
            (
                (
                    f"{actual_total:.2f}s "
                    f"(target <= "
                    f"{float(max_total_seconds):.2f}s)"
                )
                if (
                    actual_total
                    is not None
                )
                else
                "not found"
            ),
            required=(
                strict_performance
            ),
        )

    # =====================================================
    # REPORT
    # =====================================================

    print()

    print(
        "========================================"
    )

    print(
        "       STT REGRESSION CHECK"
    )

    print(
        "========================================"
    )

    for check in (
        checks
    ):

        if check.passed:

            status = (
                "PASS"
            )

        elif check.required:

            status = (
                "FAIL"
            )

        else:

            status = (
                "WARN"
            )

        print(
            f"[{status}] "
            f"{check.name}: "
            f"{check.detail}"
        )

    required_failures = [
        check
        for check in checks
        if (
            check.required
            and
            not check.passed
        )
    ]

    warnings = [
        check
        for check in checks
        if (
            not check.required
            and
            not check.passed
        )
    ]

    print(
        "----------------------------------------"
    )

    if required_failures:

        print(
            "REGRESSION CHECK: FAIL "
            f"({len(required_failures)} "
            "required failure(s))"
        )

        return 1

    if warnings:

        print(
            "REGRESSION CHECK: PASS "
            f"({len(warnings)} warning(s))"
        )

        return 0

    print(
        "REGRESSION CHECK: PASS"
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
