import copy
import os
import re
import subprocess


# =========================================================
# CONFIG
# =========================================================

RECOVERY_SOURCES = {
    "recovery",
    "recovery_long",
    "recovery_micro",
}

DEFAULT_BATCH_CONTEXT = 0.50

SECONDARY_MIN_DURATION = 3.0
SECONDARY_MIN_TEXT_CHARS = 8

SECONDARY_BOUNDARY_MIN_GAP = 0.18
SECONDARY_BOUNDARY_MAX_GAP = 1.50


# =========================================================
# CONSERVATIVE BATCH OPTIMIZATION
# =========================================================
#
# 현재 secondary_passes는 omission뿐 아니라 subtitle alignment도
# 함께 사용한다.
#
# 따라서 이번 단계에서는 secondary target 85개를 줄이지 않는다.
# target을 줄이면 short secondary match와 global word timeline이
# 같이 줄어들 수 있기 때문이다.
#
# 대신 인접 batch가 context 때문에 같은 오디오를 중복 처리하는
# 부분만 제거한다.
#
# 단, canonical segment 사이의 gap이 omission 의심 범위
# (0.18 ~ 1.50초)에 있으면 그 경계는 기존 overlap을 보존한다.
#
# 즉:
#   - target coverage 유지
#   - batch 개수 유지
#   - omission 위험 gap 보호
#   - 안전한 중복 오디오만 제거
#

ENABLE_SAFE_OVERLAP_OPTIMIZATION = True

PROTECT_OMISSION_GAP_MIN = SECONDARY_BOUNDARY_MIN_GAP
PROTECT_OMISSION_GAP_MAX = SECONDARY_BOUNDARY_MAX_GAP


# =========================================================
# TEXT / SEGMENT HELPERS
# =========================================================

def normalize_text(
    text: str,
) -> str:

    text = str(
        text or ""
    ).lower()

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    text = re.sub(
        r"[^\w가-힣]",
        "",
        text,
    )

    return text


def words_to_text(
    words: list[dict],
) -> str:

    return " ".join(
        str(
            word.get(
                "word",
                "",
            )
        ).strip()
        for word
        in words
        if str(
            word.get(
                "word",
                "",
            )
        ).strip()
    )


def segment_duration(
    segment: dict,
) -> float:

    return max(
        0.0,
        float(
            segment[
                "end"
            ]
        )
        -
        float(
            segment[
                "start"
            ]
        ),
    )


def segment_text_length(
    segment: dict,
) -> int:

    return len(
        normalize_text(
            segment.get(
                "text",
                "",
            )
        )
    )


# =========================================================
# SECONDARY PROBE TARGET
# =========================================================

def needs_secondary_probe(
    segments: list[dict],
    index: int,
    alignment_target_indices: set[int],
) -> tuple[
    bool,
    list[str],
]:

    segment = segments[
        index
    ]

    reasons = []

    # -----------------------------------------------------
    # SUBTITLE ALIGNMENT TARGET
    # -----------------------------------------------------

    if (
        index
        in
        alignment_target_indices
    ):

        reasons.append(
            "alignment_target"
        )

    # -----------------------------------------------------
    # RECOVERY SEGMENT
    # -----------------------------------------------------

    source = segment.get(
        "source",
        "primary",
    )

    if (
        source
        in
        RECOVERY_SOURCES
    ):

        reasons.append(
            "recovery_source"
        )

    # -----------------------------------------------------
    # LONG / TEXT-HEAVY SEGMENT
    # -----------------------------------------------------

    duration = segment_duration(
        segment
    )

    text_length = segment_text_length(
        segment
    )

    if (
        duration
        >=
        SECONDARY_MIN_DURATION
        and
        text_length
        >=
        SECONDARY_MIN_TEXT_CHARS
    ):

        reasons.append(
            "duration"
        )

    # -----------------------------------------------------
    # BEFORE GAP
    # -----------------------------------------------------

    if index > 0:

        previous = segments[
            index - 1
        ]

        gap_before = (
            float(
                segment[
                    "start"
                ]
            )
            -
            float(
                previous[
                    "end"
                ]
            )
        )

        if (
            SECONDARY_BOUNDARY_MIN_GAP
            <=
            gap_before
            <=
            SECONDARY_BOUNDARY_MAX_GAP
        ):

            reasons.append(
                "boundary_before"
            )

    # -----------------------------------------------------
    # AFTER GAP
    # -----------------------------------------------------

    if (
        index
        <
        len(
            segments
        )
        -
        1
    ):

        next_segment = segments[
            index + 1
        ]

        gap_after = (
            float(
                next_segment[
                    "start"
                ]
            )
            -
            float(
                segment[
                    "end"
                ]
            )
        )

        if (
            SECONDARY_BOUNDARY_MIN_GAP
            <=
            gap_after
            <=
            SECONDARY_BOUNDARY_MAX_GAP
        ):

            reasons.append(
                "boundary_after"
            )

    return (
        bool(
            reasons
        ),
        reasons,
    )


# =========================================================
# AUDIO EXTRACTION
# =========================================================

def extract_secondary_audio(
    input_file: str,
    output_file: str,
    start: float,
    end: float,
):

    duration = max(
        0.01,
        end
        -
        start,
    )

    os.makedirs(
        os.path.dirname(
            output_file
        ),
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        input_file,
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_file,
    ]

    subprocess.run(
        command,
        check=True,
    )


# =========================================================
# CONTEXT SEGMENT INDICES
# =========================================================

def collect_context_segment_indices(
    segments: list[dict],
    start: float,
    end: float,
) -> list[int]:

    context_indices = []

    for segment_index, segment in enumerate(
        segments
    ):

        segment_start = float(
            segment[
                "start"
            ]
        )

        segment_end = float(
            segment[
                "end"
            ]
        )

        if (
            segment_end
            <
            start
        ):

            continue

        if (
            segment_start
            >
            end
        ):

            continue

        context_indices.append(
            segment_index
        )

    return context_indices


# =========================================================
# BATCH BUILDER
# =========================================================

def build_secondary_batches(
    segments: list[dict],
    target_indices: list[int],
    media_duration: float,
    max_batch_duration: float,
    max_target_gap: float,
    context: float = DEFAULT_BATCH_CONTEXT,
) -> list[dict]:

    if not target_indices:
        return []

    target_indices = sorted(
        set(
            target_indices
        )
    )

    grouped = []

    current = [
        target_indices[
            0
        ]
    ]

    for index in (
        target_indices[
            1:
        ]
    ):

        previous_index = current[
            -1
        ]

        previous = segments[
            previous_index
        ]

        segment = segments[
            index
        ]

        target_gap = (
            float(
                segment[
                    "start"
                ]
            )
            -
            float(
                previous[
                    "end"
                ]
            )
        )

        candidate_start = (
            float(
                segments[
                    current[
                        0
                    ]
                ][
                    "start"
                ]
            )
            -
            context
        )

        candidate_end = (
            float(
                segment[
                    "end"
                ]
            )
            +
            context
        )

        candidate_duration = (
            candidate_end
            -
            candidate_start
        )

        if (
            target_gap
            <=
            max_target_gap
            and
            candidate_duration
            <=
            max_batch_duration
        ):

            current.append(
                index
            )

        else:

            grouped.append(
                current
            )

            current = [
                index
            ]

    if current:

        grouped.append(
            current
        )

    results = []

    for indices in grouped:

        first = segments[
            indices[
                0
            ]
        ]

        last = segments[
            indices[
                -1
            ]
        ]

        start = max(
            0.0,
            float(
                first[
                    "start"
                ]
            )
            -
            context,
        )

        end = min(
            media_duration,
            float(
                last[
                    "end"
                ]
            )
            +
            context,
        )

        context_indices = (
            collect_context_segment_indices(
                segments=(
                    segments
                ),
                start=(
                    start
                ),
                end=(
                    end
                ),
            )
        )

        results.append(
            {
                "start": (
                    start
                ),

                "end": (
                    end
                ),

                "secondary_target_indices": (
                    copy.deepcopy(
                        indices
                    )
                ),

                "context_segment_indices": (
                    context_indices
                ),
            }
        )

    return results


# =========================================================
# SAFE OVERLAP OPTIMIZER
# =========================================================

def optimize_secondary_batch_overlaps(
    segments: list[dict],
    batches: list[dict],
) -> tuple[
    list[dict],
    dict,
]:
    """
    Remove only redundant overlap between adjacent secondary batches.

    Safety rules:
    1. target indices are never changed.
    2. each target segment remains fully inside its batch.
    3. omission-risk canonical gaps (0.18~1.50 sec) keep original overlap.
    4. overlapping canonical target segments keep original overlap.
    5. context_segment_indices are recomputed after trimming.

    This is deliberately conservative.  It does NOT prune secondary targets.
    """

    optimized = copy.deepcopy(
        batches
    )

    raw_audio = sum(
        max(
            0.0,
            float(
                batch[
                    "end"
                ]
            )
            -
            float(
                batch[
                    "start"
                ]
            ),
        )
        for batch
        in optimized
    )

    trimmed_boundaries = 0
    protected_boundaries = 0
    overlap_seconds_removed = 0.0

    if (
        not ENABLE_SAFE_OVERLAP_OPTIMIZATION
        or
        len(
            optimized
        )
        <
        2
    ):

        return (
            optimized,
            {
                "raw_audio": (
                    raw_audio
                ),
                "optimized_audio": (
                    raw_audio
                ),
                "saved_audio": (
                    0.0
                ),
                "trimmed_boundaries": (
                    0
                ),
                "protected_boundaries": (
                    0
                ),
            },
        )

    for batch_index in range(
        len(
            optimized
        )
        -
        1
    ):

        left = optimized[
            batch_index
        ]

        right = optimized[
            batch_index
            +
            1
        ]

        left_end = float(
            left[
                "end"
            ]
        )

        right_start = float(
            right[
                "start"
            ]
        )

        current_overlap = max(
            0.0,
            left_end
            -
            right_start,
        )

        # No duplicated audio at this boundary.
        if current_overlap <= 0.0:
            continue

        left_targets = left.get(
            "secondary_target_indices",
            [],
        )

        right_targets = right.get(
            "secondary_target_indices",
            [],
        )

        if (
            not left_targets
            or
            not right_targets
        ):

            protected_boundaries += 1
            continue

        left_last_index = left_targets[
            -1
        ]

        right_first_index = right_targets[
            0
        ]

        left_target_end = float(
            segments[
                left_last_index
            ][
                "end"
            ]
        )

        right_target_start = float(
            segments[
                right_first_index
            ][
                "start"
            ]
        )

        canonical_gap = (
            right_target_start
            -
            left_target_end
        )

        # -------------------------------------------------
        # PROTECTION 1:
        # target segments themselves overlap
        # -------------------------------------------------

        if canonical_gap < 0.0:

            protected_boundaries += 1
            continue

        # -------------------------------------------------
        # PROTECTION 2:
        # omission-sensitive canonical gap
        # -------------------------------------------------

        if (
            PROTECT_OMISSION_GAP_MIN
            <=
            canonical_gap
            <=
            PROTECT_OMISSION_GAP_MAX
        ):

            protected_boundaries += 1
            continue

        # -------------------------------------------------
        # SAFE CUT
        #
        # Place one shared boundary between:
        #   left target end
        #   right target start
        #
        # Then clamp it to the existing overlap region.
        # -------------------------------------------------

        desired_cut = (
            left_target_end
            +
            right_target_start
        ) / 2.0

        overlap_low = (
            right_start
        )

        overlap_high = (
            left_end
        )

        cut = min(
            overlap_high,
            max(
                overlap_low,
                desired_cut,
            ),
        )

        # target coverage must remain intact
        if cut < left_target_end:
            protected_boundaries += 1
            continue

        if cut > right_target_start:
            protected_boundaries += 1
            continue

        old_left_end = float(
            left[
                "end"
            ]
        )

        old_right_start = float(
            right[
                "start"
            ]
        )

        left[
            "end"
        ] = cut

        right[
            "start"
        ] = cut

        removed = (
            old_left_end
            -
            old_right_start
        )

        if removed > 0.0:

            overlap_seconds_removed += (
                removed
            )

            trimmed_boundaries += 1

    # =====================================================
    # RECOMPUTE CONTEXT INDICES
    # =====================================================

    for batch in optimized:

        batch[
            "context_segment_indices"
        ] = (
            collect_context_segment_indices(
                segments=(
                    segments
                ),
                start=(
                    float(
                        batch[
                            "start"
                        ]
                    )
                ),
                end=(
                    float(
                        batch[
                            "end"
                        ]
                    )
                ),
            )
        )

    optimized_audio = sum(
        max(
            0.0,
            float(
                batch[
                    "end"
                ]
            )
            -
            float(
                batch[
                    "start"
                ]
            ),
        )
        for batch
        in optimized
    )

    saved_audio = max(
        0.0,
        raw_audio
        -
        optimized_audio,
    )

    return (
        optimized,
        {
            "raw_audio": (
                raw_audio
            ),

            "optimized_audio": (
                optimized_audio
            ),

            "saved_audio": (
                saved_audio
            ),

            "trimmed_boundaries": (
                trimmed_boundaries
            ),

            "protected_boundaries": (
                protected_boundaries
            ),

            "overlap_seconds_removed": (
                overlap_seconds_removed
            ),
        },
    )


# =========================================================
# MAIN
# =========================================================

def build_secondary_passes(
    segments: list[dict],
    alignment_target_indices: list[int],
    normalized_audio: str,
    whisper,
    work_dir: str,
    media_duration: float,
    max_batch_duration: float,
    max_target_gap: float,
) -> list[dict]:
    """
    Create shared secondary ASR evidence for:

    1. subtitle word alignment
    2. omission detection

    Optimization policy:
    - keep the exact same target-selection logic
    - keep the same number/order of batches
    - remove only safe duplicated overlap between adjacent batches
    - preserve omission-sensitive boundaries

    The purpose of this stage is regression-safe optimization.
    Aggressive target pruning is intentionally postponed.
    """

    os.makedirs(
        work_dir,
        exist_ok=True,
    )

    alignment_target_set = set(
        alignment_target_indices
    )

    secondary_target_indices = []
    secondary_reasons = {}

    # =====================================================
    # TARGET SELECTION
    # =====================================================

    for index in range(
        len(
            segments
        )
    ):

        (
            needed,
            reasons,
        ) = (
            needs_secondary_probe(
                segments=(
                    segments
                ),
                index=(
                    index
                ),
                alignment_target_indices=(
                    alignment_target_set
                ),
            )
        )

        if not needed:
            continue

        secondary_target_indices.append(
            index
        )

        secondary_reasons[
            index
        ] = reasons

    # =====================================================
    # ORIGINAL BATCH BUILD
    # =====================================================

    raw_batches = (
        build_secondary_batches(
            segments=(
                segments
            ),
            target_indices=(
                secondary_target_indices
            ),
            media_duration=(
                media_duration
            ),
            max_batch_duration=(
                max_batch_duration
            ),
            max_target_gap=(
                max_target_gap
            ),
        )
    )

    # =====================================================
    # SAFE OVERLAP OPTIMIZATION
    # =====================================================

    (
        batches,
        optimization_stats,
    ) = (
        optimize_secondary_batch_overlaps(
            segments=(
                segments
            ),
            batches=(
                raw_batches
            ),
        )
    )

    total_batch_audio = sum(
        batch[
            "end"
        ]
        -
        batch[
            "start"
        ]
        for batch
        in batches
    )

    # =====================================================
    # LOG
    # =====================================================

    print()

    print(
        "Starting secondary ASR pass generation..."
    )

    print(
        f"  total segments       : "
        f"{len(segments)}"
    )

    print(
        f"  alignment targets    : "
        f"{len(alignment_target_indices)}"
    )

    print(
        f"  skipped short segs   : "
        f"{len(segments) - len(alignment_target_indices)}"
    )

    print(
        f"  secondary targets    : "
        f"{len(secondary_target_indices)}"
    )

    print(
        f"  secondary extra      : "
        f"{len(secondary_target_indices) - len(alignment_target_indices)}"
    )

    print(
        f"  secondary batches    : "
        f"{len(batches)}"
    )

    print(
        f"  raw batch audio      : "
        f"{optimization_stats['raw_audio']:.2f} sec"
    )

    print(
        f"  optimized audio      : "
        f"{optimization_stats['optimized_audio']:.2f} sec"
    )

    print(
        f"  audio saved          : "
        f"{optimization_stats['saved_audio']:.2f} sec"
    )

    print(
        f"  trimmed boundaries   : "
        f"{optimization_stats['trimmed_boundaries']}"
    )

    print(
        f"  protected boundaries : "
        f"{optimization_stats['protected_boundaries']}"
    )

    # =====================================================
    # SECONDARY ASR
    # =====================================================

    secondary_passes = []

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        start = float(
            batch[
                "start"
            ]
        )

        end = float(
            batch[
                "end"
            ]
        )

        clip_duration = (
            end
            -
            start
        )

        output_file = os.path.join(
            work_dir,
            f"batch_{batch_number:04d}.wav",
        )

        extract_secondary_audio(
            input_file=(
                normalized_audio
            ),
            output_file=(
                output_file
            ),
            start=(
                start
            ),
            end=(
                end
            ),
        )

        batch_alignment_targets = [
            index
            for index
            in batch[
                "secondary_target_indices"
            ]
            if (
                index
                in
                alignment_target_set
            )
        ]

        print()

        print(
            f"  [SECONDARY BATCH {batch_number}] "
            f"{start:.2f} ~ {end:.2f} "
            f"({clip_duration:.2f}s, "
            f"{len(batch_alignment_targets)} align / "
            f"{len(batch['secondary_target_indices'])} secondary)"
        )

        words = (
            whisper.transcribe_words(
                audio_file=(
                    output_file
                ),
                global_offset=(
                    start
                ),
                audio_duration=(
                    clip_duration
                ),
            )
        )

        print(
            f"    words detected: "
            f"{len(words)}"
        )

        target_reason_map = {}

        for target_index in (
            batch[
                "secondary_target_indices"
            ]
        ):

            target_reason_map[
                str(
                    target_index
                )
            ] = (
                secondary_reasons.get(
                    target_index,
                    [],
                )
            )

        secondary_passes.append(
            {
                "batch_number": (
                    batch_number
                ),

                "start": (
                    start
                ),

                "end": (
                    end
                ),

                "secondary_target_indices": (
                    copy.deepcopy(
                        batch[
                            "secondary_target_indices"
                        ]
                    )
                ),

                "alignment_target_indices": (
                    copy.deepcopy(
                        batch_alignment_targets
                    )
                ),

                "target_reasons": (
                    target_reason_map
                ),

                "context_segment_indices": (
                    copy.deepcopy(
                        batch[
                            "context_segment_indices"
                        ]
                    )
                ),

                "words": (
                    copy.deepcopy(
                        words
                    )
                ),

                "text": (
                    words_to_text(
                        words
                    )
                ),
            }
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()

    print(
        "Secondary ASR generation:"
    )

    print(
        f"  alignment targets    : "
        f"{len(alignment_target_indices)}"
    )

    print(
        f"  secondary targets    : "
        f"{len(secondary_target_indices)}"
    )

    print(
        f"  batch passes         : "
        f"{len(batches)}"
    )

    print(
        f"  raw batch audio      : "
        f"{optimization_stats['raw_audio']:.2f} sec"
    )

    print(
        f"  batch audio total    : "
        f"{total_batch_audio:.2f} sec"
    )

    print(
        f"  audio saved          : "
        f"{optimization_stats['saved_audio']:.2f} sec"
    )

    print(
        f"  secondary passes     : "
        f"{len(secondary_passes)}"
    )

    return secondary_passes
