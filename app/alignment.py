import copy
import os
import re
import subprocess

from difflib import SequenceMatcher


# =========================================================
# CONFIG
# =========================================================

RECOVERY_SOURCES = {
    "recovery",
    "recovery_long",
    "recovery_micro",
}

DEFAULT_BATCH_CONTEXT = 0.50
DEFAULT_LOCAL_CONTEXT = 0.25

MIN_ALIGNMENT_SIMILARITY = 0.45


# =========================================================
# SECONDARY COVERAGE
# =========================================================

SECONDARY_MIN_DURATION = 3.0
SECONDARY_MIN_TEXT_CHARS = 8

SECONDARY_BOUNDARY_MIN_GAP = 0.18
SECONDARY_BOUNDARY_MAX_GAP = 1.50


# =========================================================
# SECONDARY -> SHORT SEGMENT WORD MATCH
# =========================================================

#
# 기존 short segment timestamp가 약간 틀려 있어도
# 실제 secondary word를 찾을 수 있도록 주변을 넉넉하게 검색
#

SECONDARY_WORD_SEARCH_MARGIN = 2.0

#
# canonical segment text와 secondary word span이
# 이 정도 이상 일치해야 실제 subtitle timing에 사용
#

MIN_SECONDARY_WORD_MATCH_SIMILARITY = 0.62

#
# 너무 짧은 1글자/2글자 segment는 오매칭 가능성이 큼
#

MIN_SECONDARY_MATCH_TEXT_CHARS = 4


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(text: str) -> str:

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
        for word in words
        if str(
            word.get(
                "word",
                "",
            )
        ).strip()
    )


def calculate_text_similarity(
    left: str,
    right: str,
) -> float:

    left_normalized = (
        normalize_text(
            left
        )
    )

    right_normalized = (
        normalize_text(
            right
        )
    )

    if not left_normalized:
        return 0.0

    if not right_normalized:
        return 0.0

    return SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()


# =========================================================
# SEGMENT HELPERS
# =========================================================

def segment_duration(
    segment: dict,
) -> float:

    return max(
        0.0,
        float(
            segment["end"]
        )
        -
        float(
            segment["start"]
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
# NORMAL SUBTITLE ALIGNMENT TARGET
# =========================================================

def needs_word_alignment(
    segment: dict,
    max_cue_duration: float,
    max_chars: int,
) -> bool:

    duration = (
        segment_duration(
            segment
        )
    )

    text = str(
        segment.get(
            "text",
            "",
        )
    ).strip()

    if duration > max_cue_duration:
        return True

    if len(text) > max_chars:
        return True

    return False


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

    segment = (
        segments[index]
    )

    reasons = []

    if index in alignment_target_indices:

        reasons.append(
            "alignment_target"
        )

    source = (
        segment.get(
            "source",
            "primary",
        )
    )

    if source in RECOVERY_SOURCES:

        reasons.append(
            "recovery_source"
        )

    duration = (
        segment_duration(
            segment
        )
    )

    text_length = (
        segment_text_length(
            segment
        )
    )

    if (
        duration
        >= SECONDARY_MIN_DURATION
        and
        text_length
        >= SECONDARY_MIN_TEXT_CHARS
    ):

        reasons.append(
            "duration"
        )

    # -----------------------------------------------------
    # BEFORE GAP
    # -----------------------------------------------------

    if index > 0:

        previous = (
            segments[
                index - 1
            ]
        )

        gap_before = (
            float(
                segment["start"]
            )
            -
            float(
                previous["end"]
            )
        )

        if (
            SECONDARY_BOUNDARY_MIN_GAP
            <= gap_before
            <= SECONDARY_BOUNDARY_MAX_GAP
        ):

            reasons.append(
                "boundary_before"
            )

    # -----------------------------------------------------
    # AFTER GAP
    # -----------------------------------------------------

    if index < len(
        segments
    ) - 1:

        next_segment = (
            segments[
                index + 1
            ]
        )

        gap_after = (
            float(
                next_segment["start"]
            )
            -
            float(
                segment["end"]
            )
        )

        if (
            SECONDARY_BOUNDARY_MIN_GAP
            <= gap_after
            <= SECONDARY_BOUNDARY_MAX_GAP
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

def extract_alignment_audio(
    input_file: str,
    output_file: str,
    start: float,
    end: float,
):

    duration = max(
        0.01,
        end - start,
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
# BATCH BUILDER
# =========================================================

def build_alignment_batches(
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
        target_indices[0]
    ]

    for index in (
        target_indices[1:]
    ):

        previous_index = (
            current[-1]
        )

        previous = (
            segments[
                previous_index
            ]
        )

        segment = (
            segments[
                index
            ]
        )

        target_gap = (
            float(
                segment["start"]
            )
            -
            float(
                previous["end"]
            )
        )

        candidate_start = (
            float(
                segments[
                    current[0]
                ]["start"]
            )
            -
            context
        )

        candidate_end = (
            float(
                segment["end"]
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
            <= max_target_gap
            and
            candidate_duration
            <= max_batch_duration
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

        first = (
            segments[
                indices[0]
            ]
        )

        last = (
            segments[
                indices[-1]
            ]
        )

        start = max(
            0.0,
            float(
                first["start"]
            )
            -
            context,
        )

        end = min(
            media_duration,
            float(
                last["end"]
            )
            +
            context,
        )

        context_indices = []

        for segment_index, segment in enumerate(
            segments
        ):

            segment_start = float(
                segment["start"]
            )

            segment_end = float(
                segment["end"]
            )

            if segment_end < start:
                continue

            if segment_start > end:
                continue

            context_indices.append(
                segment_index
            )

        results.append(
            {
                "start": start,
                "end": end,
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
# EXISTING TEMPORAL WORD ASSIGNMENT
# =========================================================

def assign_words_to_segment(
    words: list[dict],
    segment: dict,
) -> list[dict]:

    segment_start = float(
        segment["start"]
    )

    segment_end = float(
        segment["end"]
    )

    assigned = []

    for word in words:

        word_start = float(
            word["start"]
        )

        word_end = float(
            word["end"]
        )

        midpoint = (
            word_start
            +
            word_end
        ) / 2.0

        if (
            segment_start
            <= midpoint
            <= segment_end
        ):

            assigned.append(
                copy.deepcopy(
                    word
                )
            )

            continue

        overlap_start = max(
            segment_start,
            word_start,
        )

        overlap_end = min(
            segment_end,
            word_end,
        )

        overlap = max(
            0.0,
            overlap_end
            -
            overlap_start,
        )

        word_duration = max(
            0.001,
            word_end
            -
            word_start,
        )

        if (
            overlap
            /
            word_duration
            >= 0.50
        ):

            assigned.append(
                copy.deepcopy(
                    word
                )
            )

    return assigned


# =========================================================
# TEXT-BASED SECONDARY WORD MATCH
# =========================================================

def find_best_secondary_word_span(
    segment: dict,
    words: list[dict],
) -> dict | None:

    canonical_text = str(
        segment.get(
            "text",
            "",
        )
    ).strip()

    canonical_normalized = (
        normalize_text(
            canonical_text
        )
    )

    if (
        len(
            canonical_normalized
        )
        <
        MIN_SECONDARY_MATCH_TEXT_CHARS
    ):

        return None

    segment_start = float(
        segment["start"]
    )

    segment_end = float(
        segment["end"]
    )

    # -----------------------------------------------------
    # 기존 segment timestamp 자체가 정확하지 않을 수 있으므로
    # 앞뒤 2초까지 secondary word 후보에 포함
    # -----------------------------------------------------

    nearby_words = []

    for word in words:

        word_start = float(
            word["start"]
        )

        word_end = float(
            word["end"]
        )

        if (
            word_end
            <
            segment_start
            -
            SECONDARY_WORD_SEARCH_MARGIN
        ):
            continue

        if (
            word_start
            >
            segment_end
            +
            SECONDARY_WORD_SEARCH_MARGIN
        ):
            continue

        nearby_words.append(
            copy.deepcopy(
                word
            )
        )

    if not nearby_words:

        return None

    canonical_word_count = max(
        1,
        len(
            canonical_text.split()
        ),
    )

    # canonical보다 약간 짧거나 긴 secondary span까지 허용
    min_span = max(
        1,
        canonical_word_count - 3,
    )

    max_span = min(
        len(
            nearby_words
        ),
        canonical_word_count + 5,
    )

    best = None

    for start_index in range(
        len(
            nearby_words
        )
    ):

        for span_length in range(
            min_span,
            max_span + 1,
        ):

            end_index = (
                start_index
                +
                span_length
            )

            if end_index > len(
                nearby_words
            ):
                break

            candidate_words = (
                nearby_words[
                    start_index:
                    end_index
                ]
            )

            candidate_text = (
                words_to_text(
                    candidate_words
                )
            )

            similarity = (
                calculate_text_similarity(
                    canonical_text,
                    candidate_text,
                )
            )

            if best is None:

                best = {
                    "words": (
                        candidate_words
                    ),
                    "text": (
                        candidate_text
                    ),
                    "similarity": (
                        similarity
                    ),
                }

                continue

            if (
                similarity
                >
                best[
                    "similarity"
                ]
            ):

                best = {
                    "words": (
                        candidate_words
                    ),
                    "text": (
                        candidate_text
                    ),
                    "similarity": (
                        similarity
                    ),
                }

    if best is None:
        return None

    if (
        best[
            "similarity"
        ]
        <
        MIN_SECONDARY_WORD_MATCH_SIMILARITY
    ):

        return None

    return best


# =========================================================
# LOCAL FALLBACK
# =========================================================

def local_align_segment(
    segment: dict,
    normalized_audio: str,
    whisper,
    alignment_dir: str,
    media_duration: float,
    local_index: int,
) -> list[dict]:

    start = max(
        0.0,
        float(
            segment["start"]
        )
        -
        DEFAULT_LOCAL_CONTEXT,
    )

    end = min(
        media_duration,
        float(
            segment["end"]
        )
        +
        DEFAULT_LOCAL_CONTEXT,
    )

    output_file = os.path.join(
        alignment_dir,
        (
            f"local_"
            f"{local_index:04d}.wav"
        ),
    )

    extract_alignment_audio(
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

    words = (
        whisper.transcribe_words(
            audio_file=(
                output_file
            ),
            global_offset=(
                start
            ),
            audio_duration=(
                end - start
            ),
        )
    )

    return assign_words_to_segment(
        words=(
            words
        ),
        segment=(
            segment
        ),
    )


# =========================================================
# MAIN
# =========================================================

def attach_word_timestamps_v3(
    segments: list[dict],
    normalized_audio: str,
    whisper,
    alignment_dir: str,
    media_duration: float,
    max_cue_duration: float,
    max_chars: int,
    max_batch_duration: float,
    max_target_gap: float,
) -> tuple[
    list[dict],
    list[dict],
]:

    os.makedirs(
        alignment_dir,
        exist_ok=True,
    )

    aligned_segments = (
        copy.deepcopy(
            segments
        )
    )

    for segment in (
        aligned_segments
    ):

        segment[
            "words"
        ] = []

    # =====================================================
    # ORIGINAL ALIGNMENT TARGETS
    # =====================================================

    alignment_target_indices = []

    for index, segment in enumerate(
        aligned_segments
    ):

        if needs_word_alignment(
            segment=(
                segment
            ),
            max_cue_duration=(
                max_cue_duration
            ),
            max_chars=(
                max_chars
            ),
        ):

            alignment_target_indices.append(
                index
            )

    alignment_target_set = set(
        alignment_target_indices
    )

    # =====================================================
    # SECONDARY TARGETS
    # =====================================================

    secondary_target_indices = []

    secondary_reasons = {}

    for index in range(
        len(
            aligned_segments
        )
    ):

        (
            needed,
            reasons,
        ) = (
            needs_secondary_probe(
                segments=(
                    aligned_segments
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
        ] = (
            reasons
        )

    batches = (
        build_alignment_batches(
            segments=(
                aligned_segments
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

    total_batch_audio = sum(
        batch[
            "end"
        ]
        -
        batch[
            "start"
        ]
        for batch in batches
    )

    print()
    print(
        "Starting selective batch word alignment..."
    )

    print(
        f"  total segments       : "
        f"{len(aligned_segments)}"
    )

    print(
        f"  alignment targets    : "
        f"{len(alignment_target_indices)}"
    )

    print(
        f"  skipped short segs   : "
        f"{len(aligned_segments) - len(alignment_target_indices)}"
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
        f"  alignment batches    : "
        f"{len(batches)}"
    )

    print(
        f"  batch audio total    : "
        f"{total_batch_audio:.2f} sec"
    )

    # =====================================================
    # STATS
    # =====================================================

    batch_success = 0
    local_fallbacks = 0
    local_success = 0
    unresolved = 0

    secondary_short_success = 0

    secondary_passes = []

    processed_alignment_targets = set()

    #
    # 여러 overlapping batch가 같은 short segment를
    # 볼 수 있으므로 가장 similarity가 좋은 결과만 저장
    #
    secondary_word_matches = {}

    # =====================================================
    # BATCH LOOP
    # =====================================================

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        start = (
            batch[
                "start"
            ]
        )

        end = (
            batch[
                "end"
            ]
        )

        output_file = os.path.join(
            alignment_dir,
            (
                f"batch_"
                f"{batch_number:04d}.wav"
            ),
        )

        extract_alignment_audio(
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

        clip_duration = (
            end
            -
            start
        )

        batch_alignment_targets = [
            index
            for index
            in batch[
                "secondary_target_indices"
            ]
            if index
            in alignment_target_set
        ]

        print()
        print(
            f"  [ALIGN BATCH {batch_number}] "
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

        # =================================================
        # SECONDARY PASS SAVE
        # =================================================

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

        # =================================================
        # NEW:
        # 모든 canonical context segment에 대해
        # secondary text/word matching을 시도
        # =================================================

        for segment_index in (
            batch[
                "context_segment_indices"
            ]
        ):

            #
            # 기존 long alignment target은 아래 기존 로직이
            # 담당하므로 short/non-target에 집중한다.
            #

            if (
                segment_index
                in
                alignment_target_set
            ):

                continue

            segment = (
                aligned_segments[
                    segment_index
                ]
            )

            match = (
                find_best_secondary_word_span(
                    segment=(
                        segment
                    ),
                    words=(
                        words
                    ),
                )
            )

            if match is None:
                continue

            current = (
                secondary_word_matches.get(
                    segment_index
                )
            )

            if (
                current is None
                or
                match[
                    "similarity"
                ]
                >
                current[
                    "similarity"
                ]
            ):

                secondary_word_matches[
                    segment_index
                ] = (
                    match
                )

        # =================================================
        # EXISTING LONG/TEXT-HEAVY ALIGNMENT
        # =================================================

        for target_index in (
            batch_alignment_targets
        ):

            if (
                target_index
                in
                processed_alignment_targets
            ):
                continue

            processed_alignment_targets.add(
                target_index
            )

            segment = (
                aligned_segments[
                    target_index
                ]
            )

            assigned_words = (
                assign_words_to_segment(
                    words=(
                        words
                    ),
                    segment=(
                        segment
                    ),
                )
            )

            detected_text = (
                words_to_text(
                    assigned_words
                )
            )

            similarity = (
                calculate_text_similarity(
                    segment.get(
                        "text",
                        "",
                    ),
                    detected_text,
                )
            )

            if (
                assigned_words
                and
                similarity
                >=
                MIN_ALIGNMENT_SIMILARITY
            ):

                segment[
                    "words"
                ] = (
                    assigned_words
                )

                segment[
                    "alignment_source"
                ] = (
                    "batch"
                )

                segment[
                    "alignment_similarity"
                ] = (
                    similarity
                )

                batch_success += 1

                continue

            source = (
                segment.get(
                    "source",
                    "primary",
                )
            )

            if source in RECOVERY_SOURCES:

                local_fallbacks += 1

                local_words = (
                    local_align_segment(
                        segment=(
                            segment
                        ),
                        normalized_audio=(
                            normalized_audio
                        ),
                        whisper=(
                            whisper
                        ),
                        alignment_dir=(
                            alignment_dir
                        ),
                        media_duration=(
                            media_duration
                        ),
                        local_index=(
                            target_index
                        ),
                    )
                )

                if local_words:

                    segment[
                        "words"
                    ] = (
                        local_words
                    )

                    segment[
                        "alignment_source"
                    ] = (
                        "local_fallback"
                    )

                    segment[
                        "alignment_similarity"
                    ] = (
                        calculate_text_similarity(
                            segment.get(
                                "text",
                                "",
                            ),
                            words_to_text(
                                local_words
                            ),
                        )
                    )

                    local_success += 1

                    continue

            if assigned_words:

                segment[
                    "words"
                ] = (
                    assigned_words
                )

                segment[
                    "alignment_source"
                ] = (
                    "batch_low_similarity"
                )

                segment[
                    "alignment_similarity"
                ] = (
                    similarity
                )

                batch_success += 1

                continue

            segment[
                "alignment_source"
            ] = (
                "unresolved"
            )

            segment[
                "alignment_similarity"
            ] = (
                similarity
            )

            unresolved += 1

    # =====================================================
    # APPLY SECONDARY WORD MATCHES TO SHORT SEGMENTS
    # =====================================================

    for (
        segment_index,
        match,
    ) in secondary_word_matches.items():

        segment = (
            aligned_segments[
                segment_index
            ]
        )

        if segment.get(
            "words"
        ):
            continue

        segment[
            "words"
        ] = (
            copy.deepcopy(
                match[
                    "words"
                ]
            )
        )

        segment[
            "alignment_source"
        ] = (
            "secondary_text_match"
        )

        segment[
            "alignment_similarity"
        ] = (
            match[
                "similarity"
            ]
        )

        segment[
            "alignment_text"
        ] = (
            match[
                "text"
            ]
        )

        secondary_short_success += 1

    # =====================================================
    # MISSING TARGET CHECK
    # =====================================================

    missing_alignment_targets = (
        alignment_target_set
        -
        processed_alignment_targets
    )

    if missing_alignment_targets:

        print()
        print(
            "WARNING: missing alignment targets:"
        )

        print(
            sorted(
                missing_alignment_targets
            )
        )

        unresolved += len(
            missing_alignment_targets
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print(
        "Selective batch alignment:"
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
        f"  batch success        : "
        f"{batch_success}"
    )

    print(
        f"  local fallbacks      : "
        f"{local_fallbacks}"
    )

    print(
        f"  local success        : "
        f"{local_success}"
    )

    print(
        f"  short secondary match: "
        f"{secondary_short_success}"
    )

    print(
        f"  unresolved           : "
        f"{unresolved}"
    )

    print(
        f"  secondary passes     : "
        f"{len(secondary_passes)}"
    )

    return (
        aligned_segments,
        secondary_passes,
    )


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def attach_word_timestamps_v2(
    segments: list[dict],
    normalized_audio: str,
    whisper,
    alignment_dir: str,
    media_duration: float,
    max_cue_duration: float,
    max_chars: int,
    max_batch_duration: float,
    max_target_gap: float,
) -> list[dict]:

    (
        aligned_segments,
        _secondary_passes,
    ) = (
        attach_word_timestamps_v3(
            segments=(
                segments
            ),
            normalized_audio=(
                normalized_audio
            ),
            whisper=(
                whisper
            ),
            alignment_dir=(
                alignment_dir
            ),
            media_duration=(
                media_duration
            ),
            max_cue_duration=(
                max_cue_duration
            ),
            max_chars=(
                max_chars
            ),
            max_batch_duration=(
                max_batch_duration
            ),
            max_target_gap=(
                max_target_gap
            ),
        )
    )

    return (
        aligned_segments
    )