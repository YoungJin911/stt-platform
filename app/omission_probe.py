import copy
import os
import re
import subprocess


# =========================================================
# PHASE 3-B.1
# HIGH-PRECISION OMISSION BOUNDARY PROBES
# =========================================================
#
# Purpose:
# - keep omission ASR computationally independent from subtitle alignment
# - reduce Phase 3-B's 33 probes / 348.85 sec probe audio
# - preserve the known true omission:
#
#     "... 시설물을"  +  "또 휴게시설 ..."
#                ↓
#           "살폈습니다." missing
#
# Strategy:
# - stop probing every vaguely incomplete Korean segment
# - score only boundaries with multiple omission signals
# - especially favor:
#     1) object-particle endings such as 을/를
#     2) next segment starting a new discourse unit (또, 또한, 이에...)
#     3) short canonical boundary gaps
#     4) recovery-transition boundaries
#
# omission_v2.py and main.py do not need to change.
# =========================================================


# =========================================================
# CONFIG
# =========================================================

RECOVERY_SOURCES = {
    "recovery",
    "recovery_long",
    "recovery_micro",
}

# Probe window around a selected boundary.
BOUNDARY_LEFT_CONTEXT = 4.5
BOUNDARY_RIGHT_CONTEXT = 4.5

# Canonical boundary gap accepted for risk scoring.
BOUNDARY_MIN_GAP = -0.20
BOUNDARY_MAX_GAP = 1.50

# Stronger bonus range.
PREFERRED_GAP_MIN = 0.10
PREFERRED_GAP_MAX = 1.00

# A boundary must reach this score to trigger local ASR.
MIN_BOUNDARY_RISK_SCORE = 6

# At least this much normalized text on both sides.
MIN_SEGMENT_TEXT_CHARS = 4

# Do not let merged windows become too large.
MAX_PROBE_DURATION = 12.0

# Merge overlapping / nearly adjacent selected probes.
MERGE_WINDOW_GAP = 0.35


# =========================================================
# KOREAN BOUNDARY HEURISTICS
# =========================================================

# Strong signal:
# a phrase ends with an object particle, while the next
# segment appears to begin a new clause/sentence.
#
# Example:
#   "시설물을"  -> likely predicate should follow
STRONG_OBJECT_PARTICLE_SUFFIXES = (
    "을",
    "를",
)

# Medium signal. These are not enough alone.
MEDIUM_PARTICLE_SUFFIXES = (
    "은",
    "는",
    "이",
    "가",
    "에",
    "에서",
    "으로",
    "로",
    "와",
    "과",
    "도",
    "만",
)

# Connective / dependent endings that can indicate that the
# segment ended before the actual predicate or completion.
CONNECTIVE_SUFFIXES = (
    "하고",
    "하며",
    "하면서",
    "했고",
    "지만",
    "는데",
    "는데도",
    "며",
    "면서",
    "거나",
    "따라",
    "위해",
    "때문에",
    "속에",
    "통해",
    "대해",
    "대한",
    "하는",
    "한",
    "된",
    "되는",
    "있는",
    "없는",
    "같은",
    "위한",
)

# New discourse / sentence starter.
# "또" is critical for the known omission case.
NEXT_DISCOURSE_PREFIXES = (
    "또",
    "또한",
    "이에",
    "한편",
    "다만",
    "하지만",
    "그러나",
    "그리고",
    "특히",
    "반면",
    "따라서",
    "이와",
    "이밖에",
    "이밖에도",
    "아울러",
)

# Clear sentence-final forms.
TERMINAL_SUFFIXES = (
    "습니다",
    "했습니다",
    "됩니다",
    "됐습니다",
    "입니다",
    "였습니다",
    "합니다",
    "있습니다",
    "없습니다",
    "예정입니다",
    "계획입니다",
    "보입니다",
    "전했습니다",
    "밝혔습니다",
    "말했습니다",
    "설명했습니다",
    "확인됐습니다",
    "나타났습니다",
    "제기됩니다",
    "예상됩니다",
    "전망됩니다",
    "필요합니다",
    "가능합니다",
    "어렵습니다",
    "높습니다",
    "낮습니다",
    "같습니다",
    "해요",
    "돼요",
    "네요",
    "군요",
    "죠",
    "요",
)


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(
    text: str,
) -> str:

    text = str(
        text or ""
    ).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def compact_text(
    text: str,
) -> str:

    text = normalize_text(
        text
    ).lower()

    return re.sub(
        r"[^\w가-힣]",
        "",
        text,
    )


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


def clean_token(
    token: str,
) -> str:

    token = str(
        token or ""
    ).strip()

    token = re.sub(
        r"^[.!?。！？…,，\"'“”‘’()\[\]{}]+",
        "",
        token,
    )

    token = re.sub(
        r"[.!?。！？…,，\"'“”‘’()\[\]{}]+$",
        "",
        token,
    )

    return token


def first_token(
    text: str,
) -> str:

    pieces = normalize_text(
        text
    ).split()

    if not pieces:
        return ""

    return clean_token(
        pieces[
            0
        ]
    )


def last_token(
    text: str,
) -> str:

    pieces = normalize_text(
        text
    ).split()

    if not pieces:
        return ""

    return clean_token(
        pieces[
            -1
        ]
    )


# =========================================================
# SENTENCE COMPLETENESS
# =========================================================

def has_terminal_punctuation(
    text: str,
) -> bool:

    text = normalize_text(
        text
    )

    return bool(
        re.search(
            r"[.!?。！？…]+[\"'”’)]*$",
            text,
        )
    )


def looks_sentence_complete(
    text: str,
) -> bool:

    text = normalize_text(
        text
    )

    if not text:
        return True

    if has_terminal_punctuation(
        text
    ):
        return True

    token = last_token(
        text
    )

    if not token:
        return True

    for suffix in TERMINAL_SUFFIXES:

        if token.endswith(
            suffix
        ):
            return True

    return False


# =========================================================
# PREFIX / SUFFIX SIGNALS
# =========================================================

def starts_with_discourse_marker(
    text: str,
) -> tuple[
    bool,
    str,
]:

    token = first_token(
        text
    )

    if not token:
        return (
            False,
            "",
        )

    for prefix in NEXT_DISCOURSE_PREFIXES:

        if token.startswith(
            prefix
        ):

            return (
                True,
                prefix,
            )

    return (
        False,
        "",
    )


def suffix_signal(
    token: str,
) -> tuple[
    str,
    str,
]:

    token = clean_token(
        token
    )

    if not token:

        return (
            "none",
            "",
        )

    # Strongest first.
    for suffix in (
        STRONG_OBJECT_PARTICLE_SUFFIXES
    ):

        if token.endswith(
            suffix
        ):

            # Avoid a few obvious lexical false positives.
            # These are words whose final syllable looks like a
            # particle but is normally lexical content.
            lexical_exceptions = {
                "마을",
                "서울",
                "가을",
                "겨울",
            }

            if token in lexical_exceptions:

                return (
                    "none",
                    "",
                )

            return (
                "strong_object",
                suffix,
            )

    for suffix in (
        CONNECTIVE_SUFFIXES
    ):

        if token.endswith(
            suffix
        ):

            return (
                "connective",
                suffix,
            )

    for suffix in (
        MEDIUM_PARTICLE_SUFFIXES
    ):

        if token.endswith(
            suffix
        ):

            return (
                "medium_particle",
                suffix,
            )

    return (
        "none",
        "",
    )


# =========================================================
# BOUNDARY RISK SCORE
# =========================================================

def score_boundary(
    segments: list[dict],
    index: int,
) -> dict:

    left = segments[
        index
    ]

    right = segments[
        index + 1
    ]

    left_text = normalize_text(
        left.get(
            "text",
            "",
        )
    )

    right_text = normalize_text(
        right.get(
            "text",
            "",
        )
    )

    gap = (
        float(
            right[
                "start"
            ]
        )
        -
        float(
            left[
                "end"
            ]
        )
    )

    result = {
        "index": (
            index
        ),
        "score": (
            0
        ),
        "gap": (
            gap
        ),
        "left_text": (
            left_text
        ),
        "right_text": (
            right_text
        ),
        "left_token": (
            last_token(
                left_text
            )
        ),
        "right_token": (
            first_token(
                right_text
            )
        ),
        "suffix_type": (
            "none"
        ),
        "suffix": (
            ""
        ),
        "discourse_prefix": (
            ""
        ),
        "reasons": (
            []
        ),
    }

    if (
        len(
            compact_text(
                left_text
            )
        )
        <
        MIN_SEGMENT_TEXT_CHARS
    ):

        return result

    if (
        len(
            compact_text(
                right_text
            )
        )
        <
        MIN_SEGMENT_TEXT_CHARS
    ):

        return result

    if not (
        BOUNDARY_MIN_GAP
        <=
        gap
        <=
        BOUNDARY_MAX_GAP
    ):

        return result

    # A clearly finished sentence is normally not omission-risk.
    if looks_sentence_complete(
        left_text
    ):

        return result

    score = 1

    result[
        "reasons"
    ].append(
        "left_incomplete"
    )

    (
        suffix_type,
        suffix,
    ) = (
        suffix_signal(
            result[
                "left_token"
            ]
        )
    )

    result[
        "suffix_type"
    ] = suffix_type

    result[
        "suffix"
    ] = suffix

    if (
        suffix_type
        ==
        "strong_object"
    ):

        score += 4

        result[
            "reasons"
        ].append(
            f"object_particle:{suffix}"
        )

    elif (
        suffix_type
        ==
        "connective"
    ):

        score += 2

        result[
            "reasons"
        ].append(
            f"connective:{suffix}"
        )

    elif (
        suffix_type
        ==
        "medium_particle"
    ):

        score += 1

        result[
            "reasons"
        ].append(
            f"particle:{suffix}"
        )

    (
        has_discourse_prefix,
        discourse_prefix,
    ) = (
        starts_with_discourse_marker(
            right_text
        )
    )

    result[
        "discourse_prefix"
    ] = (
        discourse_prefix
    )

    if has_discourse_prefix:

        score += 3

        result[
            "reasons"
        ].append(
                f"next_discourse:{discourse_prefix}"
        )

    if (
        PREFERRED_GAP_MIN
        <=
        gap
        <=
        PREFERRED_GAP_MAX
    ):

        score += 1

        result[
            "reasons"
        ].append(
            "preferred_gap"
        )

    left_source = left.get(
        "source",
        "primary",
    )

    right_source = right.get(
        "source",
        "primary",
    )

    if (
        left_source
        in
        RECOVERY_SOURCES
        or
        right_source
        in
        RECOVERY_SOURCES
    ):

        score += 1

        result[
            "reasons"
        ].append(
            "recovery_transition"
        )

    # Strong object particle is useful, but without any other
    # corroborating signal it should not automatically trigger.
    #
    # Known omission:
    #   시설물을  + 또 ...
    # gets:
    #   incomplete 1
    #   object 4
    #   discourse 3
    #   preferred gap 1
    #   = 9
    result[
        "score"
    ] = (
        score
    )

    return result


# =========================================================
# CONTEXT SEGMENTS
# =========================================================

def collect_context_segment_indices(
    segments: list[dict],
    start: float,
    end: float,
) -> list[int]:

    indices = []

    for index, segment in enumerate(
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

        indices.append(
            index
        )

    return indices


# =========================================================
# PROBE WINDOWS
# =========================================================

def make_boundary_window(
    segments: list[dict],
    boundary: dict,
    media_duration: float,
) -> dict | None:

    index = int(
        boundary[
            "index"
        ]
    )

    left = segments[
        index
    ]

    right = segments[
        index + 1
    ]

    anchor = (
        float(
            left[
                "end"
            ]
        )
        +
        float(
            right[
                "start"
            ]
        )
    ) / 2.0

    start = max(
        0.0,
        anchor
        -
        BOUNDARY_LEFT_CONTEXT,
    )

    end = min(
        float(
            media_duration
        ),
        anchor
        +
        BOUNDARY_RIGHT_CONTEXT,
    )

    if (
        end
        -
        start
        <
        0.50
    ):

        return None

    return {
        "start": (
            start
        ),
        "end": (
            end
        ),
        "reasons": (
            copy.deepcopy(
                boundary[
                    "reasons"
                ]
            )
        ),
        "boundary_indices": [
            index
        ],
        "risk_scores": [
            int(
                boundary[
                    "score"
                ]
            )
        ],
        "debug": [
            {
                "boundary_index": (
                    index
                ),
                "score": (
                    int(
                        boundary[
                            "score"
                        ]
                    )
                ),
                "gap": (
                    float(
                        boundary[
                            "gap"
                        ]
                    )
                ),
                "left_token": (
                    boundary[
                        "left_token"
                    ]
                ),
                "right_token": (
                    boundary[
                        "right_token"
                    ]
                ),
                "suffix_type": (
                    boundary[
                        "suffix_type"
                    ]
                ),
                "suffix": (
                    boundary[
                        "suffix"
                    ]
                ),
                "discourse_prefix": (
                    boundary[
                        "discourse_prefix"
                    ]
                ),
            }
        ],
    }


def merge_probe_windows(
    windows: list[dict],
) -> list[dict]:

    if not windows:
        return []

    windows = sorted(
        windows,
        key=lambda item: (
            item[
                "start"
            ],
            item[
                "end"
            ],
        ),
    )

    merged = []

    for window in windows:

        if not merged:

            merged.append(
                copy.deepcopy(
                    window
                )
            )

            continue

        previous = merged[
            -1
        ]

        gap = (
            float(
                window[
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

        merged_end = max(
            float(
                previous[
                    "end"
                ]
            ),
            float(
                window[
                    "end"
                ]
            ),
        )

        merged_duration = (
            merged_end
            -
            float(
                previous[
                    "start"
                ]
            )
        )

        if (
            gap
            <=
            MERGE_WINDOW_GAP
            and
            merged_duration
            <=
            MAX_PROBE_DURATION
        ):

            previous[
                "end"
            ] = (
                merged_end
            )

            for reason in (
                window.get(
                    "reasons",
                    [],
                )
            ):

                if (
                    reason
                    not in
                    previous[
                        "reasons"
                    ]
                ):

                    previous[
                        "reasons"
                    ].append(
                        reason
                    )

            previous[
                "boundary_indices"
            ].extend(
                index
                for index
                in window.get(
                    "boundary_indices",
                    [],
                )
                if (
                    index
                    not in
                    previous[
                        "boundary_indices"
                    ]
                )
            )

            previous[
                "risk_scores"
            ].extend(
                window.get(
                    "risk_scores",
                    [],
                )
            )

            previous[
                "debug"
            ].extend(
                copy.deepcopy(
                    window.get(
                        "debug",
                        [],
                    )
                )
            )

            continue

        merged.append(
            copy.deepcopy(
                window
            )
        )

    return merged


def build_probe_windows(
    segments: list[dict],
    media_duration: float,
) -> tuple[
    list[dict],
    dict,
]:

    selected_boundaries = []
    all_scored_boundaries = []

    for index in range(
        len(
            segments
        )
        -
        1
    ):

        boundary = (
            score_boundary(
                segments=(
                    segments
                ),
                index=(
                    index
                ),
            )
        )

        if (
            boundary[
                "score"
            ]
            >
            0
        ):

            all_scored_boundaries.append(
                boundary
            )

        if (
            boundary[
                "score"
            ]
            <
            MIN_BOUNDARY_RISK_SCORE
        ):

            continue

        selected_boundaries.append(
            boundary
        )

    windows = []

    for boundary in (
        selected_boundaries
    ):

        window = (
            make_boundary_window(
                segments=(
                    segments
                ),
                boundary=(
                    boundary
                ),
                media_duration=(
                    media_duration
                ),
            )
        )

        if (
            window
            is not None
        ):

            windows.append(
                window
            )

    windows = (
        merge_probe_windows(
            windows
        )
    )

    for window in windows:

        window[
            "context_segment_indices"
        ] = (
            collect_context_segment_indices(
                segments=(
                    segments
                ),
                start=(
                    float(
                        window[
                            "start"
                        ]
                    )
                ),
                end=(
                    float(
                        window[
                            "end"
                        ]
                    )
                ),
            )
        )

    stats = {
        "scored_boundaries": (
            len(
                all_scored_boundaries
            )
        ),
        "selected_boundaries": (
            len(
                selected_boundaries
            )
        ),
        "probe_windows": (
            len(
                windows
            )
        ),
        "probe_audio": sum(
            max(
                0.0,
                float(
                    window[
                        "end"
                    ]
                )
                -
                float(
                    window[
                        "start"
                    ]
                ),
            )
            for window in windows
        ),
    }

    return (
        windows,
        stats,
    )


# =========================================================
# AUDIO EXTRACTION
# =========================================================

def extract_probe_audio(
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
# MAIN
# =========================================================

def build_omission_probe_passes(
    segments: list[dict],
    normalized_audio: str,
    whisper,
    work_dir: str,
    media_duration: float,
) -> list[dict]:
    """
    Create high-precision omission-specific local ASR evidence.

    Output schema remains compatible with omission_v2.py:
        batch_number
        start
        end
        context_segment_indices
        words
        text
    """

    os.makedirs(
        work_dir,
        exist_ok=True,
    )

    (
        windows,
        stats,
    ) = (
        build_probe_windows(
            segments=(
                segments
            ),
            media_duration=(
                media_duration
            ),
        )
    )

    print()

    print(
        "Starting high-precision omission probes..."
    )

    print(
        f"  canonical segments  : "
        f"{len(segments)}"
    )

    print(
        f"  scored boundaries   : "
        f"{stats['scored_boundaries']}"
    )

    print(
        f"  selected boundaries : "
        f"{stats['selected_boundaries']}"
    )

    print(
        f"  probe windows       : "
        f"{stats['probe_windows']}"
    )

    print(
        f"  probe audio total   : "
        f"{stats['probe_audio']:.2f} sec"
    )

    probe_passes = []

    for probe_number, window in enumerate(
        windows,
        start=1,
    ):

        start = float(
            window[
                "start"
            ]
        )

        end = float(
            window[
                "end"
            ]
        )

        duration = max(
            0.01,
            end
            -
            start,
        )

        output_file = os.path.join(
            work_dir,
            f"probe_{probe_number:04d}.wav",
        )

        extract_probe_audio(
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

        print()

        print(
            f"  [OMISSION PROBE {probe_number}] "
            f"{start:.2f} ~ {end:.2f} "
            f"({duration:.2f}s)"
        )

        print(
            f"    boundaries : "
            f"{window.get('boundary_indices', [])}"
        )

        print(
            f"    scores     : "
            f"{window.get('risk_scores', [])}"
        )

        print(
            f"    reasons    : "
            f"{', '.join(window.get('reasons', []))}"
        )

        print(
            f"    context    : "
            f"{window.get('context_segment_indices', [])}"
        )

        for debug_item in (
            window.get(
                "debug",
                [],
            )
        ):

            print(
                "    boundary   : "
                f"idx={debug_item['boundary_index']} "
                f"gap={debug_item['gap']:.2f} "
                f"left='{debug_item['left_token']}' "
                f"right='{debug_item['right_token']}' "
                f"suffix={debug_item['suffix_type']}:{debug_item['suffix']} "
                f"next={debug_item['discourse_prefix'] or '-'}"
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
                    duration
                ),
            )
        )

        print(
            f"    words       : "
            f"{len(words)}"
        )

        probe_passes.append(
            {
                "batch_number": (
                    probe_number
                ),

                "probe_number": (
                    probe_number
                ),

                "probe_source": (
                    "omission_high_precision_local_asr"
                ),

                "start": (
                    start
                ),

                "end": (
                    end
                ),

                "context_segment_indices": (
                    copy.deepcopy(
                        window.get(
                            "context_segment_indices",
                            [],
                        )
                    )
                ),

                "boundary_indices": (
                    copy.deepcopy(
                        window.get(
                            "boundary_indices",
                            [],
                        )
                    )
                ),

                "risk_scores": (
                    copy.deepcopy(
                        window.get(
                            "risk_scores",
                            [],
                        )
                    )
                ),

                "reasons": (
                    copy.deepcopy(
                        window.get(
                            "reasons",
                            [],
                        )
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

    print()

    print(
        "High-precision omission probe generation:"
    )

    print(
        f"  probe passes        : "
        f"{len(probe_passes)}"
    )

    print(
        f"  probe audio total   : "
        f"{stats['probe_audio']:.2f} sec"
    )

    return probe_passes
