import copy
import re

from difflib import SequenceMatcher


# =========================================================
# CONFIG
# =========================================================

#
# canonical token의 문자 중 최소 이 비율 이상이
# alignment word와 대응되어야 실제 word timing을 신뢰
#
MIN_TOKEN_MATCH_RATIO = 0.45


# =========================================================
# CUE TIMING
# =========================================================

#
# 일반 cue 최소 표시 시간
#
MIN_CUE_DURATION = 0.60

#
# cue 사이 최소 간격
#
MIN_CUE_GAP = 0.02


# =========================================================
# SHORT CUE MERGE
# =========================================================

#
# 이 시간보다 짧으면 병합 후보
#
SHORT_CUE_DURATION = 1.00

#
# 이 단어 수 이하일 때만 병합 후보
#
SHORT_CUE_MAX_WORDS = 2

#
# 짧은 cue 병합 시 일반 max duration보다
# 약간 길어지는 것은 허용한다.
#
SHORT_MERGE_MAX_DURATION = 6.50

#
# 병합 후 글자 수도 일반 max_chars보다
# 약간 길게 허용
#
SHORT_MERGE_EXTRA_CHARS = 8


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(
    text: str,
) -> str:

    text = str(
        text or ""
    )

    text = text.lower()

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


def split_text_tokens(
    text: str,
) -> list[str]:

    text = str(
        text or ""
    ).strip()

    if not text:
        return []

    return [
        token
        for token in re.split(
            r"\s+",
            text,
        )
        if token
    ]


def has_sentence_end(
    text: str,
) -> bool:

    text = str(
        text or ""
    ).strip()

    if not text:
        return False

    return bool(
        re.search(
            r"[.!?。！？]$",
            text,
        )
    )


# =========================================================
# GLOBAL WORD TIMELINE
# =========================================================

def collect_global_words(
    segments: list[dict],
) -> list[dict]:

    collected = []

    seen = set()

    for segment in segments:

        for word in segment.get(
            "words",
            [],
        ):

            text = str(
                word.get(
                    "word",
                    "",
                )
            ).strip()

            normalized = (
                normalize_text(
                    text
                )
            )

            if not normalized:
                continue

            try:

                start = float(
                    word[
                        "start"
                    ]
                )

                end = float(
                    word[
                        "end"
                    ]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):

                continue

            if end <= start:
                continue

            #
            # overlapping batch 때문에 동일 word가
            # 여러 segment에 복제될 수 있으므로 dedupe
            #

            key = (
                round(
                    start,
                    2,
                ),

                round(
                    end,
                    2,
                ),

                normalized,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            collected.append(
                {
                    "start": (
                        start
                    ),

                    "end": (
                        end
                    ),

                    "word": (
                        text
                    ),

                    "normalized": (
                        normalized
                    ),

                    "probability": (
                        word.get(
                            "probability"
                        )
                    ),
                }
            )

    collected.sort(
        key=lambda item: (
            item[
                "start"
            ],
            item[
                "end"
            ],
        )
    )

    return collected


# =========================================================
# CANONICAL TOKEN TIMELINE
# =========================================================

def build_canonical_tokens(
    segments: list[dict],
) -> tuple[
    list[dict],
    str,
]:

    tokens = []

    char_cursor = 0

    canonical_chars = []

    for segment_index, segment in enumerate(
        segments
    ):

        text = str(
            segment.get(
                "text",
                "",
            )
        ).strip()

        segment_tokens = (
            split_text_tokens(
                text
            )
        )

        for token_index, token in enumerate(
            segment_tokens
        ):

            normalized = (
                normalize_text(
                    token
                )
            )

            if not normalized:
                continue

            char_start = (
                char_cursor
            )

            canonical_chars.append(
                normalized
            )

            char_cursor += len(
                normalized
            )

            char_end = (
                char_cursor
            )

            tokens.append(
                {
                    "segment_index": (
                        segment_index
                    ),

                    "token_index": (
                        token_index
                    ),

                    "text": (
                        token
                    ),

                    "normalized": (
                        normalized
                    ),

                    "char_start": (
                        char_start
                    ),

                    "char_end": (
                        char_end
                    ),

                    "start": (
                        None
                    ),

                    "end": (
                        None
                    ),

                    "matched_chars": (
                        0
                    ),

                    "matched_word_indices": (
                        set()
                    ),
                }
            )

    return (
        tokens,
        "".join(
            canonical_chars
        ),
    )


# =========================================================
# SECONDARY CHARACTER MAP
# =========================================================

def build_secondary_char_map(
    words: list[dict],
) -> tuple[
    str,
    list[int],
]:

    chars = []

    char_to_word = []

    for word_index, word in enumerate(
        words
    ):

        normalized = (
            word[
                "normalized"
            ]
        )

        for character in normalized:

            chars.append(
                character
            )

            char_to_word.append(
                word_index
            )

    return (
        "".join(
            chars
        ),

        char_to_word,
    )


# =========================================================
# GLOBAL CHARACTER ALIGNMENT
# =========================================================

def align_canonical_to_words(
    canonical_tokens: list[dict],
    canonical_text: str,
    global_words: list[dict],
):

    if not canonical_text:
        return

    if not global_words:
        return

    (
        secondary_text,
        secondary_char_to_word,
    ) = (
        build_secondary_char_map(
            global_words
        )
    )

    if not secondary_text:
        return

    matcher = SequenceMatcher(
        None,
        canonical_text,
        secondary_text,
        autojunk=False,
    )

    canonical_to_secondary = {}

    for block in (
        matcher.get_matching_blocks()
    ):

        if block.size <= 0:
            continue

        for offset in range(
            block.size
        ):

            canonical_position = (
                block.a
                +
                offset
            )

            secondary_position = (
                block.b
                +
                offset
            )

            canonical_to_secondary[
                canonical_position
            ] = (
                secondary_position
            )

    # =====================================================
    # TOKEN -> ACTUAL WORD TIMESTAMP
    # =====================================================

    for token in (
        canonical_tokens
    ):

        char_start = (
            token[
                "char_start"
            ]
        )

        char_end = (
            token[
                "char_end"
            ]
        )

        matched_word_indices = []

        matched_chars = 0

        for canonical_position in range(
            char_start,
            char_end,
        ):

            secondary_position = (
                canonical_to_secondary.get(
                    canonical_position
                )
            )

            if (
                secondary_position
                is None
            ):
                continue

            if (
                secondary_position
                >=
                len(
                    secondary_char_to_word
                )
            ):
                continue

            word_index = (
                secondary_char_to_word[
                    secondary_position
                ]
            )

            matched_word_indices.append(
                word_index
            )

            matched_chars += 1

        token_length = max(
            1,
            char_end
            -
            char_start,
        )

        match_ratio = (
            matched_chars
            /
            token_length
        )

        token[
            "matched_chars"
        ] = (
            matched_chars
        )

        if (
            match_ratio
            <
            MIN_TOKEN_MATCH_RATIO
        ):
            continue

        unique_word_indices = sorted(
            set(
                matched_word_indices
            )
        )

        if not unique_word_indices:
            continue

        matched_words = [
            global_words[
                word_index
            ]
            for word_index
            in unique_word_indices
        ]

        token[
            "matched_word_indices"
        ] = set(
            unique_word_indices
        )

        token[
            "start"
        ] = min(
            word[
                "start"
            ]
            for word in matched_words
        )

        token[
            "end"
        ] = max(
            word[
                "end"
            ]
            for word in matched_words
        )


# =========================================================
# MISSING TOKEN TIMING INTERPOLATION
# =========================================================

def interpolate_missing_token_times(
    tokens: list[dict],
    segments: list[dict],
):

    if not tokens:
        return

    previous_anchor = [
        None
        for _ in tokens
    ]

    next_anchor = [
        None
        for _ in tokens
    ]

    current = None

    for index, token in enumerate(
        tokens
    ):

        if (
            token[
                "start"
            ]
            is not None
            and
            token[
                "end"
            ]
            is not None
        ):

            current = (
                index
            )

        previous_anchor[
            index
        ] = (
            current
        )

    current = None

    for index in range(
        len(
            tokens
        )
        -
        1,
        -1,
        -1,
    ):

        token = (
            tokens[
                index
            ]
        )

        if (
            token[
                "start"
            ]
            is not None
            and
            token[
                "end"
            ]
            is not None
        ):

            current = (
                index
            )

        next_anchor[
            index
        ] = (
            current
        )

    # =====================================================
    # SEGMENT-BY-SEGMENT
    # =====================================================

    for segment_index, segment in enumerate(
        segments
    ):

        segment_token_indices = [
            index
            for index, token
            in enumerate(
                tokens
            )
            if (
                token[
                    "segment_index"
                ]
                ==
                segment_index
            )
        ]

        if not segment_token_indices:
            continue

        position = 0

        while position < len(
            segment_token_indices
        ):

            token_index = (
                segment_token_indices[
                    position
                ]
            )

            token = (
                tokens[
                    token_index
                ]
            )

            if (
                token[
                    "start"
                ]
                is not None
            ):

                position += 1

                continue

            run_start_position = (
                position
            )

            while (
                position
                <
                len(
                    segment_token_indices
                )
            ):

                current_index = (
                    segment_token_indices[
                        position
                    ]
                )

                if (
                    tokens[
                        current_index
                    ][
                        "start"
                    ]
                    is not None
                ):

                    break

                position += 1

            run_end_position = (
                position
                -
                1
            )

            run_indices = (
                segment_token_indices[
                    run_start_position:
                    run_end_position + 1
                ]
            )

            first_run_index = (
                run_indices[
                    0
                ]
            )

            last_run_index = (
                run_indices[
                    -1
                ]
            )

            previous_index = (
                previous_anchor[
                    first_run_index
                ]
            )

            next_index = (
                next_anchor[
                    last_run_index
                ]
            )

            # =================================================
            # LEFT ANCHOR
            # =================================================

            if (
                previous_index
                is not None
                and
                previous_index
                <
                first_run_index
            ):

                left_time = float(
                    tokens[
                        previous_index
                    ][
                        "end"
                    ]
                )

            else:

                left_time = float(
                    segment[
                        "start"
                    ]
                )

            # =================================================
            # RIGHT ANCHOR
            #
            # 다음 canonical segment의 실제 word timing도
            # anchor로 사용할 수 있다.
            # =================================================

            if (
                next_index
                is not None
                and
                next_index
                >
                last_run_index
            ):

                right_time = float(
                    tokens[
                        next_index
                    ][
                        "start"
                    ]
                )

            else:

                right_time = float(
                    segment[
                        "end"
                    ]
                )

            if (
                right_time
                <=
                left_time
            ):

                right_time = max(
                    float(
                        segment[
                            "end"
                        ]
                    ),

                    left_time
                    +
                    (
                        0.20
                        *
                        len(
                            run_indices
                        )
                    ),
                )

            weights = [
                max(
                    1,
                    len(
                        normalize_text(
                            tokens[
                                index
                            ][
                                "text"
                            ]
                        )
                    ),
                )
                for index in run_indices
            ]

            total_weight = sum(
                weights
            )

            available_duration = max(
                0.05,
                right_time
                -
                left_time,
            )

            cursor = (
                left_time
            )

            for (
                run_index,
                weight,
            ) in zip(
                run_indices,
                weights,
            ):

                duration = (
                    available_duration
                    *
                    (
                        weight
                        /
                        total_weight
                    )
                )

                duration = max(
                    0.05,
                    duration,
                )

                tokens[
                    run_index
                ][
                    "start"
                ] = (
                    cursor
                )

                tokens[
                    run_index
                ][
                    "end"
                ] = (
                    cursor
                    +
                    duration
                )

                cursor += (
                    duration
                )


# =========================================================
# TOKEN TIMING SANITY
# =========================================================

def sanitize_token_times(
    tokens: list[dict],
):

    previous_start = (
        0.0
    )

    previous_end = (
        0.0
    )

    for token in (
        tokens
    ):

        if (
            token[
                "start"
            ]
            is None
        ):

            token[
                "start"
            ] = (
                previous_end
            )

        if (
            token[
                "end"
            ]
            is None
        ):

            token[
                "end"
            ] = (
                token[
                    "start"
                ]
                +
                0.10
            )

        start = float(
            token[
                "start"
            ]
        )

        end = float(
            token[
                "end"
            ]
        )

        if (
            start
            <
            previous_start
        ):

            start = (
                previous_start
            )

        if end < start:

            end = (
                start
                +
                0.05
            )

        token[
            "start"
        ] = (
            start
        )

        token[
            "end"
        ] = (
            end
        )

        previous_start = (
            start
        )

        previous_end = max(
            previous_end,
            end,
        )


# =========================================================
# CUE SPLITTING
# =========================================================

def should_break_cue(
    current_tokens: list[dict],
    next_token: dict,
    target_chars: int,
    max_chars: int,
    max_cue_duration: float,
) -> bool:

    if not current_tokens:
        return False

    current_text = " ".join(
        token[
            "text"
        ]
        for token
        in current_tokens
    )

    candidate_text = (
        current_text
        +
        " "
        +
        next_token[
            "text"
        ]
    ).strip()

    candidate_start = float(
        current_tokens[
            0
        ][
            "start"
        ]
    )

    candidate_end = float(
        next_token[
            "end"
        ]
    )

    candidate_duration = (
        candidate_end
        -
        candidate_start
    )

    # =====================================================
    # HARD LIMIT
    # =====================================================

    if (
        len(
            candidate_text
        )
        >
        max_chars
    ):

        return True

    if (
        candidate_duration
        >
        max_cue_duration
    ):

        return True

    # =====================================================
    # NATURAL SENTENCE BREAK
    # =====================================================

    if (
        len(
            current_text
        )
        >=
        target_chars
    ):

        last_text = (
            current_tokens[
                -1
            ][
                "text"
            ]
        )

        if has_sentence_end(
            last_text
        ):

            return True

    return False


def build_cue(
    cue_tokens: list[dict],
) -> dict | None:

    if not cue_tokens:
        return None

    text = " ".join(
        token[
            "text"
        ]
        for token in cue_tokens
    ).strip()

    if not text:
        return None

    start = float(
        cue_tokens[
            0
        ][
            "start"
        ]
    )

    end = float(
        cue_tokens[
            -1
        ][
            "end"
        ]
    )

    if (
        end
        -
        start
        <
        MIN_CUE_DURATION
    ):

        end = (
            start
            +
            MIN_CUE_DURATION
        )

    return {
        "start": (
            start
        ),

        "end": (
            end
        ),

        "text": (
            text
        ),

        "source": (
            "global_word_timeline"
        ),
    }


# =========================================================
# PROPORTIONAL FALLBACK
# =========================================================

def proportional_segment_fallback(
    segment: dict,
    target_chars: int,
    max_chars: int,
    max_cue_duration: float,
) -> list[dict]:

    text = str(
        segment.get(
            "text",
            "",
        )
    ).strip()

    if not text:
        return []

    tokens = (
        split_text_tokens(
            text
        )
    )

    if not tokens:
        return []

    start = float(
        segment[
            "start"
        ]
    )

    end = float(
        segment[
            "end"
        ]
    )

    duration = max(
        0.10,
        end
        -
        start,
    )

    chunks = []

    current = []

    for token in tokens:

        candidate = (
            " ".join(
                current
                +
                [
                    token
                ]
            )
        )

        if (
            current
            and
            len(
                candidate
            )
            >
            max_chars
        ):

            chunks.append(
                current
            )

            current = [
                token
            ]

        else:

            current.append(
                token
            )

    if current:

        chunks.append(
            current
        )

    total_chars = sum(
        max(
            1,
            len(
                normalize_text(
                    " ".join(
                        chunk
                    )
                )
            ),
        )
        for chunk
        in chunks
    )

    cursor = (
        start
    )

    results = []

    for chunk in chunks:

        chunk_text = (
            " ".join(
                chunk
            )
        )

        chunk_chars = max(
            1,
            len(
                normalize_text(
                    chunk_text
                )
            ),
        )

        chunk_duration = (
            duration
            *
            (
                chunk_chars
                /
                total_chars
            )
        )

        chunk_duration = min(
            max_cue_duration,
            max(
                MIN_CUE_DURATION,
                chunk_duration,
            ),
        )

        chunk_end = min(
            end,
            cursor
            +
            chunk_duration,
        )

        results.append(
            {
                "start": (
                    cursor
                ),

                "end": (
                    chunk_end
                ),

                "text": (
                    chunk_text
                ),

                "source": (
                    "proportional_fallback"
                ),
            }
        )

        cursor = (
            chunk_end
        )

    return results


# =========================================================
# SHORT CUE MERGE HELPERS
# =========================================================

def cue_duration(
    cue: dict,
) -> float:

    return max(
        0.0,
        float(
            cue[
                "end"
            ]
        )
        -
        float(
            cue[
                "start"
            ]
        ),
    )


def cue_word_count(
    cue: dict,
) -> int:

    return len(
        split_text_tokens(
            cue.get(
                "text",
                "",
            )
        )
    )


def is_short_cue(
    cue: dict,
) -> bool:

    duration = (
        cue_duration(
            cue
        )
    )

    word_count = (
        cue_word_count(
            cue
        )
    )

    return (
        duration
        <
        SHORT_CUE_DURATION
        and
        word_count
        <=
        SHORT_CUE_MAX_WORDS
    )


def can_merge_cues(
    left: dict,
    right: dict,
    max_chars: int,
) -> bool:

    merged_start = float(
        left[
            "start"
        ]
    )

    merged_end = float(
        right[
            "end"
        ]
    )

    merged_duration = (
        merged_end
        -
        merged_start
    )

    if (
        merged_duration
        >
        SHORT_MERGE_MAX_DURATION
    ):

        return False

    merged_text = (
        str(
            left.get(
                "text",
                "",
            )
        ).strip()
        +
        " "
        +
        str(
            right.get(
                "text",
                "",
            )
        ).strip()
    ).strip()

    if (
        len(
            merged_text
        )
        >
        (
            max_chars
            +
            SHORT_MERGE_EXTRA_CHARS
        )
    ):

        return False

    return True


# =========================================================
# MERGE SHORT CUES
# =========================================================

def merge_short_cues(
    cues: list[dict],
    max_chars: int,
) -> tuple[
    list[dict],
    int,
]:

    if len(
        cues
    ) < 2:

        return (
            cues,
            0,
        )

    cues = [
        copy.deepcopy(
            cue
        )
        for cue in cues
    ]

    merge_count = 0

    index = 0

    while index < len(
        cues
    ):

        cue = (
            cues[
                index
            ]
        )

        if not is_short_cue(
            cue
        ):

            index += 1

            continue

        previous = (
            cues[
                index - 1
            ]
            if index > 0
            else None
        )

        next_cue = (
            cues[
                index + 1
            ]
            if (
                index + 1
                <
                len(
                    cues
                )
            )
            else None
        )

        can_previous = (
            previous
            is not None
            and
            can_merge_cues(
                previous,
                cue,
                max_chars,
            )
        )

        can_next = (
            next_cue
            is not None
            and
            can_merge_cues(
                cue,
                next_cue,
                max_chars,
            )
        )

        # =================================================
        # NO AVAILABLE MERGE
        # =================================================

        if (
            not can_previous
            and
            not can_next
        ):

            index += 1

            continue

        # =================================================
        # CHOOSE DIRECTION
        # =================================================

        merge_to_next = False

        if (
            can_next
            and
            not can_previous
        ):

            merge_to_next = True

        elif (
            can_previous
            and
            not can_next
        ):

            merge_to_next = False

        else:

            # ---------------------------------------------
            # 둘 다 가능할 경우
            #
            # 이전 cue가 문장 종결이면 다음에 붙인다.
            #
            # 예:
            # "... 산불"
            # "위험이"
            # "커짐에 따라..."
            #
            # "위험이 커짐에 따라..." 쪽이 자연스럽다.
            # ---------------------------------------------

            previous_text = str(
                previous.get(
                    "text",
                    "",
                )
            ).strip()

            cue_text = str(
                cue.get(
                    "text",
                    "",
                )
            ).strip()

            if has_sentence_end(
                previous_text
            ):

                merge_to_next = True

            elif has_sentence_end(
                cue_text
            ):

                merge_to_next = False

            else:

                # -----------------------------------------
                # 앞/뒤 중 병합 후 duration이 짧은 쪽 선택
                # -----------------------------------------

                previous_duration = (
                    float(
                        cue[
                            "end"
                        ]
                    )
                    -
                    float(
                        previous[
                            "start"
                        ]
                    )
                )

                next_duration = (
                    float(
                        next_cue[
                            "end"
                        ]
                    )
                    -
                    float(
                        cue[
                            "start"
                        ]
                    )
                )

                #
                # 짧은 cue는 보통 다음 말과 연결되는
                # 조사/주어/목적어인 경우가 많아서
                # duration 차이가 작으면 NEXT 우선
                #

                if abs(
                    previous_duration
                    -
                    next_duration
                ) <= 0.50:

                    merge_to_next = True

                else:

                    merge_to_next = (
                        next_duration
                        <
                        previous_duration
                    )

        # =================================================
        # MERGE TO NEXT
        # =================================================

        if merge_to_next:

            merged = {
                "start": (
                    float(
                        cue[
                            "start"
                        ]
                    )
                ),

                "end": (
                    float(
                        next_cue[
                            "end"
                        ]
                    )
                ),

                "text": (
                    str(
                        cue.get(
                            "text",
                            "",
                        )
                    ).strip()
                    +
                    " "
                    +
                    str(
                        next_cue.get(
                            "text",
                            "",
                        )
                    ).strip()
                ).strip(),

                "source": (
                    "short_cue_merge_next"
                ),
            }

            cues[
                index:
                index + 2
            ] = [
                merged
            ]

            merge_count += 1

            #
            # 새 merged cue가 또 너무 짧은지
            # 같은 위치에서 재검사
            #

            continue

        # =================================================
        # MERGE TO PREVIOUS
        # =================================================

        merged = {
            "start": (
                float(
                    previous[
                        "start"
                    ]
                )
            ),

            "end": (
                float(
                    cue[
                        "end"
                    ]
                )
            ),

            "text": (
                str(
                    previous.get(
                        "text",
                        "",
                    )
                ).strip()
                +
                " "
                +
                str(
                    cue.get(
                        "text",
                        "",
                    )
                ).strip()
            ).strip(),

            "source": (
                "short_cue_merge_previous"
            ),
        }

        cues[
            index - 1:
            index + 1
        ] = [
            merged
        ]

        merge_count += 1

        index = max(
            0,
            index - 1,
        )

    return (
        cues,
        merge_count,
    )


# =========================================================
# REMOVE CUE OVERLAP
# =========================================================

def sanitize_cues(
    cues: list[dict],
) -> list[dict]:

    if not cues:
        return []

    cues = sorted(
        cues,
        key=lambda cue: (
            cue[
                "start"
            ],
            cue[
                "end"
            ],
        )
    )

    results = []

    for cue in cues:

        cue = (
            copy.deepcopy(
                cue
            )
        )

        cue_start = float(
            cue[
                "start"
            ]
        )

        cue_end = float(
            cue[
                "end"
            ]
        )

        if results:

            previous = (
                results[
                    -1
                ]
            )

            previous_end = float(
                previous[
                    "end"
                ]
            )

            if (
                cue_start
                <
                previous_end
            ):

                midpoint = (
                    cue_start
                    +
                    previous_end
                ) / 2.0

                previous[
                    "end"
                ] = max(
                    float(
                        previous[
                            "start"
                        ]
                    )
                    +
                    0.10,

                    midpoint
                    -
                    (
                        MIN_CUE_GAP
                        /
                        2.0
                    ),
                )

                cue_start = (
                    midpoint
                    +
                    (
                        MIN_CUE_GAP
                        /
                        2.0
                    )
                )

        if (
            cue_end
            <=
            cue_start
        ):

            cue_end = (
                cue_start
                +
                MIN_CUE_DURATION
            )

        cue[
            "start"
        ] = (
            cue_start
        )

        cue[
            "end"
        ] = (
            cue_end
        )

        results.append(
            cue
        )

    return results


# =========================================================
# MAIN FORMATTER
# =========================================================

def format_subtitle_segments(
    segments: list[dict],
    max_cue_duration: float = 6.0,
    target_chars: int = 24,
    max_chars: int = 42,
) -> list[dict]:

    if not segments:
        return []

    # =====================================================
    # GLOBAL WORD TIMELINE
    # =====================================================

    global_words = (
        collect_global_words(
            segments
        )
    )

    (
        canonical_tokens,
        canonical_text,
    ) = (
        build_canonical_tokens(
            segments
        )
    )

    align_canonical_to_words(
        canonical_tokens=(
            canonical_tokens
        ),

        canonical_text=(
            canonical_text
        ),

        global_words=(
            global_words
        ),
    )

    interpolate_missing_token_times(
        tokens=(
            canonical_tokens
        ),

        segments=(
            segments
        ),
    )

    sanitize_token_times(
        canonical_tokens
    )

    # =====================================================
    # BUILD RAW CUES
    # =====================================================

    results = []

    word_based_cues = 0

    proportional_cues = 0

    passthrough_cues = 0

    for segment_index, segment in enumerate(
        segments
    ):

        segment_tokens = [
            token
            for token
            in canonical_tokens
            if (
                token[
                    "segment_index"
                ]
                ==
                segment_index
            )
        ]

        if not segment_tokens:

            fallback = (
                proportional_segment_fallback(
                    segment=(
                        segment
                    ),

                    target_chars=(
                        target_chars
                    ),

                    max_chars=(
                        max_chars
                    ),

                    max_cue_duration=(
                        max_cue_duration
                    ),
                )
            )

            results.extend(
                fallback
            )

            proportional_cues += len(
                fallback
            )

            continue

        matched_token_count = sum(
            1
            for token
            in segment_tokens
            if (
                token[
                    "matched_chars"
                ]
                >
                0
            )
        )

        match_ratio = (
            matched_token_count
            /
            len(
                segment_tokens
            )
        )

        # =================================================
        # VERY LOW ALIGNMENT COVERAGE
        # =================================================

        if (
            match_ratio
            <
            0.15
        ):

            fallback = (
                proportional_segment_fallback(
                    segment=(
                        segment
                    ),

                    target_chars=(
                        target_chars
                    ),

                    max_chars=(
                        max_chars
                    ),

                    max_cue_duration=(
                        max_cue_duration
                    ),
                )
            )

            results.extend(
                fallback
            )

            proportional_cues += len(
                fallback
            )

            continue

        # =================================================
        # WORD-BASED
        # =================================================

        current = []

        for token in (
            segment_tokens
        ):

            if (
                current
                and
                should_break_cue(
                    current_tokens=(
                        current
                    ),

                    next_token=(
                        token
                    ),

                    target_chars=(
                        target_chars
                    ),

                    max_chars=(
                        max_chars
                    ),

                    max_cue_duration=(
                        max_cue_duration
                    ),
                )
            ):

                cue = (
                    build_cue(
                        current
                    )
                )

                if cue:

                    results.append(
                        cue
                    )

                    word_based_cues += 1

                current = []

            current.append(
                token
            )

        if current:

            cue = (
                build_cue(
                    current
                )
            )

            if cue:

                results.append(
                    cue
                )

                word_based_cues += 1

    # =====================================================
    # BASIC TIMING SANITIZE
    # =====================================================

    results = (
        sanitize_cues(
            results
        )
    )

    before_short_merge = len(
        results
    )

    # =====================================================
    # NEW:
    # SHORT CUE READABILITY MERGE
    # =====================================================

    (
        results,
        short_merge_count,
    ) = (
        merge_short_cues(
            cues=(
                results
            ),

            max_chars=(
                max_chars
            ),
        )
    )

    #
    # 병합 후 timing overlap 다시 정리
    #

    results = (
        sanitize_cues(
            results
        )
    )

    # =====================================================
    # LOG
    # =====================================================

    print()
    print(
        "Subtitle formatter:"
    )

    print(
        f"  global words       : "
        f"{len(global_words)}"
    )

    print(
        f"  canonical tokens   : "
        f"{len(canonical_tokens)}"
    )

    print(
        f"  word-based cues    : "
        f"{word_based_cues}"
    )

    print(
        f"  passthrough cues   : "
        f"{passthrough_cues}"
    )

    print(
        f"  proportional cues  : "
        f"{proportional_cues}"
    )

    print(
        f"  before short merge : "
        f"{before_short_merge}"
    )

    print(
        f"  short cue merges   : "
        f"{short_merge_count}"
    )

    print(
        f"  total cues         : "
        f"{len(results)}"
    )

    return (
        results
    )