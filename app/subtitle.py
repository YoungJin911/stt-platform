import copy
import re

from difflib import SequenceMatcher


# =========================================================
# CONFIG
# =========================================================

MIN_TOKEN_MATCH_RATIO = 0.45


# =========================================================
# CUE TIMING
# =========================================================

MIN_CUE_DURATION = 0.60
MIN_CUE_GAP = 0.02

# 실제 발화 사이 공백이 이 이상이면 강제로 cue를 분리
MAX_INTER_TOKEN_GAP = 1.20

# cue가 너무 짧을 때 의미 단위 분리를 억제
MIN_SEMANTIC_CUE_CHARS = 12
MIN_SEMANTIC_CUE_DURATION = 1.20

# 사람이 읽기 편한 평균 cue 지속시간 목표
TARGET_CUE_DURATION = 3.40

# hard limit 직전 breakpoint 탐색 시 너무 짧은 앞 조각 방지
MIN_BREAK_CHARS = 10
MIN_BREAK_DURATION = 1.00


# =========================================================
# KOREAN SEMANTIC BREAK HINTS
# =========================================================

SENTENCE_END_PATTERN = re.compile(
    r"[.!?。！？]$"
)

# 이 어절 뒤는 의미 단위 경계가 되기 쉬움.
# 완전한 형태소 분석기가 아니라 "힌트"로만 사용한다.
PREFERRED_BREAK_SUFFIXES = (
    "하고",
    "하며",
    "하면서",
    "했으며",
    "됐으며",
    "되며",
    "한편",
    "따라",
    "위해",
    "때문에",
    "경우",
    "반면",
)

# 다음 어절이 이런 담화 표지이면 그 직전에 끊는 것이 자연스러운 경우가 많음.
PREFERRED_NEXT_PREFIXES = (
    "또",
    "또한",
    "이에",
    "다만",
    "하지만",
    "그러나",
    "그리고",
    "한편",
    "특히",
    "반면",
    "따라서",
)

# 짧은 명사/수식어가 바로 뒤 어절과 매우 촘촘히 붙어 있으면
# 중간 분리를 강하게 억제한다.
SHORT_PAIR_MAX_GAP = 0.16
SHORT_PAIR_MAX_LEFT_CHARS = 2
SHORT_PAIR_MAX_RIGHT_CHARS = 4


# =========================================================
# SHORT CUE MERGE
# =========================================================

SHORT_CUE_DURATION = 1.00
SHORT_CUE_MAX_WORDS = 2
SHORT_MERGE_MAX_DURATION = 6.50
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
        SENTENCE_END_PATTERN.search(
            text
        )
    )


def clean_token_text(
    text: str,
) -> str:

    return re.sub(
        r"[.!?。！？,，:;\"'“”‘’()\[\]{}]",
        "",
        str(
            text or ""
        ).strip(),
    )


def displayed_text(
    tokens: list[dict],
) -> str:

    return " ".join(
        str(
            token.get(
                "text",
                "",
            )
        ).strip()
        for token in tokens
        if str(
            token.get(
                "text",
                "",
            )
        ).strip()
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

            normalized = normalize_text(
                text
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
                    "start": start,
                    "end": end,
                    "word": text,
                    "normalized": normalized,
                    "probability": word.get(
                        "probability"
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

        segment_tokens = split_text_tokens(
            text
        )

        for token_index, token in enumerate(
            segment_tokens
        ):

            normalized = normalize_text(
                token
            )

            if not normalized:
                continue

            char_start = char_cursor

            canonical_chars.append(
                normalized
            )

            char_cursor += len(
                normalized
            )

            char_end = char_cursor

            tokens.append(
                {
                    "segment_index": segment_index,
                    "token_index": token_index,
                    "text": token,
                    "normalized": normalized,
                    "char_start": char_start,
                    "char_end": char_end,
                    "start": None,
                    "end": None,
                    "matched_chars": 0,
                    "matched_word_indices": set(),
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

        normalized = word[
            "normalized"
        ]

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
    ) = build_secondary_char_map(
        global_words
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

    for block in matcher.get_matching_blocks():

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
            ] = secondary_position

    for token in canonical_tokens:

        char_start = token[
            "char_start"
        ]

        char_end = token[
            "char_end"
        ]

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

            if secondary_position is None:
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
        ] = matched_chars

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
            current = index

        previous_anchor[
            index
        ] = current

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

        token = tokens[
            index
        ]

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
            current = index

        next_anchor[
            index
        ] = current

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

        while (
            position
            <
            len(
                segment_token_indices
            )
        ):

            token_index = (
                segment_token_indices[
                    position
                ]
            )

            token = tokens[
                token_index
            ]

            if (
                token[
                    "start"
                ]
                is not None
            ):
                position += 1
                continue

            run_start_position = position

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

            first_run_index = run_indices[
                0
            ]

            last_run_index = run_indices[
                -1
            ]

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
                for index
                in run_indices
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

            cursor = left_time

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
                ] = cursor

                tokens[
                    run_index
                ][
                    "end"
                ] = (
                    cursor
                    +
                    duration
                )

                cursor += duration


# =========================================================
# TOKEN TIMING SANITY
# =========================================================

def sanitize_token_times(
    tokens: list[dict],
):

    previous_start = 0.0
    previous_end = 0.0

    for token in tokens:

        if (
            token[
                "start"
            ]
            is None
        ):
            token[
                "start"
            ] = previous_end

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
            start = previous_start

        if end < start:
            end = (
                start
                +
                0.05
            )

        token[
            "start"
        ] = start

        token[
            "end"
        ] = end

        previous_start = start
        previous_end = max(
            previous_end,
            end,
        )


# =========================================================
# SEMANTIC BREAK SCORING
# =========================================================

def token_gap(
    left_token: dict,
    right_token: dict,
) -> float:

    return max(
        0.0,
        float(
            right_token[
                "start"
            ]
        )
        -
        float(
            left_token[
                "end"
            ]
        ),
    )


def is_preferred_break_suffix(
    text: str,
) -> bool:

    cleaned = clean_token_text(
        text
    )

    if not cleaned:
        return False

    return any(
        cleaned.endswith(
            suffix
        )
        for suffix
        in PREFERRED_BREAK_SUFFIXES
    )


def is_preferred_next_prefix(
    text: str,
) -> bool:

    cleaned = clean_token_text(
        text
    )

    if not cleaned:
        return False

    return any(
        cleaned.startswith(
            prefix
        )
        for prefix
        in PREFERRED_NEXT_PREFIXES
    )


def is_tight_short_pair(
    left_token: dict,
    right_token: dict,
) -> bool:

    left_text = clean_token_text(
        left_token.get(
            "text",
            "",
        )
    )

    right_text = clean_token_text(
        right_token.get(
            "text",
            "",
        )
    )

    if not left_text:
        return False

    if not right_text:
        return False

    gap = token_gap(
        left_token,
        right_token,
    )

    return (
        gap
        <=
        SHORT_PAIR_MAX_GAP
        and
        len(
            normalize_text(
                left_text
            )
        )
        <=
        SHORT_PAIR_MAX_LEFT_CHARS
        and
        len(
            normalize_text(
                right_text
            )
        )
        <=
        SHORT_PAIR_MAX_RIGHT_CHARS
    )


def break_score(
    left_tokens: list[dict],
    right_token: dict,
    target_chars: int,
) -> float:

    if not left_tokens:
        return -9999.0

    left_text = displayed_text(
        left_tokens
    )

    if not left_text:
        return -9999.0

    first_token = left_tokens[
        0
    ]

    last_token = left_tokens[
        -1
    ]

    chars = len(
        left_text
    )

    duration = max(
        0.0,
        float(
            last_token[
                "end"
            ]
        )
        -
        float(
            first_token[
                "start"
            ]
        ),
    )

    gap = token_gap(
        last_token,
        right_token,
    )

    score = 0.0

    # -----------------------------------------------------
    # 길이 / 지속시간이 목표값 근처일수록 가산점
    # -----------------------------------------------------

    score += max(
        0.0,
        20.0
        -
        (
            abs(
                chars
                -
                target_chars
            )
            *
            0.9
        ),
    )

    score += max(
        0.0,
        10.0
        -
        (
            abs(
                duration
                -
                TARGET_CUE_DURATION
            )
            *
            2.2
        ),
    )

    # -----------------------------------------------------
    # 실제 발화 pause는 매우 강한 힌트
    # -----------------------------------------------------

    if gap >= 0.60:
        score += 70.0

    elif gap >= 0.35:
        score += 45.0

    elif gap >= 0.22:
        score += 28.0

    elif gap >= 0.14:
        score += 12.0

    # -----------------------------------------------------
    # 문장 종결은 가장 강한 의미 경계
    # -----------------------------------------------------

    if has_sentence_end(
        last_token.get(
            "text",
            "",
        )
    ):
        score += 100.0

    # -----------------------------------------------------
    # 연결형 어미
    #
    # 예: "배치하고 | 순찰과 단속을..."
    # -----------------------------------------------------

    if is_preferred_break_suffix(
        last_token.get(
            "text",
            "",
        )
    ):
        score += 30.0

    # -----------------------------------------------------
    # 다음 어절이 새 문장/담화 단위로 시작
    # -----------------------------------------------------

    if is_preferred_next_prefix(
        right_token.get(
            "text",
            "",
        )
    ):
        score += 34.0

    # -----------------------------------------------------
    # 짧은 두 어절이 pause 없이 붙어 있으면
    # 중간 절단 강하게 억제
    #
    # 예: "고정 | 배치하고", "주의 | 단계로"
    # -----------------------------------------------------

    if is_tight_short_pair(
        last_token,
        right_token,
    ):
        score -= 90.0

    # -----------------------------------------------------
    # 지나치게 짧은 cue 방지
    # -----------------------------------------------------

    if (
        chars
        <
        MIN_BREAK_CHARS
    ):
        score -= 50.0

    if (
        duration
        <
        MIN_BREAK_DURATION
    ):
        score -= 40.0

    return score


def choose_best_break_index(
    tokens: list[dict],
    target_chars: int,
) -> int:

    """
    tokens 안에서 가장 자연스러운 분할 위치를 찾는다.

    반환값:
        tokens[:index]  -> 현재 cue
        tokens[index:]  -> 다음 cue
    """

    if len(
        tokens
    ) <= 1:
        return 1

    best_index = None
    best_score = -999999.0

    # 마지막 token 하나는 다음 cue 후보로 남겨둔다.
    for index in range(
        1,
        len(
            tokens
        ),
    ):

        left_tokens = tokens[
            :index
        ]

        right_token = tokens[
            index
        ]

        score = break_score(
            left_tokens=(
                left_tokens
            ),
            right_token=(
                right_token
            ),
            target_chars=(
                target_chars
            ),
        )

        # 동점이면 뒤쪽 breakpoint를 선호
        if (
            best_index is None
            or
            score
            >
            best_score
            or
            (
                abs(
                    score
                    -
                    best_score
                )
                <
                0.001
                and
                index
                >
                best_index
            )
        ):
            best_index = index
            best_score = score

    if best_index is None:
        return max(
            1,
            len(
                tokens
            )
            -
            1,
        )

    return best_index


# =========================================================
# BUILD CUE
# =========================================================

def build_cue(
    cue_tokens: list[dict],
) -> dict | None:

    if not cue_tokens:
        return None

    text = displayed_text(
        cue_tokens
    )

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
        "start": start,
        "end": end,
        "text": text,
        "source": (
            "semantic_global_timeline"
        ),
    }


# =========================================================
# SEMANTIC GLOBAL CUE BUILDER
# =========================================================

def build_semantic_global_cues(
    tokens: list[dict],
    target_chars: int,
    max_chars: int,
    max_cue_duration: float,
) -> list[dict]:

    if not tokens:
        return []

    results = []
    current = []

    def flush(
        cue_tokens: list[dict],
    ):

        cue = build_cue(
            cue_tokens
        )

        if cue:
            results.append(
                cue
            )

    for token in tokens:

        if not current:
            current.append(
                token
            )
            continue

        previous = current[
            -1
        ]

        gap = token_gap(
            previous,
            token,
        )

        # -------------------------------------------------
        # 긴 실제 음성 gap이면 먼저 현재 cue 종료
        # -------------------------------------------------

        if (
            gap
            >
            MAX_INTER_TOKEN_GAP
        ):
            flush(
                current
            )

            current = [
                token
            ]

            continue

        # -------------------------------------------------
        # 새 token까지 포함해서 hard limit 확인
        # -------------------------------------------------

        candidate = (
            current
            +
            [
                token
            ]
        )

        candidate_text = displayed_text(
            candidate
        )

        candidate_duration = (
            float(
                candidate[
                    -1
                ][
                    "end"
                ]
            )
            -
            float(
                candidate[
                    0
                ][
                    "start"
                ]
            )
        )

        overflow = (
            len(
                candidate_text
            )
            >
            max_chars
            or
            candidate_duration
            >
            max_cue_duration
        )

        if overflow:

            # current + token 범위 안에서
            # hard limit 이전의 가장 자연스러운 위치를 선택
            break_index = (
                choose_best_break_index(
                    tokens=(
                        candidate
                    ),
                    target_chars=(
                        target_chars
                    ),
                )
            )

            left = candidate[
                :break_index
            ]

            right = candidate[
                break_index:
            ]

            flush(
                left
            )

            current = right

            # 드물게 남은 right 자체가 또 너무 길다면 반복 분리
            while len(
                current
            ) > 1:

                current_text = displayed_text(
                    current
                )

                current_duration = (
                    float(
                        current[
                            -1
                        ][
                            "end"
                        ]
                    )
                    -
                    float(
                        current[
                            0
                        ][
                            "start"
                        ]
                    )
                )

                if (
                    len(
                        current_text
                    )
                    <=
                    max_chars
                    and
                    current_duration
                    <=
                    max_cue_duration
                ):
                    break

                break_index = (
                    choose_best_break_index(
                        tokens=(
                            current
                        ),
                        target_chars=(
                            target_chars
                        ),
                    )
                )

                left = current[
                    :break_index
                ]

                current = current[
                    break_index:
                ]

                flush(
                    left
                )

            continue

        # -------------------------------------------------
        # hard limit은 아니지만 문장 종결이면
        # 어느 정도 길이/시간이 확보됐을 때 즉시 종료
        # -------------------------------------------------

        current.append(
            token
        )

        current_text = displayed_text(
            current
        )

        current_duration = (
            float(
                current[
                    -1
                ][
                    "end"
                ]
            )
            -
            float(
                current[
                    0
                ][
                    "start"
                ]
            )
        )

        if (
            has_sentence_end(
                token.get(
                    "text",
                    "",
                )
            )
            and
            len(
                current_text
            )
            >=
            MIN_SEMANTIC_CUE_CHARS
            and
            current_duration
            >=
            MIN_SEMANTIC_CUE_DURATION
        ):
            flush(
                current
            )

            current = []

    if current:
        flush(
            current
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

    duration = cue_duration(
        cue
    )

    word_count = cue_word_count(
        cue
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
# SHORT CUE MERGE
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

    while (
        index
        <
        len(
            cues
        )
    ):

        cue = cues[
            index
        ]

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

        if (
            not can_previous
            and
            not can_next
        ):
            index += 1
            continue

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

        if merge_to_next:

            merged = {
                "start": float(
                    cue[
                        "start"
                    ]
                ),
                "end": float(
                    next_cue[
                        "end"
                    ]
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

            continue

        merged = {
            "start": float(
                previous[
                    "start"
                ]
            ),
            "end": float(
                cue[
                    "end"
                ]
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
# CUE TIMING SANITY
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

        cue = copy.deepcopy(
            cue
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

            previous = results[
                -1
            ]

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
        ] = cue_start

        cue[
            "end"
        ] = cue_end

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
    # 1. GLOBAL WORD TIMELINE
    # =====================================================

    global_words = collect_global_words(
        segments
    )

    # =====================================================
    # 2. CANONICAL TOKENS
    # =====================================================

    (
        canonical_tokens,
        canonical_text,
    ) = build_canonical_tokens(
        segments
    )

    # =====================================================
    # 3. CANONICAL ↔ WORD TIMESTAMP
    # =====================================================

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

    # =====================================================
    # 4. MISSING TIMING INTERPOLATION
    # =====================================================

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
    # 5. SEMANTIC GLOBAL CUE BUILD
    #
    # canonical segment 경계를 무시하고:
    #
    # - 실제 pause
    # - 문장 종결
    # - 연결형 어미
    # - 다음 담화 표지
    # - target chars / duration
    #
    # 를 종합해서 breakpoint를 선택한다.
    # =====================================================

    results = (
        build_semantic_global_cues(
            tokens=(
                canonical_tokens
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

    before_short_merge = len(
        results
    )

    results = sanitize_cues(
        results
    )

    (
        results,
        short_merge_count,
    ) = merge_short_cues(
        cues=(
            results
        ),
        max_chars=(
            max_chars
        ),
    )

    results = sanitize_cues(
        results
    )

    # =====================================================
    # STATS
    # =====================================================

    matched_tokens = sum(
        1
        for token
        in canonical_tokens
        if (
            token[
                "matched_chars"
            ]
            >
            0
        )
    )

    unmatched_tokens = (
        len(
            canonical_tokens
        )
        -
        matched_tokens
    )

    token_match_ratio = (
        matched_tokens
        /
        len(
            canonical_tokens
        )
        if canonical_tokens
        else 0.0
    )

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
        f"  matched tokens     : "
        f"{matched_tokens}"
    )

    print(
        f"  unmatched tokens   : "
        f"{unmatched_tokens}"
    )

    print(
        f"  token match ratio  : "
        f"{token_match_ratio:.1%}"
    )

    print(
        f"  semantic cue build : "
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

    return results
