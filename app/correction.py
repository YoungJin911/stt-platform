import copy
import re

from difflib import SequenceMatcher


# =========================================================
# CONFIG
# =========================================================

LOW_WORD_CONFIDENCE_THRESHOLD = 0.40

MIN_LOW_CONFIDENCE_WORD_LENGTH = 2

WORD_MATCH_SIMILARITY_THRESHOLD = 0.60


# =========================================================
# LEVEL 1
# CONFIRMED AUTO CORRECTIONS
# =========================================================

CONFIRMED_CORRECTION_GROUPS = [

    {
        "canonical": "가공 라인",
        "variants": [
            "가옥 라인",
        ],
        "reason": (
            "영상 자막 및 원음 확인 완료."
        ),
    },

    {
        "canonical": "대전지방고용노동청",
        "variants": [
            "대전지방고용노동척",
        ],
        "reason": (
            "기관명 확인."
        ),
    },

    {
        "canonical": "천장에 유증기가",
        "variants": [
            "전장에 유증기가",
        ],
        "reason": (
            "원음 및 영상 자막 확인."
        ),
    },

    {
        "canonical": "체류를 하게 돼요",
        "variants": [
            "퇴류를 하게 돼요",
        ],
        "reason": (
            "원음 확인."
        ),
    },

    # -----------------------------------------------------
    # 신규 확정 교정
    # -----------------------------------------------------

    {
        "canonical": "증축",
        "variants": [
            "중축",
        ],
        "reason": (
            "영상 자막 및 원음에서 "
            "'불법 증축'으로 확인됨."
        ),
    },
]


# =========================================================
# LEVEL 2
# REVIEW-ONLY DOMAIN TERMS
# =========================================================

REVIEW_TERM_GROUPS = [

    {
        "canonical": "절삭유",

        "variants": [
            "절사규",
            "절사기",
            "절사유",
            "절삭규",
        ],

        "confidence": (
            0.90
        ),

        "reason": (
            "금속 가공 과정에서 마찰과 열을 줄이는 "
            "유체인 '절삭유'의 ASR 오인식 가능성이 높음."
        ),
    },

    {
        "canonical": "증개축",

        # '중축'은 여기에서 제거.
        # 이번 영상에서 중축은 실제로 증축임이 확인됨.
        "variants": [
            "중계측",
            "중개측",
            "증계측",
        ],

        "confidence": (
            0.88
        ),

        "reason": (
            "건축·소방 문맥에서 "
            "'증개축'의 ASR 오인식 가능성이 있음."
        ),
    },

    {
        "canonical": "불쏘시개",

        "variants": [
            "불쏘식",
            "불쏘시계",
            "불쏘시게",
        ],

        "confidence": (
            0.88
        ),

        "reason": (
            "화재 확대·점화 역할 문맥에서 "
            "'불쏘시개'의 ASR 오인식 가능성이 있음."
        ),
    },
]


# =========================================================
# TEXT NORMALIZATION
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


def contains_korean(
    text: str,
) -> bool:

    return bool(
        re.search(
            r"[가-힣]",
            text,
        )
    )


# =========================================================
# CONTEXT
# =========================================================

def build_context(
    segments: list[dict],
    index: int,
    radius: int = 1,
) -> str:

    start_index = max(
        0,
        index - radius,
    )

    end_index = min(
        len(
            segments
        ),
        index + radius + 1,
    )

    texts = []

    for position in range(
        start_index,
        end_index,
    ):

        text = (
            segments[position]
            .get(
                "text",
                "",
            )
            .strip()
        )

        if text:

            texts.append(
                text
            )

    return " ".join(
        texts
    )


# =========================================================
# CONFIRMED CORRECTION
# =========================================================

def apply_confirmed_corrections(
    segment: dict,
) -> tuple[
    dict,
    list[dict],
]:

    item = (
        copy.deepcopy(
            segment
        )
    )

    raw_text = str(
        item.get(
            "text",
            "",
        )
    ).strip()

    corrected_text = (
        raw_text
    )

    applied_rules = []

    for group in (
        CONFIRMED_CORRECTION_GROUPS
    ):

        canonical = (
            group[
                "canonical"
            ]
        )

        reason = (
            group.get(
                "reason",
                "",
            )
        )

        for variant in (
            group[
                "variants"
            ]
        ):

            if (
                variant
                not in
                corrected_text
            ):
                continue

            corrected_text = (
                corrected_text.replace(
                    variant,
                    canonical,
                )
            )

            applied_rules.append(
                {
                    "from": (
                        variant
                    ),

                    "to": (
                        canonical
                    ),

                    "reason": (
                        reason
                    ),
                }
            )

    item[
        "raw_text"
    ] = (
        raw_text
    )

    item[
        "raw_words"
    ] = (
        copy.deepcopy(
            item.get(
                "words",
                [],
            )
        )
    )

    item[
        "text"
    ] = (
        corrected_text
    )

    return (
        item,
        applied_rules,
    )


# =========================================================
# DOMAIN TERM REVIEW
# =========================================================

def detect_domain_term_candidates(
    segment: dict,
    context: str,
) -> list[dict]:

    text = str(
        segment.get(
            "text",
            "",
        )
    )

    results = []

    for group in (
        REVIEW_TERM_GROUPS
    ):

        canonical = (
            group[
                "canonical"
            ]
        )

        confidence = float(
            group.get(
                "confidence",
                0.80,
            )
        )

        reason = (
            group.get(
                "reason",
                "",
            )
        )

        for variant in (
            group[
                "variants"
            ]
        ):

            if variant not in text:
                continue

            results.append(
                {
                    "type": (
                        "domain_term_candidate"
                    ),

                    "start": (
                        segment[
                            "start"
                        ]
                    ),

                    "end": (
                        segment[
                            "end"
                        ]
                    ),

                    "source": (
                        segment.get(
                            "source",
                            "unknown",
                        )
                    ),

                    "raw_text": (
                        text
                    ),

                    "suspect": (
                        variant
                    ),

                    "candidate": (
                        canonical
                    ),

                    "confidence": (
                        confidence
                    ),

                    "reason": (
                        reason
                    ),

                    "context": (
                        context
                    ),

                    "action": (
                        "review"
                    ),
                }
            )

    return results


# =========================================================
# WORD/CANONICAL MATCH
# =========================================================

def word_exists_in_canonical_text(
    word_text: str,
    canonical_text: str,
) -> bool:

    word_normalized = (
        normalize_text(
            word_text
        )
    )

    canonical_normalized = (
        normalize_text(
            canonical_text
        )
    )

    if not word_normalized:
        return False

    if not canonical_normalized:
        return False

    if (
        word_normalized
        in
        canonical_normalized
    ):

        return True

    word_length = len(
        word_normalized
    )

    if word_length < 2:
        return False

    min_length = max(
        2,
        word_length - 1,
    )

    max_length = min(
        len(
            canonical_normalized
        ),
        word_length + 2,
    )

    best_similarity = 0.0

    for candidate_length in range(
        min_length,
        max_length + 1,
    ):

        for start in range(
            0,
            len(
                canonical_normalized
            )
            -
            candidate_length
            +
            1,
        ):

            candidate = (
                canonical_normalized[
                    start:
                    start + candidate_length
                ]
            )

            similarity = (
                SequenceMatcher(
                    None,
                    word_normalized,
                    candidate,
                )
                .ratio()
            )

            best_similarity = max(
                best_similarity,
                similarity,
            )

            if (
                best_similarity
                >=
                WORD_MATCH_SIMILARITY_THRESHOLD
            ):

                return True

    return False


# =========================================================
# LOW CONFIDENCE
# =========================================================

def detect_low_confidence_words(
    segment: dict,
    context: str,
    threshold: float = (
        LOW_WORD_CONFIDENCE_THRESHOLD
    ),
) -> list[dict]:

    results = []

    canonical_text = str(
        segment.get(
            "text",
            "",
        )
    )

    words = (
        segment.get(
            "words",
            [],
        )
    )

    for word in words:

        probability = (
            word.get(
                "probability"
            )
        )

        if probability is None:
            continue

        try:

            probability = float(
                probability
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        if probability >= threshold:
            continue

        word_text = str(
            word.get(
                "word",
                "",
            )
        ).strip()

        if not word_text:
            continue

        normalized_word = (
            normalize_text(
                word_text
            )
        )

        if (
            len(
                normalized_word
            )
            <
            MIN_LOW_CONFIDENCE_WORD_LENGTH
        ):
            continue

        if not contains_korean(
            normalized_word
        ):
            continue

        if not word_exists_in_canonical_text(
            word_text=(
                word_text
            ),
            canonical_text=(
                canonical_text
            ),
        ):

            continue

        results.append(
            {
                "type": (
                    "low_confidence_word"
                ),

                "start": (
                    word.get(
                        "start",
                        segment[
                            "start"
                        ],
                    )
                ),

                "end": (
                    word.get(
                        "end",
                        segment[
                            "end"
                        ],
                    )
                ),

                "segment_start": (
                    segment[
                        "start"
                    ]
                ),

                "segment_end": (
                    segment[
                        "end"
                    ]
                ),

                "source": (
                    segment.get(
                        "source",
                        "unknown",
                    )
                ),

                "raw_text": (
                    canonical_text
                ),

                "suspect": (
                    word_text
                ),

                "candidate": None,

                "confidence": (
                    probability
                ),

                "reason": (
                    "Canonical transcript와 대응되는 단어이나 "
                    "Whisper alignment probability가 "
                    f"{probability:.3f}로 낮음."
                ),

                "context": (
                    context
                ),

                "action": (
                    "review"
                ),
            }
        )

    return results


# =========================================================
# DEDUP
# =========================================================

def deduplicate_review_candidates(
    candidates: list[dict],
) -> list[dict]:

    result = []

    seen = set()

    for candidate in candidates:

        key = (
            candidate.get(
                "type"
            ),

            round(
                float(
                    candidate.get(
                        "start",
                        0.0,
                    )
                ),
                2,
            ),

            candidate.get(
                "suspect"
            ),

            candidate.get(
                "candidate"
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            candidate
        )

    result.sort(
        key=lambda item: (
            item.get(
                "start",
                0.0,
            ),
            item.get(
                "type",
                "",
            ),
        )
    )

    return result


# =========================================================
# PIPELINE
# =========================================================

def correct_segments(
    segments: list[dict],
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
]:

    corrected_segments = []

    correction_log = []

    review_candidates = []

    # =====================================================
    # PASS 1
    # AUTO CORRECTION
    # =====================================================

    for segment in segments:

        (
            corrected_item,
            applied_rules,
        ) = (
            apply_confirmed_corrections(
                segment
            )
        )

        corrected_segments.append(
            corrected_item
        )

        if not applied_rules:
            continue

        correction_log.append(
            {
                "start": (
                    corrected_item[
                        "start"
                    ]
                ),

                "end": (
                    corrected_item[
                        "end"
                    ]
                ),

                "source": (
                    corrected_item.get(
                        "source",
                        "unknown",
                    )
                ),

                "raw_text": (
                    corrected_item[
                        "raw_text"
                    ]
                ),

                "corrected_text": (
                    corrected_item[
                        "text"
                    ]
                ),

                "rules": (
                    applied_rules
                ),

                "action": (
                    "auto_corrected"
                ),
            }
        )

    # =====================================================
    # PASS 2
    # REVIEW
    # =====================================================

    for index, segment in enumerate(
        corrected_segments
    ):

        context = (
            build_context(
                segments=(
                    corrected_segments
                ),
                index=(
                    index
                ),
                radius=1,
            )
        )

        review_candidates.extend(
            detect_domain_term_candidates(
                segment=(
                    segment
                ),
                context=(
                    context
                ),
            )
        )

        review_candidates.extend(
            detect_low_confidence_words(
                segment=(
                    segment
                ),
                context=(
                    context
                ),
            )
        )

    review_candidates = (
        deduplicate_review_candidates(
            review_candidates
        )
    )

    return (
        corrected_segments,
        correction_log,
        review_candidates,
    )