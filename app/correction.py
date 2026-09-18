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
# CONTEXTUAL HIGH-CONFIDENCE AUTO CORRECTIONS
# =========================================================
#
# These are NOT unconditional replacements.
#
# A variant is automatically corrected only when:
#   1. the suspect variant is present in the segment, and
#   2. the surrounding context contains enough domain signals.
#
# If the context condition is not satisfied, the text is left
# untouched and the existing REVIEW_TERM_GROUPS logic can still
# surface it for manual review.
#
# This keeps confirmed corrections and contextual corrections
# clearly separated.
# =========================================================

CONTEXTUAL_AUTO_CORRECTION_GROUPS = [

    {
        "canonical": "절삭유",

        "variants": [
            "절사규",
            "절사기",
            "절사유",
            "절삭규",
        ],

        "context_keywords": [
            "금속",
            "가공",
            "마찰",
            "열",
            "기름",
            "유증기",
            "바닥",
            "공장",
            "절삭",
        ],

        "min_context_hits": 1,

        "confidence": 0.97,

        "reason": (
            "금속 가공·마찰·열·기름·유증기 등 "
            "절삭유 문맥이 확인된 경우에만 자동 교정."
        ),
    },

    {
        "canonical": "불쏘시개",

        "variants": [
            "불쏘식",
            "불쏘시계",
            "불쏘시게",
        ],

        "context_keywords": [
            "화재",
            "점화",
            "연소",
            "유증기",
            "기름",
            "역할",
            "불",
        ],

        "min_context_hits": 2,

        "confidence": 0.96,

        "reason": (
            "화재·점화·연소·유증기·역할 등 "
            "불쏘시개 문맥이 복수 확인된 경우에만 자동 교정."
        ),
    },
]



# =========================================================
# LEVEL 3
# SAFE SPACING NORMALIZATION
# =========================================================
#
# This is intentionally NOT a general Korean spacing corrector.
#
# We only fix exact, repeatedly observed concatenation patterns
# that are highly unlikely to change meaning.
#
# Important:
# - text only is changed
# - word timestamps are preserved
# - proper nouns / organization names / technical terms are
#   not split by generic heuristics
# - every applied spacing rule is written to correction_log
# =========================================================

SAFE_SPACING_RULES = [

    {
        "from": "라인에대해서",
        "to": "라인에 대해서",
        "reason": (
            "조사 결합 과정에서 붙은 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "시설물에대해서",
        "to": "시설물에 대해서",
        "reason": (
            "조사 결합 과정에서 붙은 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "구조물에대해서",
        "to": "구조물에 대해서",
        "reason": (
            "조사 결합 과정에서 붙은 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "안전에관련된",
        "to": "안전에 관련된",
        "reason": (
            "의존 표현 '관련된' 앞의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "으로인해",
        "to": "으로 인해",
        "reason": (
            "의존 표현 '인해' 앞의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "인해서사실상",
        "to": "인해서 사실상",
        "reason": (
            "문장 내부에서 붙은 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "건축법이나소방법",
        "to": "건축법이나 소방법",
        "reason": (
            "병렬 명사 사이의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "적법하게맞춰주지",
        "to": "적법하게 맞춰주지",
        "reason": (
            "부사와 용언 사이의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "대피가어려웠",
        "to": "대피가 어려웠",
        "reason": (
            "주어와 서술어 사이의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "피해자가대피",
        "to": "피해자가 대피",
        "reason": (
            "주어와 용언 사이의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "살펴볼예정",
        "to": "살펴볼 예정",
        "reason": (
            "관형형과 의존 명사 '예정' 사이의 어절 경계를 복원."
        ),
    },

    {
        "from": "된원인",
        "to": "된 원인",
        "reason": (
            "관형형과 명사 사이의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "여러가지",
        "to": "여러 가지",
        "reason": (
            "고정적으로 확인된 띄어쓰기 형태를 적용."
        ),
    },

    {
        "from": "결합하면서불쏘시개",
        "to": "결합하면서 불쏘시개",
        "reason": (
            "연결 어미 뒤에 붙은 독립 명사의 어절 경계를 복원."
        ),
    },

    {
        "from": "결합하면서불쏘식",
        "to": "결합하면서 불쏘식",
        "reason": (
            "연결 어미 뒤에 붙은 독립 명사의 어절 경계를 복원."
        ),
    },

    {
        "from": "절삭유가흥건",
        "to": "절삭유가 흥건",
        "reason": (
            "주어와 서술어 사이의 명백한 어절 경계를 복원."
        ),
    },

    {
        "from": "절사규가흥건",
        "to": "절사규가 흥건",
        "reason": (
            "주어와 서술어 사이의 명백한 어절 경계를 복원."
        ),
    },
]


# =========================================================
# LEVEL 4
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
# CONTEXTUAL AUTO CORRECTION
# =========================================================

def count_context_keyword_hits(
    context: str,
    keywords: list[str],
) -> tuple[
    int,
    list[str],
]:

    normalized_context = (
        normalize_text(
            context
        )
    )

    matched = []

    for keyword in keywords:

        normalized_keyword = (
            normalize_text(
                keyword
            )
        )

        if not normalized_keyword:
            continue

        if (
            normalized_keyword
            in
            normalized_context
        ):

            matched.append(
                keyword
            )

    return (
        len(
            matched
        ),
        matched,
    )


def apply_contextual_domain_corrections(
    segment: dict,
    context: str,
) -> tuple[
    dict,
    list[dict],
]:
    """
    Apply high-confidence domain corrections only when the
    configured context requirements are satisfied.

    If requirements fail, no replacement occurs. The remaining
    suspect token can then be emitted by REVIEW_TERM_GROUPS.
    """

    item = (
        copy.deepcopy(
            segment
        )
    )

    corrected_text = str(
        item.get(
            "text",
            "",
        )
    ).strip()

    applied_rules = []

    for group in (
        CONTEXTUAL_AUTO_CORRECTION_GROUPS
    ):

        canonical = (
            group[
                "canonical"
            ]
        )

        context_keywords = (
            group.get(
                "context_keywords",
                [],
            )
        )

        min_context_hits = int(
            group.get(
                "min_context_hits",
                1,
            )
        )

        confidence = float(
            group.get(
                "confidence",
                0.95,
            )
        )

        reason = str(
            group.get(
                "reason",
                "",
            )
        )

        (
            context_hit_count,
            matched_keywords,
        ) = (
            count_context_keyword_hits(
                context=(
                    context
                ),
                keywords=(
                    context_keywords
                ),
            )
        )

        if (
            context_hit_count
            <
            min_context_hits
        ):
            continue

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

                    "mode": (
                        "contextual_auto"
                    ),

                    "confidence": (
                        confidence
                    ),

                    "context_hits": (
                        context_hit_count
                    ),

                    "matched_keywords": (
                        copy.deepcopy(
                            matched_keywords
                        )
                    ),
                }
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
# SAFE SPACING NORMALIZATION
# =========================================================

def apply_safe_spacing_normalization(
    segment: dict,
) -> tuple[
    dict,
    list[dict],
]:
    """
    Apply only exact, high-confidence spacing repairs.

    This function deliberately avoids morphology-based or
    language-model-based spacing correction.  It changes only
    segment["text"]; segment word timestamps remain untouched.
    """

    item = (
        copy.deepcopy(
            segment
        )
    )

    corrected_text = str(
        item.get(
            "text",
            "",
        )
    ).strip()

    applied_rules = []

    for rule in (
        SAFE_SPACING_RULES
    ):

        source_text = str(
            rule[
                "from"
            ]
        )

        target_text = str(
            rule[
                "to"
            ]
        )

        if (
            source_text
            not in
            corrected_text
        ):
            continue

        corrected_text = (
            corrected_text.replace(
                source_text,
                target_text,
            )
        )

        applied_rules.append(
            {
                "from": (
                    source_text
                ),

                "to": (
                    target_text
                ),

                "reason": (
                    str(
                        rule.get(
                            "reason",
                            "Safe spacing normalization.",
                        )
                    )
                ),

                "mode": (
                    "safe_spacing"
                ),
            }
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

    rules_by_index = {}

    # =====================================================
    # PASS 1
    # CONFIRMED AUTO CORRECTION
    # =====================================================

    for index, segment in enumerate(
        segments
    ):

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

        if applied_rules:

            rules_by_index[
                index
            ] = (
                copy.deepcopy(
                    applied_rules
                )
            )

    # =====================================================
    # PASS 2
    # CONTEXTUAL HIGH-CONFIDENCE AUTO CORRECTION
    #
    # Context is built from the confirmed-correction result.
    # This means known corrections such as "증축" can help
    # contextual decisions without altering the source timing.
    # =====================================================

    contextual_base = (
        copy.deepcopy(
            corrected_segments
        )
    )

    for index, segment in enumerate(
        contextual_base
    ):

        context = (
            build_context(
                segments=(
                    contextual_base
                ),
                index=(
                    index
                ),
                radius=1,
            )
        )

        (
            contextual_item,
            contextual_rules,
        ) = (
            apply_contextual_domain_corrections(
                segment=(
                    segment
                ),
                context=(
                    context
                ),
            )
        )

        corrected_segments[
            index
        ] = (
            contextual_item
        )

        if contextual_rules:

            rules_by_index.setdefault(
                index,
                [],
            ).extend(
                contextual_rules
            )

    # =====================================================
    # PASS 3
    # SAFE SPACING NORMALIZATION
    #
    # Only exact high-confidence concatenation patterns are
    # repaired here.  Word timestamps are intentionally kept.
    # =====================================================

    for index, segment in enumerate(
        corrected_segments
    ):

        (
            spacing_item,
            spacing_rules,
        ) = (
            apply_safe_spacing_normalization(
                segment
            )
        )

        corrected_segments[
            index
        ] = (
            spacing_item
        )

        if spacing_rules:

            rules_by_index.setdefault(
                index,
                [],
            ).extend(
                spacing_rules
            )

    # =====================================================
    # BUILD CORRECTION LOG
    #
    # One log entry per corrected segment, preserving the
    # original raw_text and grouping all rules that affected it.
    # =====================================================

    for index in sorted(
        rules_by_index
    ):

        corrected_item = (
            corrected_segments[
                index
            ]
        )

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
                    corrected_item.get(
                        "raw_text",
                        str(
                            segments[
                                index
                            ].get(
                                "text",
                                "",
                            )
                        ).strip(),
                    )
                ),

                "corrected_text": (
                    corrected_item[
                        "text"
                    ]
                ),

                "rules": (
                    copy.deepcopy(
                        rules_by_index[
                            index
                        ]
                    )
                ),

                "action": (
                    "auto_corrected"
                ),
            }
        )

    # =====================================================
    # PASS 4
    # REVIEW
    #
    # Any contextual candidate that was NOT safe enough for
    # auto correction remains in the text and is still caught
    # here as a review-only candidate.
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

