import copy
import os
import re
import subprocess

from difflib import SequenceMatcher


# =========================================================
# CONFIG
# =========================================================

MIN_CANDIDATE_CHARS = 3

MAX_CANDIDATE_WORDS = 4


# =========================================================
# SECONDARY DIFF FILTER
# =========================================================

#
# Secondary diff 단계에서 canonical과 이미 상당히
# 비슷한 단어/phrase라면 omission으로 보지 않는다.
#
CANONICAL_SIMILARITY_REJECT = 0.78


# =========================================================
# NEW:
# TIME-LOCAL CANONICAL DUPLICATE GUARD
# =========================================================

#
# candidate가 발생한 실제 시간 기준으로
# 앞뒤 canonical segment를 다시 검색한다.
#
LOCAL_CANONICAL_MARGIN = 6.0


#
# exact substring은 무조건 reject.
#
# fuzzy phrase similarity도 아래 이상이면
# "이미 canonical에 있는 문구"로 판단.
#
LOCAL_CANONICAL_FUZZY_REJECT = 0.82


#
# candidate가 1단어일 때 canonical 단어와
# 이 이상 유사하면 이미 있는 것으로 본다.
#
LOCAL_SINGLE_TOKEN_REJECT = 0.84


# =========================================================
# LOCAL AUDIO VERIFICATION
# =========================================================

VERIFY_SIMILARITY = 0.78

MIN_VERIFY_COUNT = 2


# =========================================================
# AUTO APPLY CONFIG
# =========================================================

AUTO_APPLY_MIN_PROBABILITY = 0.90

AUTO_APPLY_REQUIRED_VERIFY_COUNT = 2


#
# candidate가 왼쪽 segment 끝 주변에 있으면
# 새로운 segment를 만들지 않고 왼쪽에 이어 붙인다.
#
APPEND_LEFT_BEFORE_END = 1.00

APPEND_LEFT_AFTER_END = 0.80


# =========================================================
# FINAL DUPLICATE GUARD
# =========================================================

#
# 실제 apply 직전에 다시 한번 주변 canonical을 확인한다.
#
FINAL_DUPLICATE_MARGIN = 4.0

FINAL_DUPLICATE_FUZZY_REJECT = 0.84


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_token(
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


def normalize_text(
    text: str,
) -> str:

    return normalize_token(
        text
    )


def tokenize_text(
    text: str,
) -> list[str]:

    tokens = []

    for piece in str(
        text or ""
    ).split():

        normalized = (
            normalize_token(
                piece
            )
        )

        if normalized:

            tokens.append(
                normalized
            )

    return tokens


def text_similarity(
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
# CANONICAL CONTEXT
# =========================================================

def get_canonical_context(
    segments: list[dict],
    indices: list[int],
) -> tuple[
    str,
    list[str],
]:

    texts = []

    for index in indices:

        if (
            index < 0
            or
            index >= len(
                segments
            )
        ):
            continue

        text = str(
            segments[index].get(
                "text",
                "",
            )
        ).strip()

        if text:

            texts.append(
                text
            )

    combined = (
        " ".join(
            texts
        )
    )

    return (
        combined,
        tokenize_text(
            combined
        ),
    )


# =========================================================
# NEW:
# GET CANONICAL TEXT AROUND ACTUAL TIMESTAMP
# =========================================================

def get_local_canonical_context(
    segments: list[dict],
    start: float,
    end: float,
    margin: float = LOCAL_CANONICAL_MARGIN,
) -> tuple[
    str,
    list[dict],
]:

    window_start = (
        float(
            start
        )
        -
        margin
    )

    window_end = (
        float(
            end
        )
        +
        margin
    )

    matched_segments = []

    texts = []

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
            window_start
        ):
            continue

        if (
            segment_start
            >
            window_end
        ):
            continue

        text = str(
            segment.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        texts.append(
            text
        )

        matched_segments.append(
            {
                "index": (
                    index
                ),

                "start": (
                    segment_start
                ),

                "end": (
                    segment_end
                ),

                "text": (
                    text
                ),
            }
        )

    return (
        " ".join(
            texts
        ),
        matched_segments,
    )


# =========================================================
# PHRASE MATCH AGAINST CANONICAL
# =========================================================

def phrase_exists_fuzzily(
    candidate_text: str,
    canonical_text: str,
    *,
    phrase_threshold: float,
    single_token_threshold: float,
) -> tuple[
    bool,
    float,
    str,
]:

    candidate_normalized = (
        normalize_text(
            candidate_text
        )
    )

    canonical_normalized = (
        normalize_text(
            canonical_text
        )
    )

    if not candidate_normalized:

        return (
            True,
            1.0,
            "empty_candidate",
        )

    if not canonical_normalized:

        return (
            False,
            0.0,
            "",
        )

    # =====================================================
    # EXACT SUBSTRING
    # =====================================================

    if (
        candidate_normalized
        in
        canonical_normalized
    ):

        return (
            True,
            1.0,
            "exact_substring",
        )

    candidate_tokens = (
        tokenize_text(
            candidate_text
        )
    )

    canonical_tokens = (
        tokenize_text(
            canonical_text
        )
    )

    if not candidate_tokens:

        return (
            False,
            0.0,
            "",
        )

    # =====================================================
    # SINGLE TOKEN
    # =====================================================

    if len(
        candidate_tokens
    ) == 1:

        candidate = (
            candidate_tokens[
                0
            ]
        )

        best_similarity = (
            0.0
        )

        best_phrase = ""

        for canonical in (
            canonical_tokens
        ):

            similarity = (
                SequenceMatcher(
                    None,
                    candidate,
                    canonical,
                ).ratio()
            )

            if (
                similarity
                >
                best_similarity
            ):

                best_similarity = (
                    similarity
                )

                best_phrase = (
                    canonical
                )

        if (
            best_similarity
            >=
            single_token_threshold
        ):

            return (
                True,
                best_similarity,
                best_phrase,
            )

        return (
            False,
            best_similarity,
            best_phrase,
        )

    # =====================================================
    # MULTI TOKEN
    # =====================================================

    candidate_count = len(
        candidate_tokens
    )

    candidate_joined = "".join(
        candidate_tokens
    )

    best_similarity = (
        0.0
    )

    best_phrase = ""

    #
    # candidate보다 1~2단어 짧거나 긴 phrase도 비교
    #
    minimum_size = max(
        1,
        candidate_count - 2,
    )

    maximum_size = (
        candidate_count + 2
    )

    for start_index in range(
        len(
            canonical_tokens
        )
    ):

        for size in range(
            minimum_size,
            maximum_size + 1,
        ):

            end_index = (
                start_index
                +
                size
            )

            if (
                end_index
                >
                len(
                    canonical_tokens
                )
            ):

                break

            phrase_tokens = (
                canonical_tokens[
                    start_index:
                    end_index
                ]
            )

            phrase = "".join(
                phrase_tokens
            )

            similarity = (
                SequenceMatcher(
                    None,
                    candidate_joined,
                    phrase,
                ).ratio()
            )

            if (
                similarity
                >
                best_similarity
            ):

                best_similarity = (
                    similarity
                )

                best_phrase = (
                    " ".join(
                        phrase_tokens
                    )
                )

    if (
        best_similarity
        >=
        phrase_threshold
    ):

        return (
            True,
            best_similarity,
            best_phrase,
        )

    return (
        False,
        best_similarity,
        best_phrase,
    )


# =========================================================
# SECONDARY WORD TOKENS
# =========================================================

def secondary_word_tokens(
    words: list[dict],
) -> tuple[
    list[str],
    list[dict],
]:

    tokens = []

    token_words = []

    for word in words:

        text = str(
            word.get(
                "word",
                "",
            )
        ).strip()

        normalized = (
            normalize_token(
                text
            )
        )

        if not normalized:
            continue

        tokens.append(
            normalized
        )

        token_words.append(
            word
        )

    return (
        tokens,
        token_words,
    )


# =========================================================
# ORIGINAL CONTEXT FALSE-POSITIVE FILTER
# =========================================================

def candidate_already_in_canonical(
    candidate_text: str,
    canonical_text: str,
    canonical_tokens: list[str],
) -> bool:

    candidate_normalized = (
        normalize_text(
            candidate_text
        )
    )

    canonical_normalized = (
        normalize_text(
            canonical_text
        )
    )

    if not candidate_normalized:
        return True

    if (
        candidate_normalized
        in
        canonical_normalized
    ):

        return True

    candidate_tokens = (
        tokenize_text(
            candidate_text
        )
    )

    # =====================================================
    # SINGLE TOKEN
    # =====================================================

    if len(
        candidate_tokens
    ) == 1:

        candidate = (
            candidate_tokens[
                0
            ]
        )

        for canonical in (
            canonical_tokens
        ):

            similarity = (
                SequenceMatcher(
                    None,
                    candidate,
                    canonical,
                ).ratio()
            )

            if (
                similarity
                >=
                CANONICAL_SIMILARITY_REJECT
            ):

                return True

    # =====================================================
    # MULTI TOKEN
    # =====================================================

    candidate_count = len(
        candidate_tokens
    )

    if candidate_count >= 2:

        for start in range(
            len(
                canonical_tokens
            )
        ):

            for size in range(
                max(
                    1,
                    candidate_count - 1,
                ),
                min(
                    len(
                        canonical_tokens
                    )
                    -
                    start,
                    candidate_count + 1,
                )
                + 1,
            ):

                canonical_phrase = "".join(
                    canonical_tokens[
                        start:
                        start + size
                    ]
                )

                similarity = (
                    SequenceMatcher(
                        None,
                        candidate_normalized,
                        canonical_phrase,
                    ).ratio()
                )

                if (
                    similarity
                    >=
                    CANONICAL_SIMILARITY_REJECT
                ):

                    return True

    return False


# =========================================================
# RAW DISCREPANCY DETECTION
# =========================================================

def detect_secondary_insertions(
    segments: list[dict],
    secondary_passes: list[dict],
) -> list[dict]:

    candidates = []

    for secondary_pass in (
        secondary_passes
    ):

        (
            canonical_text,
            canonical_tokens,
        ) = (
            get_canonical_context(
                segments=(
                    segments
                ),

                indices=(
                    secondary_pass[
                        "context_segment_indices"
                    ]
                ),
            )
        )

        (
            secondary_tokens,
            secondary_words,
        ) = (
            secondary_word_tokens(
                secondary_pass[
                    "words"
                ]
            )
        )

        if not canonical_tokens:
            continue

        if not secondary_tokens:
            continue

        matcher = (
            SequenceMatcher(
                None,
                canonical_tokens,
                secondary_tokens,
                autojunk=False,
            )
        )

        for (
            tag,
            i1,
            i2,
            j1,
            j2,
        ) in (
            matcher.get_opcodes()
        ):

            if tag != "insert":
                continue

            inserted_count = (
                j2
                -
                j1
            )

            if inserted_count <= 0:
                continue

            if (
                inserted_count
                >
                MAX_CANDIDATE_WORDS
            ):
                continue

            inserted_words = (
                secondary_words[
                    j1:
                    j2
                ]
            )

            if not inserted_words:
                continue

            candidate_text = (
                " ".join(
                    str(
                        word.get(
                            "word",
                            "",
                        )
                    ).strip()
                    for word
                    in inserted_words
                ).strip()
            )

            if not candidate_text:
                continue

            normalized = (
                normalize_text(
                    candidate_text
                )
            )

            if (
                len(
                    normalized
                )
                <
                MIN_CANDIDATE_CHARS
            ):

                continue

            # =================================================
            # ORIGINAL CONTEXT CHECK
            # =================================================

            if candidate_already_in_canonical(
                candidate_text=(
                    candidate_text
                ),

                canonical_text=(
                    canonical_text
                ),

                canonical_tokens=(
                    canonical_tokens
                ),
            ):

                continue

            start = float(
                inserted_words[
                    0
                ][
                    "start"
                ]
            )

            end = float(
                inserted_words[
                    -1
                ][
                    "end"
                ]
            )

            # =================================================
            # NEW:
            # ACTUAL TIME-LOCAL CANONICAL CHECK
            #
            # batch context가 잘렸더라도
            # candidate 실제 시간 앞뒤 전체 canonical을 다시 본다.
            # =================================================

            (
                local_canonical_text,
                local_segments,
            ) = (
                get_local_canonical_context(
                    segments=(
                        segments
                    ),

                    start=(
                        start
                    ),

                    end=(
                        end
                    ),

                    margin=(
                        LOCAL_CANONICAL_MARGIN
                    ),
                )
            )

            (
                exists_locally,
                local_similarity,
                local_match,
            ) = (
                phrase_exists_fuzzily(
                    candidate_text=(
                        candidate_text
                    ),

                    canonical_text=(
                        local_canonical_text
                    ),

                    phrase_threshold=(
                        LOCAL_CANONICAL_FUZZY_REJECT
                    ),

                    single_token_threshold=(
                        LOCAL_SINGLE_TOKEN_REJECT
                    ),
                )
            )

            if exists_locally:

                print()

                print(
                    "[OMISSION PRE-FILTER]"
                )

                print(
                    f"  candidate : "
                    f"{candidate_text}"
                )

                print(
                    f"  time      : "
                    f"{start:.2f} ~ {end:.2f}"
                )

                print(
                    "  result    : REJECT "
                    "(already exists near timestamp)"
                )

                print(
                    f"  similarity: "
                    f"{local_similarity:.3f}"
                )

                if local_match:

                    print(
                        f"  matched   : "
                        f"{local_match}"
                    )

                continue

            probabilities = [
                float(
                    word[
                        "probability"
                    ]
                )
                for word in inserted_words
                if (
                    word.get(
                        "probability"
                    )
                    is not None
                )
            ]

            average_probability = (
                sum(
                    probabilities
                )
                /
                len(
                    probabilities
                )
                if probabilities
                else None
            )

            candidates.append(
                {
                    "type": (
                        "secondary_insertion"
                    ),

                    "start": (
                        start
                    ),

                    "end": (
                        end
                    ),

                    "text": (
                        candidate_text
                    ),

                    "secondary_probability": (
                        average_probability
                    ),

                    "candidate_words": (
                        copy.deepcopy(
                            inserted_words
                        )
                    ),

                    "batch_number": (
                        secondary_pass[
                            "batch_number"
                        ]
                    ),

                    "batch_start": (
                        secondary_pass[
                            "start"
                        ]
                    ),

                    "batch_end": (
                        secondary_pass[
                            "end"
                        ]
                    ),

                    "canonical_context": (
                        canonical_text
                    ),

                    "local_canonical_context": (
                        local_canonical_text
                    ),

                    "local_canonical_segments": (
                        local_segments
                    ),

                    "local_duplicate_similarity": (
                        local_similarity
                    ),

                    "local_duplicate_match": (
                        local_match
                    ),
                }
            )

    return candidates


# =========================================================
# DEDUP CANDIDATES
# =========================================================

def deduplicate_candidates(
    candidates: list[dict],
) -> list[dict]:

    ordered = sorted(
        candidates,
        key=lambda item: (
            item[
                "start"
            ],
            item[
                "end"
            ],
        )
    )

    results = []

    for candidate in (
        ordered
    ):

        duplicate = False

        for existing in (
            results
        ):

            time_distance = abs(
                float(
                    candidate[
                        "start"
                    ]
                )
                -
                float(
                    existing[
                        "start"
                    ]
                )
            )

            similarity = (
                text_similarity(
                    candidate[
                        "text"
                    ],
                    existing[
                        "text"
                    ],
                )
            )

            if (
                time_distance
                <=
                1.0
                and
                similarity
                >=
                0.75
            ):

                candidate_probability = (
                    candidate.get(
                        "secondary_probability"
                    )
                )

                existing_probability = (
                    existing.get(
                        "secondary_probability"
                    )
                )

                if (
                    candidate_probability
                    is not None
                    and
                    (
                        existing_probability
                        is None
                        or
                        candidate_probability
                        >
                        existing_probability
                    )
                ):

                    existing.update(
                        candidate
                    )

                duplicate = True

                break

        if not duplicate:

            results.append(
                dict(
                    candidate
                )
            )

    return results


# =========================================================
# AUDIO EXTRACTION
# =========================================================

def extract_verify_audio(
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
# AUDIO VERIFY TEXT MATCH
# =========================================================

def candidate_exists_in_transcript(
    candidate_text: str,
    transcript_text: str,
) -> bool:

    candidate_normalized = (
        normalize_text(
            candidate_text
        )
    )

    transcript_normalized = (
        normalize_text(
            transcript_text
        )
    )

    if not candidate_normalized:
        return False

    if not transcript_normalized:
        return False

    if (
        candidate_normalized
        in
        transcript_normalized
    ):

        return True

    candidate_tokens = (
        tokenize_text(
            candidate_text
        )
    )

    transcript_tokens = (
        tokenize_text(
            transcript_text
        )
    )

    # =====================================================
    # SINGLE TOKEN
    # =====================================================

    if len(
        candidate_tokens
    ) == 1:

        candidate = (
            candidate_tokens[
                0
            ]
        )

        for token in (
            transcript_tokens
        ):

            similarity = (
                SequenceMatcher(
                    None,
                    candidate,
                    token,
                ).ratio()
            )

            if (
                similarity
                >=
                VERIFY_SIMILARITY
            ):

                return True

        return False

    # =====================================================
    # MULTI TOKEN
    # =====================================================

    candidate_count = len(
        candidate_tokens
    )

    for start in range(
        len(
            transcript_tokens
        )
    ):

        for size in range(
            max(
                1,
                candidate_count - 1,
            ),
            min(
                len(
                    transcript_tokens
                )
                -
                start,
                candidate_count + 1,
            )
            + 1,
        ):

            phrase = "".join(
                transcript_tokens[
                    start:
                    start + size
                ]
            )

            similarity = (
                SequenceMatcher(
                    None,
                    candidate_normalized,
                    phrase,
                ).ratio()
            )

            if (
                similarity
                >=
                VERIFY_SIMILARITY
            ):

                return True

    return False


# =========================================================
# LOCAL AUDIO VERIFICATION
# =========================================================

def verify_candidate(
    candidate: dict,
    normalized_audio: str,
    whisper_engine,
    verify_dir: str,
    media_duration: float,
    candidate_index: int,
) -> dict:

    candidate_start = float(
        candidate[
            "start"
        ]
    )

    candidate_end = float(
        candidate[
            "end"
        ]
    )

    windows = [
        {
            "name": (
                "center"
            ),

            "start": max(
                0.0,
                candidate_start
                -
                2.5,
            ),

            "end": min(
                media_duration,
                candidate_end
                +
                2.5,
            ),
        },

        {
            "name": (
                "shifted"
            ),

            "start": max(
                0.0,
                candidate_start
                -
                4.0,
            ),

            "end": min(
                media_duration,
                candidate_end
                +
                1.5,
            ),
        },
    ]

    verify_count = 0

    verify_results = []

    for window_index, window in enumerate(
        windows,
        start=1,
    ):

        output_file = os.path.join(
            verify_dir,
            (
                f"candidate_"
                f"{candidate_index:04d}_"
                f"{window_index}_"
                f"{window['name']}.wav"
            ),
        )

        extract_verify_audio(
            input_file=(
                normalized_audio
            ),

            output_file=(
                output_file
            ),

            start=(
                window[
                    "start"
                ]
            ),

            end=(
                window[
                    "end"
                ]
            ),
        )

        clip_duration = (
            window[
                "end"
            ]
            -
            window[
                "start"
            ]
        )

        verify_segments = (
            whisper_engine.transcribe_file(
                audio_file=(
                    output_file
                ),

                global_offset=(
                    window[
                        "start"
                    ]
                ),

                source=(
                    "omission_verify"
                ),

                audio_duration=(
                    clip_duration
                ),

                strict_timestamp=True,
            )
        )

        transcript_text = " ".join(
            str(
                segment.get(
                    "text",
                    "",
                )
            ).strip()
            for segment in (
                verify_segments
            )
            if str(
                segment.get(
                    "text",
                    "",
                )
            ).strip()
        )

        matched = (
            candidate_exists_in_transcript(
                candidate_text=(
                    candidate[
                        "text"
                    ]
                ),

                transcript_text=(
                    transcript_text
                ),
            )
        )

        if matched:

            verify_count += 1

        verify_results.append(
            {
                "name": (
                    window[
                        "name"
                    ]
                ),

                "start": (
                    window[
                        "start"
                    ]
                ),

                "end": (
                    window[
                        "end"
                    ]
                ),

                "matched": (
                    matched
                ),

                "transcript": (
                    transcript_text
                ),
            }
        )

    result = dict(
        candidate
    )

    result[
        "verification_count"
    ] = (
        verify_count
    )

    result[
        "verification_total"
    ] = (
        len(
            windows
        )
    )

    result[
        "verification"
    ] = (
        verify_results
    )

    result[
        "verified"
    ] = (
        verify_count
        >=
        MIN_VERIFY_COUNT
    )

    return (
        result
    )


# =========================================================
# DETECTOR
# =========================================================

def detect_omission_candidates_v2(
    segments: list[dict],
    secondary_passes: list[dict],
    normalized_audio: str,
    whisper_engine,
    verify_dir: str,
    media_duration: float,
) -> list[dict]:

    os.makedirs(
        verify_dir,
        exist_ok=True,
    )

    print()
    print(
        "Starting omission detection v2..."
    )

    raw_candidates = (
        detect_secondary_insertions(
            segments=(
                segments
            ),

            secondary_passes=(
                secondary_passes
            ),
        )
    )

    deduplicated = (
        deduplicate_candidates(
            raw_candidates
        )
    )

    print(
        f"  secondary passes   : "
        f"{len(secondary_passes)}"
    )

    print(
        f"  raw discrepancies  : "
        f"{len(raw_candidates)}"
    )

    print(
        f"  deduplicated       : "
        f"{len(deduplicated)}"
    )

    results = []

    for index, candidate in enumerate(
        deduplicated,
        start=1,
    ):

        print()

        print(
            f"[OMISSION V2 #{index}] "
            f"{candidate['start']:.2f} "
            f"~ "
            f"{candidate['end']:.2f}"
        )

        print(
            f"  secondary candidate: "
            f"{candidate['text']}"
        )

        probability = (
            candidate.get(
                "secondary_probability"
            )
        )

        if probability is not None:

            print(
                f"  probability        : "
                f"{probability:.3f}"
            )

        verified = (
            verify_candidate(
                candidate=(
                    candidate
                ),

                normalized_audio=(
                    normalized_audio
                ),

                whisper_engine=(
                    whisper_engine
                ),

                verify_dir=(
                    verify_dir
                ),

                media_duration=(
                    media_duration
                ),

                candidate_index=(
                    index
                ),
            )
        )

        for verification in (
            verified[
                "verification"
            ]
        ):

            print(
                f"  [{verification['name']}] "
                f"matched="
                f"{verification['matched']}"
            )

            print(
                f"    "
                f"{verification['transcript']}"
            )

        print(
            f"  verification       : "
            f"{verified['verification_count']}"
            f"/"
            f"{verified['verification_total']}"
        )

        if not verified[
            "verified"
        ]:

            print(
                "  -> rejected"
            )

            continue

        verified[
            "type"
        ] = (
            "omission_candidate"
        )

        verified[
            "action"
        ] = (
            "review"
        )

        results.append(
            verified
        )

        print(
            "  -> VERIFIED omission candidate"
        )

    print()

    print(
        "Omission detector v2:"
    )

    print(
        f"  reviewed candidates : "
        f"{len(deduplicated)}"
    )

    print(
        f"  verified candidates : "
        f"{len(results)}"
    )

    return (
        results
    )


# =========================================================
# FINAL DUPLICATE CHECK
# =========================================================

def text_exists_near_candidate(
    segments: list[dict],
    candidate: dict,
    margin: float = FINAL_DUPLICATE_MARGIN,
) -> tuple[
    bool,
    float,
    str,
]:

    candidate_text = str(
        candidate.get(
            "text",
            "",
        )
    ).strip()

    if not candidate_text:

        return (
            True,
            1.0,
            "empty_candidate",
        )

    candidate_start = float(
        candidate[
            "start"
        ]
    )

    candidate_end = float(
        candidate[
            "end"
        ]
    )

    (
        nearby_text,
        _nearby_segments,
    ) = (
        get_local_canonical_context(
            segments=(
                segments
            ),

            start=(
                candidate_start
            ),

            end=(
                candidate_end
            ),

            margin=(
                margin
            ),
        )
    )

    return (
        phrase_exists_fuzzily(
            candidate_text=(
                candidate_text
            ),

            canonical_text=(
                nearby_text
            ),

            phrase_threshold=(
                FINAL_DUPLICATE_FUZZY_REJECT
            ),

            single_token_threshold=(
                FINAL_DUPLICATE_FUZZY_REJECT
            ),
        )
    )


# =========================================================
# WORD MERGE
# =========================================================

def merge_words(
    old_words: list[dict],
    candidate_words: list[dict],
) -> list[dict]:

    merged_words = (
        copy.deepcopy(
            old_words
        )
        +
        copy.deepcopy(
            candidate_words
        )
    )

    merged_words.sort(
        key=lambda word: (
            float(
                word.get(
                    "start",
                    0.0,
                )
            ),

            float(
                word.get(
                    "end",
                    0.0,
                )
            ),
        )
    )

    deduplicated_words = []

    for word in (
        merged_words
    ):

        duplicate = False

        word_text = (
            normalize_text(
                word.get(
                    "word",
                    "",
                )
            )
        )

        word_start = float(
            word.get(
                "start",
                0.0,
            )
        )

        for existing in (
            deduplicated_words
        ):

            existing_text = (
                normalize_text(
                    existing.get(
                        "word",
                        "",
                    )
                )
            )

            existing_start = float(
                existing.get(
                    "start",
                    0.0,
                )
            )

            if (
                word_text
                ==
                existing_text
                and
                abs(
                    word_start
                    -
                    existing_start
                )
                <=
                0.20
            ):

                duplicate = True

                break

        if not duplicate:

            deduplicated_words.append(
                word
            )

    return (
        deduplicated_words
    )


# =========================================================
# AUTO APPLY VERIFIED OMISSIONS
# =========================================================

def apply_verified_omissions(
    segments: list[dict],
    candidates: list[dict],
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
]:

    result_segments = (
        copy.deepcopy(
            segments
        )
    )

    applied_log = []

    pending_review = []

    candidates = sorted(
        candidates,
        key=lambda item: (
            item[
                "start"
            ],
            item[
                "end"
            ],
        )
    )

    for candidate in (
        candidates
    ):

        candidate_text = str(
            candidate.get(
                "text",
                "",
            )
        ).strip()

        if not candidate_text:
            continue

        probability = (
            candidate.get(
                "secondary_probability"
            )
        )

        verify_count = int(
            candidate.get(
                "verification_count",
                0,
            )
        )

        verify_total = int(
            candidate.get(
                "verification_total",
                0,
            )
        )

        # =================================================
        # AUTO APPLY SAFETY
        # =================================================

        if (
            probability is None
            or
            float(
                probability
            )
            <
            AUTO_APPLY_MIN_PROBABILITY
        ):

            pending_review.append(
                candidate
            )

            continue

        if (
            verify_count
            <
            AUTO_APPLY_REQUIRED_VERIFY_COUNT
        ):

            pending_review.append(
                candidate
            )

            continue

        if (
            verify_count
            !=
            verify_total
        ):

            pending_review.append(
                candidate
            )

            continue

        # =================================================
        # FINAL DUPLICATE PREVENTION
        # =================================================

        (
            duplicate,
            duplicate_similarity,
            duplicate_match,
        ) = (
            text_exists_near_candidate(
                segments=(
                    result_segments
                ),

                candidate=(
                    candidate
                ),
            )
        )

        if duplicate:

            print()

            print(
                "[OMISSION APPLY GUARD]"
            )

            print(
                f"  candidate : "
                f"{candidate_text}"
            )

            print(
                "  result    : SKIP "
                "(already exists)"
            )

            print(
                f"  similarity: "
                f"{duplicate_similarity:.3f}"
            )

            if duplicate_match:

                print(
                    f"  matched   : "
                    f"{duplicate_match}"
                )

            continue

        candidate_start = float(
            candidate[
                "start"
            ]
        )

        candidate_end = float(
            candidate[
                "end"
            ]
        )

        candidate_words = (
            copy.deepcopy(
                candidate.get(
                    "candidate_words",
                    [],
                )
            )
        )

        # =================================================
        # FIND LEFT SEGMENT
        # =================================================

        left_index = None

        for index, segment in enumerate(
            result_segments
        ):

            segment_start = float(
                segment[
                    "start"
                ]
            )

            if (
                segment_start
                <=
                candidate_start
            ):

                left_index = (
                    index
                )

            else:

                break

        # =================================================
        # APPEND TO LEFT
        # =================================================

        if left_index is not None:

            left = (
                result_segments[
                    left_index
                ]
            )

            left_end = float(
                left[
                    "end"
                ]
            )

            distance = (
                candidate_start
                -
                left_end
            )

            if (
                -
                APPEND_LEFT_BEFORE_END
                <=
                distance
                <=
                APPEND_LEFT_AFTER_END
            ):

                old_text = str(
                    left.get(
                        "text",
                        "",
                    )
                ).strip()

                new_text = (
                    old_text
                    +
                    " "
                    +
                    candidate_text
                ).strip()

                old_words = (
                    copy.deepcopy(
                        left.get(
                            "words",
                            [],
                        )
                    )
                )

                left[
                    "pre_omission_text"
                ] = (
                    old_text
                )

                left[
                    "pre_omission_words"
                ] = (
                    copy.deepcopy(
                        old_words
                    )
                )

                left[
                    "text"
                ] = (
                    new_text
                )

                left[
                    "words"
                ] = (
                    merge_words(
                        old_words=(
                            old_words
                        ),

                        candidate_words=(
                            candidate_words
                        ),
                    )
                )

                left[
                    "omission_applied"
                ] = (
                    True
                )

                left[
                    "omission_source"
                ] = (
                    "verified_secondary"
                )

                left[
                    "omission_inserted_text"
                ] = (
                    candidate_text
                )

                left[
                    "omission_candidate_words"
                ] = (
                    copy.deepcopy(
                        candidate_words
                    )
                )

                left[
                    "omission_candidate_start"
                ] = (
                    candidate_start
                )

                left[
                    "omission_candidate_end"
                ] = (
                    candidate_end
                )

                left[
                    "alignment_source"
                ] = (
                    "verified_secondary_merged"
                )

                applied_log.append(
                    {
                        "mode": (
                            "append_left"
                        ),

                        "segment_start": (
                            left[
                                "start"
                            ]
                        ),

                        "segment_end": (
                            left[
                                "end"
                            ]
                        ),

                        "candidate_start": (
                            candidate_start
                        ),

                        "candidate_end": (
                            candidate_end
                        ),

                        "inserted_text": (
                            candidate_text
                        ),

                        "before": (
                            old_text
                        ),

                        "after": (
                            new_text
                        ),

                        "probability": (
                            probability
                        ),

                        "verification": (
                            f"{verify_count}/{verify_total}"
                        ),

                        "candidate_word_count": (
                            len(
                                candidate_words
                            )
                        ),
                    }
                )

                continue

        # =================================================
        # CREATE NEW SEGMENT
        # =================================================

        insert_index = len(
            result_segments
        )

        for index, segment in enumerate(
            result_segments
        ):

            if (
                float(
                    segment[
                        "start"
                    ]
                )
                >
                candidate_start
            ):

                insert_index = (
                    index
                )

                break

        new_segment = {
            "start": (
                candidate_start
            ),

            "end": (
                candidate_end
            ),

            "text": (
                candidate_text
            ),

            "source": (
                "recovery_omission"
            ),

            "words": (
                copy.deepcopy(
                    candidate_words
                )
            ),

            "omission_applied": (
                True
            ),

            "omission_source": (
                "verified_secondary"
            ),

            "omission_inserted_text": (
                candidate_text
            ),

            "omission_candidate_words": (
                copy.deepcopy(
                    candidate_words
                )
            ),

            "omission_candidate_start": (
                candidate_start
            ),

            "omission_candidate_end": (
                candidate_end
            ),

            "alignment_source": (
                "verified_secondary"
            ),
        }

        result_segments.insert(
            insert_index,
            new_segment,
        )

        applied_log.append(
            {
                "mode": (
                    "new_segment"
                ),

                "segment_start": (
                    candidate_start
                ),

                "segment_end": (
                    candidate_end
                ),

                "candidate_start": (
                    candidate_start
                ),

                "candidate_end": (
                    candidate_end
                ),

                "inserted_text": (
                    candidate_text
                ),

                "before": "",

                "after": (
                    candidate_text
                ),

                "probability": (
                    probability
                ),

                "verification": (
                    f"{verify_count}/{verify_total}"
                ),

                "candidate_word_count": (
                    len(
                        candidate_words
                    )
                ),
            }
        )

    result_segments.sort(
        key=lambda item: (
            item[
                "start"
            ],
            item[
                "end"
            ],
        )
    )

    return (
        result_segments,
        applied_log,
        pending_review,
    )