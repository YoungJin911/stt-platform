import copy
import os
import re
import subprocess

from difflib import SequenceMatcher


# =========================================================
# CONFIG
# =========================================================

DEFAULT_CONTEXT = 0.60

WIDE_CONTEXT = 4.00

MIN_REALIGN_SIMILARITY = 0.45

CANDIDATE_MATCH_SIMILARITY = 0.78


# =========================================================
# TEXT HELPERS
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
        for word in words
        if str(
            word.get(
                "word",
                "",
            )
        ).strip()
    )


def text_similarity(
    left: str,
    right: str,
) -> float:

    left = normalize_text(
        left
    )

    right = normalize_text(
        right
    )

    if not left:
        return 0.0

    if not right:
        return 0.0

    return SequenceMatcher(
        None,
        left,
        right,
    ).ratio()


def phrase_exists(
    phrase: str,
    transcript: str,
) -> bool:

    phrase_normalized = (
        normalize_text(
            phrase
        )
    )

    transcript_normalized = (
        normalize_text(
            transcript
        )
    )

    if not phrase_normalized:
        return False

    if not transcript_normalized:
        return False

    if (
        phrase_normalized
        in
        transcript_normalized
    ):

        return True

    similarity = (
        SequenceMatcher(
            None,
            phrase_normalized,
            transcript_normalized,
        ).ratio()
    )

    return (
        similarity
        >=
        CANDIDATE_MATCH_SIMILARITY
    )


# =========================================================
# INSERTED TEXT
# =========================================================

def get_inserted_text(
    segment: dict,
) -> str:

    explicit = str(
        segment.get(
            "omission_inserted_text",
            "",
        )
    ).strip()

    if explicit:

        return explicit

    before = str(
        segment.get(
            "pre_omission_text",
            "",
        )
    ).strip()

    current = str(
        segment.get(
            "text",
            "",
        )
    ).strip()

    if before and current.startswith(
        before
    ):

        suffix = (
            current[
                len(
                    before
                ):
            ]
            .strip()
        )

        if suffix:
            return suffix

    if (
        segment.get(
            "source"
        )
        ==
        "recovery_omission"
    ):

        return current

    return ""


# =========================================================
# AUDIO
# =========================================================

def extract_audio(
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

    subprocess.run(
        [
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
        ],
        check=True,
    )


# =========================================================
# WORD ASSIGNMENT
# =========================================================

def assign_words_to_segment(
    words: list[dict],
    segment_start: float,
    segment_end: float,
) -> list[dict]:

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

        duration = max(
            0.001,
            word_end
            -
            word_start,
        )

        if (
            overlap
            /
            duration
            >= 0.50
        ):

            assigned.append(
                copy.deepcopy(
                    word
                )
            )

    return assigned


# =========================================================
# ALIGNMENT ATTEMPT
# =========================================================

def run_alignment_attempt(
    *,
    segment: dict,
    normalized_audio: str,
    whisper_engine,
    output_file: str,
    media_duration: float,
    context: float,
) -> dict:

    segment_start = float(
        segment["start"]
    )

    segment_end = float(
        segment["end"]
    )

    window_start = max(
        0.0,
        segment_start
        -
        context,
    )

    window_end = min(
        media_duration,
        segment_end
        +
        context,
    )

    extract_audio(
        input_file=(
            normalized_audio
        ),
        output_file=(
            output_file
        ),
        start=(
            window_start
        ),
        end=(
            window_end
        ),
    )

    words = (
        whisper_engine
        .transcribe_words(
            audio_file=(
                output_file
            ),
            global_offset=(
                window_start
            ),
            audio_duration=(
                window_end
                -
                window_start
            ),
        )
    )

    assigned_words = (
        assign_words_to_segment(
            words=(
                words
            ),
            segment_start=(
                segment_start
            ),
            segment_end=(
                segment_end
            ),
        )
    )

    detected_text = (
        words_to_text(
            assigned_words
        )
    )

    similarity = (
        text_similarity(
            segment.get(
                "text",
                "",
            ),
            detected_text,
        )
    )

    inserted_text = (
        get_inserted_text(
            segment
        )
    )

    candidate_present = (
        phrase_exists(
            inserted_text,
            detected_text,
        )
        if inserted_text
        else True
    )

    return {
        "context": (
            context
        ),

        "window_start": (
            window_start
        ),

        "window_end": (
            window_end
        ),

        "words": (
            assigned_words
        ),

        "detected_text": (
            detected_text
        ),

        "similarity": (
            similarity
        ),

        "candidate_present": (
            candidate_present
        ),
    }


# =========================================================
# MAIN
# =========================================================

def realign_applied_omissions(
    *,
    segments: list[dict],
    normalized_audio: str,
    whisper_engine,
    work_dir: str,
    media_duration: float,
) -> tuple[
    list[dict],
    list[dict],
]:

    result = (
        copy.deepcopy(
            segments
        )
    )

    os.makedirs(
        work_dir,
        exist_ok=True,
    )

    targets = [
        index
        for index, segment
        in enumerate(
            result
        )
        if segment.get(
            "omission_applied"
        )
    ]

    print()
    print(
        "Starting omission local re-alignment..."
    )

    print(
        f"  targets : "
        f"{len(targets)}"
    )

    logs = []

    for target_number, index in enumerate(
        targets,
        start=1,
    ):

        segment = (
            result[index]
        )

        old_end = float(
            segment["end"]
        )

        candidate_end = float(
            segment.get(
                "omission_candidate_end",
                old_end,
            )
        )

        desired_end = max(
            old_end,
            candidate_end,
        )

        if index + 1 < len(
            result
        ):

            next_start = float(
                result[
                    index + 1
                ]["start"]
            )

            desired_end = min(
                desired_end,
                next_start,
            )

        desired_end = min(
            desired_end,
            media_duration,
        )

        segment[
            "end"
        ] = (
            desired_end
        )

        inserted_text = (
            get_inserted_text(
                segment
            )
        )

        print()
        print(
            f"[OMISSION REALIGN #{target_number}]"
        )

        print(
            f"  segment : "
            f"{segment['start']:.2f} "
            f"~ "
            f"{segment['end']:.2f}"
        )

        print(
            f"  text    : "
            f"{segment.get('text', '')}"
        )

        print(
            f"  inserted: "
            f"{inserted_text}"
        )

        # =================================================
        # CASE 0:
        # apply_verified_omissions()가 secondary candidate
        # words를 이미 합쳐줬다면 다시 ASR할 이유가 없음.
        # =================================================

        existing_words = (
            copy.deepcopy(
                segment.get(
                    "words",
                    [],
                )
            )
        )

        existing_text = (
            words_to_text(
                existing_words
            )
        )

        existing_similarity = (
            text_similarity(
                segment.get(
                    "text",
                    "",
                ),
                existing_text,
            )
        )

        existing_candidate_present = (
            phrase_exists(
                inserted_text,
                existing_text,
            )
            if inserted_text
            else False
        )

        if (
            existing_words
            and
            existing_candidate_present
        ):

            segment[
                "alignment_source"
            ] = (
                "verified_secondary_merged"
            )

            segment[
                "alignment_similarity"
            ] = (
                existing_similarity
            )

            print(
                f"  [existing] "
                f"words={len(existing_words)} "
                f"similarity="
                f"{existing_similarity:.3f}"
            )

            print(
                f"    {existing_text}"
            )

            print(
                "  -> SUCCESS "
                "(verified secondary words preserved)"
            )

            logs.append(
                {
                    "segment_index": (
                        index
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

                    "text": (
                        segment.get(
                            "text",
                            "",
                        )
                    ),

                    "inserted_text": (
                        inserted_text
                    ),

                    "success": (
                        True
                    ),

                    "candidate_present": (
                        True
                    ),

                    "alignment_similarity": (
                        existing_similarity
                    ),

                    "alignment_text": (
                        existing_text
                    ),

                    "word_count": (
                        len(
                            existing_words
                        )
                    ),

                    "mode": (
                        "existing_secondary"
                    ),

                    "attempts": [],
                }
            )

            continue

        # =================================================
        # OTHERWISE REALIGN
        # =================================================

        attempts = []

        normal_file = os.path.join(
            work_dir,
            (
                f"omission_"
                f"{target_number:04d}_"
                f"normal.wav"
            ),
        )

        normal = (
            run_alignment_attempt(
                segment=(
                    segment
                ),
                normalized_audio=(
                    normalized_audio
                ),
                whisper_engine=(
                    whisper_engine
                ),
                output_file=(
                    normal_file
                ),
                media_duration=(
                    media_duration
                ),
                context=(
                    DEFAULT_CONTEXT
                ),
            )
        )

        attempts.append(
            normal
        )

        print(
            f"  [normal] "
            f"words="
            f"{len(normal['words'])} "
            f"similarity="
            f"{normal['similarity']:.3f} "
            f"candidate="
            f"{normal['candidate_present']}"
        )

        print(
            f"    "
            f"{normal['detected_text']}"
        )

        #
        # similarity만 높아서는 안 됨.
        # 삽입한 candidate 자체도 있어야 한다.
        #

        normal_success = (
            bool(
                normal[
                    "words"
                ]
            )
            and
            normal[
                "similarity"
            ]
            >=
            MIN_REALIGN_SIMILARITY
            and
            normal[
                "candidate_present"
            ]
        )

        if not normal_success:

            wide_file = os.path.join(
                work_dir,
                (
                    f"omission_"
                    f"{target_number:04d}_"
                    f"wide.wav"
                ),
            )

            wide = (
                run_alignment_attempt(
                    segment=(
                        segment
                    ),
                    normalized_audio=(
                        normalized_audio
                    ),
                    whisper_engine=(
                        whisper_engine
                    ),
                    output_file=(
                        wide_file
                    ),
                    media_duration=(
                        media_duration
                    ),
                    context=(
                        WIDE_CONTEXT
                    ),
                )
            )

            attempts.append(
                wide
            )

            print(
                f"  [wide]   "
                f"words="
                f"{len(wide['words'])} "
                f"similarity="
                f"{wide['similarity']:.3f} "
                f"candidate="
                f"{wide['candidate_present']}"
            )

            print(
                f"    "
                f"{wide['detected_text']}"
            )

        valid_attempts = [
            attempt
            for attempt in attempts
            if (
                attempt[
                    "words"
                ]
                and
                attempt[
                    "similarity"
                ]
                >=
                MIN_REALIGN_SIMILARITY
                and
                attempt[
                    "candidate_present"
                ]
            )
        ]

        if valid_attempts:

            best = max(
                valid_attempts,
                key=lambda item: (
                    item[
                        "similarity"
                    ]
                ),
            )

            segment[
                "words"
            ] = (
                copy.deepcopy(
                    best[
                        "words"
                    ]
                )
            )

            segment[
                "alignment_source"
            ] = (
                "omission_local_realign"
            )

            segment[
                "alignment_similarity"
            ] = (
                best[
                    "similarity"
                ]
            )

            success = True

            print(
                "  -> SUCCESS"
            )

        else:

            #
            # 재정렬 실패 시 verified candidate words를
            # 날리지 않고 기존 상태를 유지한다.
            #

            best = max(
                attempts,
                key=lambda item: (
                    item[
                        "similarity"
                    ]
                ),
            )

            success = False

            print(
                "  -> FAILED "
                "(candidate missing in re-alignment)"
            )

        logs.append(
            {
                "segment_index": (
                    index
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

                "text": (
                    segment.get(
                        "text",
                        "",
                    )
                ),

                "inserted_text": (
                    inserted_text
                ),

                "success": (
                    success
                ),

                "candidate_present": (
                    best[
                        "candidate_present"
                    ]
                ),

                "alignment_similarity": (
                    best[
                        "similarity"
                    ]
                ),

                "alignment_text": (
                    best[
                        "detected_text"
                    ]
                ),

                "word_count": (
                    len(
                        best[
                            "words"
                        ]
                    )
                ),

                "mode": (
                    "realign"
                ),

                "attempts": [
                    {
                        "context": (
                            attempt[
                                "context"
                            ]
                        ),

                        "similarity": (
                            attempt[
                                "similarity"
                            ]
                        ),

                        "candidate_present": (
                            attempt[
                                "candidate_present"
                            ]
                        ),

                        "detected_text": (
                            attempt[
                                "detected_text"
                            ]
                        ),

                        "word_count": (
                            len(
                                attempt[
                                    "words"
                                ]
                            )
                        ),
                    }
                    for attempt in attempts
                ],
            }
        )

    successful = sum(
        1
        for item in logs
        if item[
            "success"
        ]
    )

    print()
    print(
        "Omission local re-alignment:"
    )

    print(
        f"  targets  : "
        f"{len(logs)}"
    )

    print(
        f"  success  : "
        f"{successful}"
    )

    print(
        f"  failed   : "
        f"{len(logs) - successful}"
    )

    return (
        result,
        logs,
    )