import math
import os
import re
import subprocess
import wave

from array import array
from difflib import SequenceMatcher


# =========================================================
# CONFIG
# =========================================================

DEFAULT_MIN_GAP = 0.20
DEFAULT_MAX_GAP = 3.00

DEFAULT_MIN_ACTIVITY_RATIO = 0.20

# PCM 16-bit 기준
# -42 dBFS 이상인 frame을 activity로 본다.
DEFAULT_ACTIVITY_DBFS_THRESHOLD = -42.0

DEFAULT_BOUNDARY_MARGIN = 0.20

DEFAULT_MIN_CONSENSUS = 2

DEFAULT_PHRASE_SIMILARITY = 0.75

DEFAULT_MAX_WORDS = 6


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


def text_similarity(
    left: str,
    right: str,
) -> float:

    left_norm = normalize_text(
        left
    )

    right_norm = normalize_text(
        right
    )

    if not left_norm:
        return 0.0

    if not right_norm:
        return 0.0

    return SequenceMatcher(
        None,
        left_norm,
        right_norm,
    ).ratio()


# =========================================================
# AUDIO ACTIVITY
# =========================================================

def _calculate_dbfs(
    samples: array,
) -> float:

    if not samples:
        return -100.0

    square_sum = 0.0

    for sample in samples:

        square_sum += (
            float(sample)
            * float(sample)
        )

    rms = math.sqrt(
        square_sum
        / len(samples)
    )

    if rms <= 0.0:
        return -100.0

    full_scale = 32768.0

    return 20.0 * math.log10(
        rms / full_scale
    )


def calculate_audio_activity_ratio(
    wav_file: str,
    start: float,
    end: float,
    frame_ms: int = 20,
    dbfs_threshold: float = (
        DEFAULT_ACTIVITY_DBFS_THRESHOLD
    ),
) -> float:

    if end <= start:
        return 0.0

    with wave.open(
        wav_file,
        "rb",
    ) as wav:

        channels = (
            wav.getnchannels()
        )

        sample_width = (
            wav.getsampwidth()
        )

        sample_rate = (
            wav.getframerate()
        )

        total_frames = (
            wav.getnframes()
        )

        if sample_width != 2:

            raise RuntimeError(
                "Omission detector expects "
                "16-bit PCM WAV."
            )

        start_frame = max(
            0,
            int(
                start
                * sample_rate
            ),
        )

        end_frame = min(
            total_frames,
            int(
                end
                * sample_rate
            ),
        )

        if end_frame <= start_frame:
            return 0.0

        wav.setpos(
            start_frame
        )

        raw = wav.readframes(
            end_frame
            - start_frame
        )

    samples = array(
        "h"
    )

    samples.frombytes(
        raw
    )

    # stereo 등일 경우 mono처럼 한 channel만 검사
    if channels > 1:

        samples = array(
            "h",
            samples[
                ::channels
            ],
        )

    frame_samples = max(
        1,
        int(
            sample_rate
            * frame_ms
            / 1000
        ),
    )

    total_count = 0
    active_count = 0

    for offset in range(
        0,
        len(samples),
        frame_samples,
    ):

        frame = samples[
            offset:
            offset
            + frame_samples
        ]

        if len(frame) < (
            frame_samples // 2
        ):
            continue

        total_count += 1

        dbfs = (
            _calculate_dbfs(
                frame
            )
        )

        if (
            dbfs
            >=
            dbfs_threshold
        ):
            active_count += 1

    if total_count == 0:
        return 0.0

    return (
        active_count
        / total_count
    )


# =========================================================
# AUDIO CLIP
# =========================================================

def extract_audio_window(
    input_wav: str,
    output_wav: str,
    start: float,
    end: float,
):

    duration = max(
        0.01,
        end - start,
    )

    os.makedirs(
        os.path.dirname(
            output_wav
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
        input_wav,
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_wav,
    ]

    subprocess.run(
        command,
        check=True,
    )


# =========================================================
# SHORT GAP DETECTION
# =========================================================

def detect_short_transcript_gaps(
    segments: list[dict],
    media_duration: float,
    min_gap: float = (
        DEFAULT_MIN_GAP
    ),
    max_gap: float = (
        DEFAULT_MAX_GAP
    ),
) -> list[dict]:

    if not segments:
        return []

    ordered = sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    gaps = []

    for index in range(
        len(ordered) - 1
    ):

        left = (
            ordered[index]
        )

        right = (
            ordered[index + 1]
        )

        gap_start = max(
            0.0,
            float(
                left["end"]
            ),
        )

        gap_end = min(
            media_duration,
            float(
                right["start"]
            ),
        )

        gap_duration = (
            gap_end
            - gap_start
        )

        if gap_duration < min_gap:
            continue

        if gap_duration >= max_gap:
            # 3초 이상은 기존 Gap Recovery 담당
            continue

        gaps.append(
            {
                "index": index,
                "start": (
                    gap_start
                ),
                "end": (
                    gap_end
                ),
                "duration": (
                    gap_duration
                ),
                "left": left,
                "right": right,
            }
        )

    return gaps


# =========================================================
# WINDOW BUILD
# =========================================================

def build_probe_windows(
    gap_start: float,
    gap_end: float,
    media_duration: float,
) -> list[dict]:

    configs = [
        (
            "center",
            2.5,
            2.5,
        ),
        (
            "shift_early",
            4.0,
            1.5,
        ),
        (
            "shift_late",
            1.5,
            4.0,
        ),
    ]

    windows = []

    for (
        name,
        before,
        after,
    ) in configs:

        start = max(
            0.0,
            gap_start
            - before,
        )

        end = min(
            media_duration,
            gap_end
            + after,
        )

        if end <= start:
            continue

        windows.append(
            {
                "name": name,
                "start": start,
                "end": end,
            }
        )

    return windows


# =========================================================
# CANONICAL CONTEXT
# =========================================================

def build_neighbor_text(
    gap: dict,
) -> str:

    left_text = (
        str(
            gap["left"]
            .get(
                "text",
                "",
            )
        )
        .strip()
    )

    right_text = (
        str(
            gap["right"]
            .get(
                "text",
                "",
            )
        )
        .strip()
    )

    return (
        left_text
        + " "
        + right_text
    ).strip()


def word_exists_in_context(
    word_text: str,
    context: str,
) -> bool:

    word_norm = (
        normalize_text(
            word_text
        )
    )

    context_norm = (
        normalize_text(
            context
        )
    )

    if not word_norm:
        return True

    if word_norm in context_norm:
        return True

    # 1~2글자 조각은 오탐이 너무 많다.
    if len(word_norm) <= 2:
        return True

    return False


# =========================================================
# WORD GROUPING
# =========================================================

def group_candidate_words(
    words: list[dict],
    max_word_gap: float = 0.80,
    max_words: int = (
        DEFAULT_MAX_WORDS
    ),
) -> list[dict]:

    if not words:
        return []

    words = sorted(
        words,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    groups = []

    current = [
        words[0]
    ]

    for word in words[1:]:

        previous = (
            current[-1]
        )

        gap = (
            word["start"]
            - previous["end"]
        )

        if (
            gap
            <= max_word_gap
            and
            len(current)
            < max_words
        ):
            current.append(
                word
            )

        else:

            groups.append(
                current
            )

            current = [
                word
            ]

    if current:
        groups.append(
            current
        )

    results = []

    for group in groups:

        text = " ".join(
            str(
                item["word"]
            ).strip()
            for item in group
            if str(
                item["word"]
            ).strip()
        )

        if not text:
            continue

        probabilities = [
            float(
                item[
                    "probability"
                ]
            )
            for item in group
            if (
                item.get(
                    "probability"
                )
                is not None
            )
        ]

        average_probability = (
            sum(
                probabilities
            )
            / len(
                probabilities
            )
            if probabilities
            else None
        )

        results.append(
            {
                "start": (
                    group[0][
                        "start"
                    ]
                ),
                "end": (
                    group[-1][
                        "end"
                    ]
                ),
                "text": text,
                "words": group,
                "average_probability": (
                    average_probability
                ),
            }
        )

    return results


# =========================================================
# ONE PROBE
# =========================================================

def run_omission_probe(
    normalized_audio: str,
    whisper_engine,
    probe_dir: str,
    gap: dict,
    window: dict,
    probe_index: int,
    boundary_margin: float = (
        DEFAULT_BOUNDARY_MARGIN
    ),
) -> list[dict]:

    output_file = os.path.join(
        probe_dir,
        (
            f"gap_"
            f"{gap['index']:04d}_"
            f"{probe_index}_"
            f"{window['name']}.wav"
        ),
    )

    extract_audio_window(
        input_wav=(
            normalized_audio
        ),
        output_wav=(
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
        window["end"]
        - window["start"]
    )

    words = (
        whisper_engine
        .transcribe_words(
            audio_file=(
                output_file
            ),
            global_offset=(
                window[
                    "start"
                ]
            ),
            audio_duration=(
                clip_duration
            ),
        )
    )

    context = (
        build_neighbor_text(
            gap
        )
    )

    selected_words = []

    select_start = (
        gap["start"]
        - boundary_margin
    )

    select_end = (
        gap["end"]
        + boundary_margin
    )

    for word in words:

        midpoint = (
            float(
                word["start"]
            )
            +
            float(
                word["end"]
            )
        ) / 2.0

        if midpoint < select_start:
            continue

        if midpoint > select_end:
            continue

        word_text = str(
            word.get(
                "word",
                "",
            )
        ).strip()

        if not word_text:
            continue

        # 이미 canonical 좌/우 문장에 있는 단어라면
        # boundary overlap일 가능성이 높다.
        if word_exists_in_context(
            word_text=(
                word_text
            ),
            context=(
                context
            ),
        ):
            continue

        selected_words.append(
            word
        )

    return (
        group_candidate_words(
            selected_words
        )
    )


# =========================================================
# CONSENSUS
# =========================================================

def find_consensus_candidate(
    attempts: list[dict],
    min_consensus: int = (
        DEFAULT_MIN_CONSENSUS
    ),
    similarity_threshold: float = (
        DEFAULT_PHRASE_SIMILARITY
    ),
) -> dict | None:

    all_candidates = []

    for attempt in attempts:

        for candidate in (
            attempt[
                "candidates"
            ]
        ):

            item = dict(
                candidate
            )

            item[
                "attempt_name"
            ] = (
                attempt[
                    "name"
                ]
            )

            all_candidates.append(
                item
            )

    if not all_candidates:
        return None

    best = None

    for base in all_candidates:

        matches = [
            base
        ]

        used_attempts = {
            base[
                "attempt_name"
            ]
        }

        for other in (
            all_candidates
        ):

            if (
                other[
                    "attempt_name"
                ]
                in used_attempts
            ):
                continue

            similarity = (
                text_similarity(
                    base[
                        "text"
                    ],
                    other[
                        "text"
                    ],
                )
            )

            if (
                similarity
                <
                similarity_threshold
            ):
                continue

            matches.append(
                other
            )

            used_attempts.add(
                other[
                    "attempt_name"
                ]
            )

        consensus = len(
            used_attempts
        )

        if (
            consensus
            <
            min_consensus
        ):
            continue

        # 가장 짧고 보수적인 표현 선택
        representative = min(
            matches,
            key=lambda item: (
                len(
                    normalize_text(
                        item[
                            "text"
                        ]
                    )
                ),
                item[
                    "start"
                ],
            ),
        )

        probabilities = [
            item[
                "average_probability"
            ]
            for item in matches
            if (
                item.get(
                    "average_probability"
                )
                is not None
            )
        ]

        avg_probability = (
            sum(
                probabilities
            )
            / len(
                probabilities
            )
            if probabilities
            else None
        )

        score = (
            float(
                consensus
            )
            +
            (
                avg_probability
                if avg_probability
                is not None
                else 0.0
            )
        )

        result = {
            "text": (
                representative[
                    "text"
                ]
            ),
            "start": min(
                item[
                    "start"
                ]
                for item
                in matches
            ),
            "end": max(
                item[
                    "end"
                ]
                for item
                in matches
            ),
            "consensus": (
                consensus
            ),
            "average_probability": (
                avg_probability
            ),
            "attempts": sorted(
                used_attempts
            ),
            "_score": (
                score
            ),
        }

        if (
            best is None
            or
            result[
                "_score"
            ]
            >
            best[
                "_score"
            ]
        ):
            best = result

    if best is None:
        return None

    best.pop(
        "_score",
        None,
    )

    return best


# =========================================================
# MAIN
# =========================================================

def detect_omission_candidates(
    normalized_audio: str,
    segments: list[dict],
    whisper_engine,
    probe_dir: str,
    media_duration: float,
    min_gap: float = (
        DEFAULT_MIN_GAP
    ),
    max_gap: float = (
        DEFAULT_MAX_GAP
    ),
    min_activity_ratio: float = (
        DEFAULT_MIN_ACTIVITY_RATIO
    ),
) -> list[dict]:

    os.makedirs(
        probe_dir,
        exist_ok=True,
    )

    gaps = (
        detect_short_transcript_gaps(
            segments=(
                segments
            ),
            media_duration=(
                media_duration
            ),
            min_gap=(
                min_gap
            ),
            max_gap=(
                max_gap
            ),
        )
    )

    print()
    print(
        "Starting omission candidate detection..."
    )

    print(
        f"  short gaps found  : "
        f"{len(gaps)}"
    )

    results = []

    active_gap_count = 0
    probe_count = 0

    for gap_number, gap in enumerate(
        gaps,
        start=1,
    ):

        activity_ratio = (
            calculate_audio_activity_ratio(
                wav_file=(
                    normalized_audio
                ),
                start=(
                    gap[
                        "start"
                    ]
                ),
                end=(
                    gap[
                        "end"
                    ]
                ),
            )
        )

        if (
            activity_ratio
            <
            min_activity_ratio
        ):
            continue

        active_gap_count += 1

        print()
        print(
            f"[OMISSION GAP {gap_number}] "
            f"{gap['start']:.2f} "
            f"~ "
            f"{gap['end']:.2f} "
            f"("
            f"{gap['duration']:.2f}s"
            f")"
        )

        print(
            f"  activity          : "
            f"{activity_ratio:.2f}"
        )

        windows = (
            build_probe_windows(
                gap_start=(
                    gap[
                        "start"
                    ]
                ),
                gap_end=(
                    gap[
                        "end"
                    ]
                ),
                media_duration=(
                    media_duration
                ),
            )
        )

        attempts = []

        for probe_index, window in enumerate(
            windows,
            start=1,
        ):

            probe_count += 1

            candidates = (
                run_omission_probe(
                    normalized_audio=(
                        normalized_audio
                    ),
                    whisper_engine=(
                        whisper_engine
                    ),
                    probe_dir=(
                        probe_dir
                    ),
                    gap=(
                        gap
                    ),
                    window=(
                        window
                    ),
                    probe_index=(
                        probe_index
                    ),
                )
            )

            print(
                f"  [{window['name']}] "
                f"{window['start']:.2f} "
                f"~ "
                f"{window['end']:.2f}"
            )

            if not candidates:

                print(
                    "    candidate: none"
                )

            else:

                for candidate in (
                    candidates
                ):

                    print(
                        f"    candidate: "
                        f"{candidate['start']:.2f} "
                        f"~ "
                        f"{candidate['end']:.2f} "
                        f"{candidate['text']}"
                    )

            attempts.append(
                {
                    "name": (
                        window[
                            "name"
                        ]
                    ),
                    "candidates": (
                        candidates
                    ),
                }
            )

        consensus = (
            find_consensus_candidate(
                attempts
            )
        )

        if consensus is None:

            print(
                "  -> no stable omission candidate"
            )

            continue

        print(
            f"  -> omission candidate: "
            f"{consensus['text']}"
        )

        print(
            f"  -> consensus         : "
            f"{consensus['consensus']}"
        )

        results.append(
            {
                "type": (
                    "omission_candidate"
                ),

                "gap_start": (
                    gap[
                        "start"
                    ]
                ),

                "gap_end": (
                    gap[
                        "end"
                    ]
                ),

                "gap_duration": (
                    gap[
                        "duration"
                    ]
                ),

                "activity_ratio": (
                    activity_ratio
                ),

                "start": (
                    consensus[
                        "start"
                    ]
                ),

                "end": (
                    consensus[
                        "end"
                    ]
                ),

                "text": (
                    consensus[
                        "text"
                    ]
                ),

                "consensus": (
                    consensus[
                        "consensus"
                    ]
                ),

                "average_probability": (
                    consensus[
                        "average_probability"
                    ]
                ),

                "attempts": (
                    consensus[
                        "attempts"
                    ]
                ),

                "left_text": (
                    gap[
                        "left"
                    ]
                    .get(
                        "text",
                        "",
                    )
                ),

                "right_text": (
                    gap[
                        "right"
                    ]
                    .get(
                        "text",
                        "",
                    )
                ),

                "action": (
                    "review"
                ),
            }
        )

    print()
    print(
        "Omission detector:"
    )

    print(
        f"  active short gaps : "
        f"{active_gap_count}"
    )

    print(
        f"  probe passes      : "
        f"{probe_count}"
    )

    print(
        f"  candidates        : "
        f"{len(results)}"
    )

    return results