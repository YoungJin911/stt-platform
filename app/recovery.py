import math
import os
import re
import wave

from array import array
from difflib import SequenceMatcher

from app.media import (
    extract_audio_clip,
)


# =========================================================
# TEXT UTIL
# =========================================================

def normalize_recovery_text(
    text: str,
) -> str:
    text = (
        text
        .strip()
        .lower()
    )

    return re.sub(
        r"[\s\W_]+",
        "",
        text,
        flags=re.UNICODE,
    )


def text_similarity(
    text_a: str,
    text_b: str,
) -> float:
    a = normalize_recovery_text(
        text_a
    )

    b = normalize_recovery_text(
        text_b
    )

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def make_safe_filename_number(
    value: float,
) -> str:
    return (
        f"{value:.2f}"
        .replace(".", "_")
    )


# =========================================================
# INTERVAL UTIL
# =========================================================

def get_interval_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    start = max(
        start_a,
        start_b,
    )

    end = min(
        end_a,
        end_b,
    )

    return max(
        0.0,
        end - start,
    )


def merge_intervals(
    intervals: list[list[float]],
) -> list[list[float]]:
    if not intervals:
        return []

    intervals = sorted(
        intervals,
        key=lambda item: item[0],
    )

    merged = [
        intervals[0][:]
    ]

    for start, end in intervals[1:]:
        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(
                previous[1],
                end,
            )

        else:
            merged.append(
                [start, end]
            )

    return merged


def calculate_region_coverage(
    segments: list[dict],
    region_start: float,
    region_end: float,
) -> float:
    duration = (
        region_end
        - region_start
    )

    if duration <= 0:
        return 0.0

    intervals = []

    for segment in segments:
        start = max(
            segment["start"],
            region_start,
        )

        end = min(
            segment["end"],
            region_end,
        )

        if end > start:
            intervals.append(
                [start, end]
            )

    if not intervals:
        return 0.0

    merged = merge_intervals(
        intervals
    )

    covered = sum(
        end - start
        for start, end
        in merged
    )

    return min(
        1.0,
        covered / duration,
    )


def calculate_max_uncovered_gap(
    segments: list[dict],
    region_start: float,
    region_end: float,
) -> float:
    if region_end <= region_start:
        return 0.0

    intervals = []

    for segment in segments:
        start = max(
            segment["start"],
            region_start,
        )

        end = min(
            segment["end"],
            region_end,
        )

        if end > start:
            intervals.append(
                [start, end]
            )

    if not intervals:
        return (
            region_end
            - region_start
        )

    merged = merge_intervals(
        intervals
    )

    maximum_gap = max(
        0.0,
        merged[0][0]
        - region_start,
    )

    for previous, current in zip(
        merged,
        merged[1:],
    ):
        maximum_gap = max(
            maximum_gap,
            current[0]
            - previous[1],
        )

    maximum_gap = max(
        maximum_gap,
        region_end
        - merged[-1][1],
    )

    return maximum_gap


def find_uncovered_intervals(
    segments: list[dict],
    region_start: float,
    region_end: float,
    minimum_gap: float = 2.5,
) -> list[dict]:
    """
    특정 region 내에서 STT segment가 덮지 못한
    시간 구간을 찾는다.

    LONG reconstruction 후 남은 내부 공백 탐지용.
    """

    intervals = []

    for segment in segments:
        start = max(
            region_start,
            segment["start"],
        )

        end = min(
            region_end,
            segment["end"],
        )

        if end > start:
            intervals.append(
                [start, end]
            )

    merged = merge_intervals(
        intervals
    )

    gaps = []

    cursor = (
        region_start
    )

    for start, end in merged:
        if (
            start - cursor
            >= minimum_gap
        ):
            gaps.append(
                {
                    "start": cursor,
                    "end": start,
                    "duration": (
                        start - cursor
                    ),
                }
            )

        cursor = max(
            cursor,
            end,
        )

    if (
        region_end - cursor
        >= minimum_gap
    ):
        gaps.append(
            {
                "start": cursor,
                "end": region_end,
                "duration": (
                    region_end
                    - cursor
                ),
            }
        )

    return gaps


# =========================================================
# AUDIO ACTIVITY
# =========================================================

def calculate_dbfs(
    samples: array,
) -> float:
    if not samples:
        return -100.0

    square_sum = 0.0

    for sample in samples:
        square_sum += (
            sample * sample
        )

    rms = math.sqrt(
        square_sum
        / len(samples)
    )

    if rms <= 0:
        return -100.0

    return (
        20.0
        * math.log10(
            rms / 32768.0
        )
    )


def get_audio_activity_ratio(
    wav_file: str,
    start_time: float,
    end_time: float,
    dbfs_threshold: float = -45.0,
    window_duration: float = 0.5,
) -> float:
    with wave.open(
        wav_file,
        "rb",
    ) as wav:
        sample_rate = (
            wav.getframerate()
        )

        channels = (
            wav.getnchannels()
        )

        sample_width = (
            wav.getsampwidth()
        )

        total_frames = (
            wav.getnframes()
        )

        if channels != 1:
            raise ValueError(
                "normalized WAV는 mono여야 합니다."
            )

        if sample_width != 2:
            raise ValueError(
                "normalized WAV는 16-bit PCM이어야 합니다."
            )

        start_frame = max(
            0,
            int(
                start_time
                * sample_rate
            ),
        )

        end_frame = min(
            total_frames,
            int(
                end_time
                * sample_rate
            ),
        )

        if end_frame <= start_frame:
            return 0.0

        window_frames = max(
            1,
            int(
                window_duration
                * sample_rate
            ),
        )

        wav.setpos(
            start_frame
        )

        current_frame = (
            start_frame
        )

        total_windows = 0
        active_windows = 0

        while current_frame < end_frame:
            frames_to_read = min(
                window_frames,
                end_frame
                - current_frame,
            )

            raw = wav.readframes(
                frames_to_read
            )

            if not raw:
                break

            samples = array(
                "h"
            )

            samples.frombytes(
                raw
            )

            dbfs = calculate_dbfs(
                samples
            )

            total_windows += 1

            if dbfs >= dbfs_threshold:
                active_windows += 1

            current_frame += (
                frames_to_read
            )

        if total_windows == 0:
            return 0.0

        return (
            active_windows
            / total_windows
        )


# =========================================================
# GAP DETECTION
# =========================================================

def interval_is_ignored(
    gap_start: float,
    gap_end: float,
    ignored_intervals: list[dict],
) -> bool:
    """
    LONG micro recovery에서 이미 처리한 구간이면
    일반 GAP recovery에서 다시 돌리지 않는다.
    """

    for interval in ignored_intervals:
        overlap = get_interval_overlap(
            gap_start,
            gap_end,
            interval["start"],
            interval["end"],
        )

        gap_duration = (
            gap_end
            - gap_start
        )

        if gap_duration <= 0:
            continue

        overlap_ratio = (
            overlap
            / gap_duration
        )

        if overlap_ratio >= 0.80:
            return True

    return False


def detect_transcript_gaps(
    segments: list[dict],
    media_duration: float,
    min_gap: float = 3.0,
    ignored_intervals: list[dict] | None = None,
) -> list[dict]:
    if ignored_intervals is None:
        ignored_intervals = []

    if not segments:
        return [
            {
                "id": 1,
                "start": 0.0,
                "end": media_duration,
                "duration": media_duration,
            }
        ]

    sorted_segments = sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    gaps = []

    gap_id = 1

    first_start = (
        sorted_segments[0]["start"]
    )

    if first_start >= min_gap:
        if not interval_is_ignored(
            0.0,
            first_start,
            ignored_intervals,
        ):
            gaps.append(
                {
                    "id": gap_id,
                    "start": 0.0,
                    "end": first_start,
                    "duration": first_start,
                }
            )

            gap_id += 1

    for previous, current in zip(
        sorted_segments,
        sorted_segments[1:],
    ):
        gap_start = (
            previous["end"]
        )

        gap_end = (
            current["start"]
        )

        gap_duration = (
            gap_end
            - gap_start
        )

        if gap_duration >= min_gap:
            if interval_is_ignored(
                gap_start,
                gap_end,
                ignored_intervals,
            ):
                print(
                    f"[GAP SKIP] "
                    f"{gap_start:.2f} "
                    f"~ {gap_end:.2f} "
                    f"(already handled by micro recovery)"
                )

                continue

            gaps.append(
                {
                    "id": gap_id,
                    "start": gap_start,
                    "end": gap_end,
                    "duration": gap_duration,
                }
            )

            gap_id += 1

    last_end = (
        sorted_segments[-1]["end"]
    )

    final_gap = (
        media_duration
        - last_end
    )

    if final_gap >= min_gap:
        if not interval_is_ignored(
            last_end,
            media_duration,
            ignored_intervals,
        ):
            gaps.append(
                {
                    "id": gap_id,
                    "start": last_end,
                    "end": media_duration,
                    "duration": final_gap,
                }
            )

    return gaps


# =========================================================
# NORMAL GAP RECOVERY
# =========================================================

def build_recovery_windows(
    gap_start: float,
    gap_end: float,
    media_duration: float,
) -> list[dict]:
    strategies = [
        {
            "name": "default",
            "before": 3.0,
            "after": 3.0,
        },
        {
            "name": "shift_early",
            "before": 4.0,
            "after": 2.0,
        },
        {
            "name": "shift_earlier",
            "before": 6.0,
            "after": 2.0,
        },
        {
            "name": "shift_late",
            "before": 2.0,
            "after": 5.0,
        },
    ]

    windows = []

    for strategy in strategies:
        start = max(
            0.0,
            gap_start
            - strategy["before"],
        )

        end = min(
            media_duration,
            gap_end
            + strategy["after"],
        )

        windows.append(
            {
                "name": (
                    strategy["name"]
                ),
                "start": start,
                "end": end,
                "duration": (
                    end - start
                ),
            }
        )

    return windows


def select_gap_segments(
    segments: list[dict],
    gap: dict,
) -> list[dict]:
    gap_start = (
        gap["start"]
    )

    gap_end = (
        gap["end"]
    )

    accepted = []

    for segment in segments:
        start = (
            segment["start"]
        )

        end = (
            segment["end"]
        )

        duration = (
            end - start
        )

        if duration <= 0:
            continue

        midpoint = (
            start + end
        ) / 2.0

        overlap = (
            get_interval_overlap(
                start,
                end,
                gap_start,
                gap_end,
            )
        )

        overlap_ratio = (
            overlap / duration
        )

        if (
            midpoint < gap_start
            or midpoint > gap_end
        ) and overlap_ratio < 0.50:
            print(
                f"      REJECT "
                f"{start:.2f} "
                f"~ {end:.2f} "
                f"(outside gap)"
            )

            continue

        clipped = dict(
            segment
        )

        clipped["start"] = max(
            start,
            gap_start,
        )

        clipped["end"] = min(
            end,
            gap_end,
        )

        if (
            clipped["end"]
            <= clipped["start"]
        ):
            continue

        clipped["gap_id"] = (
            gap["id"]
        )

        accepted.append(
            clipped
        )

    return accepted


def calculate_gap_attempt_score(
    segments: list[dict],
    gap: dict,
) -> dict:
    gap_start = (
        gap["start"]
    )

    gap_end = (
        gap["end"]
    )

    gap_duration = max(
        gap_end - gap_start,
        0.001,
    )

    if not segments:
        return {
            "coverage": 0.0,
            "max_gap": gap_duration,
            "start_gap": gap_duration,
            "end_gap": gap_duration,
            "chars": 0,
            "score": -999.0,
        }

    ordered = sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    coverage = (
        calculate_region_coverage(
            segments=ordered,
            region_start=gap_start,
            region_end=gap_end,
        )
    )

    max_gap = (
        calculate_max_uncovered_gap(
            segments=ordered,
            region_start=gap_start,
            region_end=gap_end,
        )
    )

    start_gap = max(
        0.0,
        ordered[0]["start"]
        - gap_start,
    )

    end_gap = max(
        0.0,
        gap_end
        - ordered[-1]["end"],
    )

    char_count = sum(
        len(
            normalize_recovery_text(
                item["text"]
            )
        )
        for item in ordered
    )

    density_score = min(
        1.0,
        (
            char_count
            / gap_duration
        ) / 5.0,
    )

    boundary_score = (
        1.0
        - min(
            1.0,
            (
                start_gap
                + end_gap
            )
            / gap_duration,
        )
    )

    internal_score = (
        1.0
        - min(
            1.0,
            max_gap
            / gap_duration,
        )
    )

    score = (
        coverage * 0.50
        + boundary_score * 0.20
        + internal_score * 0.15
        + density_score * 0.15
    )

    return {
        "coverage": coverage,
        "max_gap": max_gap,
        "start_gap": start_gap,
        "end_gap": end_gap,
        "chars": char_count,
        "score": score,
    }


def is_gap_result_excellent(
    metrics: dict,
) -> bool:
    return (
        metrics["score"] >= 0.95
        and
        metrics["coverage"] >= 0.90
        and
        metrics["max_gap"] <= 1.0
        and
        metrics["start_gap"] <= 0.5
        and
        metrics["end_gap"] <= 0.5
    )


def recover_gaps(
    normalized_audio: str,
    gaps: list[dict],
    whisper_engine,
    recovery_dir: str,
    media_duration: float,
    minimum_active_ratio: float = 0.15,
    good_coverage_ratio: float = 0.80,
    maximum_internal_gap: float = 2.0,
    max_attempts: int = 4,
) -> list[dict]:
    all_recovered = []

    for gap in gaps:
        gap_id = (
            gap["id"]
        )

        print()
        print(
            f"[GAP {gap_id}] "
            f"{gap['start']:.2f} "
            f"~ {gap['end']:.2f} sec "
            f"({gap['duration']:.2f}s)"
        )

        activity_ratio = (
            get_audio_activity_ratio(
                wav_file=normalized_audio,
                start_time=gap["start"],
                end_time=gap["end"],
            )
        )

        print(
            f"  -> audio activity ratio: "
            f"{activity_ratio:.2f}"
        )

        if (
            activity_ratio
            < minimum_active_ratio
        ):
            print(
                "  -> recovery skip"
            )

            continue

        windows = (
            build_recovery_windows(
                gap_start=gap["start"],
                gap_end=gap["end"],
                media_duration=media_duration,
            )
        )

        candidates = []

        for attempt_index, window in enumerate(
            windows[:max_attempts],
            start=1,
        ):
            print()
            print(
                f"  [Recovery attempt "
                f"{attempt_index}] "
                f"{window['name']}"
            )

            print(
                f"    window: "
                f"{window['start']:.2f} "
                f"~ {window['end']:.2f}"
            )

            clip_file = os.path.join(
                recovery_dir,
                (
                    f"gap_{gap_id:04d}_"
                    f"{attempt_index:02d}.wav"
                ),
            )

            extract_audio_clip(
                input_file=normalized_audio,
                output_file=clip_file,
                start_time=window["start"],
                duration=window["duration"],
            )

            raw_results = (
                whisper_engine
                .transcribe_file(
                    audio_file=clip_file,
                    global_offset=window["start"],
                    source="recovery",
                    audio_duration=window["duration"],
                    strict_timestamp=True,
                )
            )

            print(
                f"    raw STT segments: "
                f"{len(raw_results)}"
            )

            for item in raw_results:
                print(
                    f"      RAW "
                    f"{item['start']:.2f} "
                    f"~ {item['end']:.2f} "
                    f"{item['text']}"
                )

            accepted = (
                select_gap_segments(
                    raw_results,
                    gap,
                )
            )

            metrics = (
                calculate_gap_attempt_score(
                    accepted,
                    gap,
                )
            )

            print(
                f"    accepted segments: "
                f"{len(accepted)}"
            )

            print(
                f"    coverage          : "
                f"{metrics['coverage']:.2f}"
            )

            print(
                f"    max internal gap  : "
                f"{metrics['max_gap']:.2f}s"
            )

            print(
                f"    start gap         : "
                f"{metrics['start_gap']:.2f}s"
            )

            print(
                f"    end gap           : "
                f"{metrics['end_gap']:.2f}s"
            )

            print(
                f"    text chars        : "
                f"{metrics['chars']}"
            )

            print(
                f"    attempt score     : "
                f"{metrics['score']:.3f}"
            )

            candidate = {
                "attempt": attempt_index,
                "strategy": window["name"],
                "segments": accepted,
                **metrics,
            }

            candidates.append(
                candidate
            )

            if (
                accepted
                and
                is_gap_result_excellent(
                    metrics
                )
            ):
                print(
                    "    -> excellent recovery, "
                    "early stop"
                )

                break

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["coverage"],
                -item["max_gap"],
                item["chars"],
            ),
            reverse=True,
        )

        best = (
            candidates[0]
        )

        print()
        print(
            f"  -> GAP {gap_id} "
            f"best attempt: "
            f"{best['attempt']} "
            f"({best['strategy']})"
        )

        if not best["segments"]:
            print(
                "  -> recovery 실패"
            )

            continue

        if (
            best["coverage"] < 0.50
            and
            best["max_gap"] > 4.0
        ):
            print(
                "  -> recovery 품질 부족"
            )

            continue

        all_recovered.extend(
            best["segments"]
        )

    return all_recovered


# =========================================================
# LONG SEGMENT DETECTION
# =========================================================

def detect_suspicious_long_segments(
    segments: list[dict],
    minimum_duration: float = 12.0,
) -> list[dict]:
    suspicious = []

    for segment in segments:
        duration = (
            segment["end"]
            - segment["start"]
        )

        if duration >= minimum_duration:
            item = dict(
                segment
            )

            item["duration"] = (
                duration
            )

            suspicious.append(
                item
            )

    return suspicious


# =========================================================
# LONG WINDOWS
# =========================================================

def build_long_recovery_windows(
    original_start: float,
    original_end: float,
    media_duration: float,
) -> list[dict]:
    strategies = [
        {
            "name": "left",
            "start": max(
                0.0,
                original_start - 3.0,
            ),
            "end": min(
                media_duration,
                original_start + 15.0,
            ),
        },
        {
            "name": "middle",
            "start": max(
                0.0,
                original_start + 6.0,
            ),
            "end": min(
                media_duration,
                original_start + 24.0,
            ),
        },
        {
            "name": "right",
            "start": max(
                0.0,
                original_end - 15.0,
            ),
            "end": min(
                media_duration,
                original_end + 3.0,
            ),
        },
    ]

    result = []
    seen = set()

    for strategy in strategies:
        if (
            strategy["end"]
            <= strategy["start"]
        ):
            continue

        key = (
            round(
                strategy["start"],
                2,
            ),
            round(
                strategy["end"],
                2,
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            {
                "name": strategy["name"],
                "start": strategy["start"],
                "end": strategy["end"],
                "duration": (
                    strategy["end"]
                    - strategy["start"]
                ),
            }
        )

    return result


# =========================================================
# LONG CANDIDATE DEDUP
# =========================================================

def deduplicate_long_candidates(
    segments: list[dict],
) -> list[dict]:
    ordered = sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    groups = []

    for current in ordered:
        matched = None

        for group in groups:
            representative = (
                group[0]
            )

            similarity = (
                text_similarity(
                    representative["text"],
                    current["text"],
                )
            )

            if similarity >= 0.88:
                matched = group
                break

        if matched is None:
            groups.append(
                [current]
            )

        else:
            matched.append(
                current
            )

    result = []

    for group in groups:
        group.sort(
            key=lambda item: (
                item["end"]
                - item["start"],
                -item["start"],
            )
        )

        result.append(
            group[0]
        )

    return sorted(
        result,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )


# =========================================================
# LONG CANDIDATES
# =========================================================

def collect_long_candidates(
    normalized_audio: str,
    original_segment: dict,
    whisper_engine,
    recovery_dir: str,
    media_duration: float,
) -> list[dict]:
    original_start = (
        original_segment["start"]
    )

    original_end = (
        original_segment["end"]
    )

    safe_start = (
        make_safe_filename_number(
            original_start
        )
    )

    windows = (
        build_long_recovery_windows(
            original_start=original_start,
            original_end=original_end,
            media_duration=media_duration,
        )
    )

    candidates = []

    for index, window in enumerate(
        windows,
        start=1,
    ):
        print()
        print(
            f"  [LONG window {index}] "
            f"{window['name']}"
        )

        print(
            f"    {window['start']:.2f} "
            f"~ {window['end']:.2f}"
        )

        clip_file = os.path.join(
            recovery_dir,
            (
                f"long_"
                f"{safe_start}_"
                f"{window['name']}.wav"
            ),
        )

        extract_audio_clip(
            input_file=normalized_audio,
            output_file=clip_file,
            start_time=window["start"],
            duration=window["duration"],
        )

        raw = (
            whisper_engine
            .transcribe_file(
                audio_file=clip_file,
                global_offset=window["start"],
                source="recovery_long",
                audio_duration=window["duration"],
                strict_timestamp=True,
            )
        )

        for item in raw:
            midpoint = (
                item["start"]
                + item["end"]
            ) / 2.0

            if not (
                original_start
                <= midpoint
                <= original_end
            ):
                continue

            clipped = dict(
                item
            )

            clipped["start"] = max(
                clipped["start"],
                original_start,
            )

            clipped["end"] = min(
                clipped["end"],
                original_end,
            )

            if (
                clipped["end"]
                <= clipped["start"]
            ):
                continue

            clipped[
                "long_window"
            ] = window["name"]

            candidates.append(
                clipped
            )

            print(
                f"      CANDIDATE "
                f"{clipped['start']:.2f} "
                f"~ {clipped['end']:.2f} "
                f"{clipped['text']}"
            )

    return (
        deduplicate_long_candidates(
            candidates
        )
    )


# =========================================================
# LONG ANCHOR
# =========================================================

def choose_long_anchor(
    original_segment: dict,
    candidates: list[dict],
    minimum_similarity: float = 0.68,
    preferred_similarity: float = 0.80,
) -> tuple[dict | None, float]:
    scored = []

    for candidate in candidates:
        similarity = (
            text_similarity(
                original_segment["text"],
                candidate["text"],
            )
        )

        duration = (
            candidate["end"]
            - candidate["start"]
        )

        scored.append(
            {
                "segment": candidate,
                "similarity": similarity,
                "duration": duration,
            }
        )

    preferred = [
        item
        for item in scored
        if (
            item["similarity"]
            >= preferred_similarity
        )
    ]

    if preferred:
        preferred.sort(
            key=lambda item: (
                item["duration"],
                -item["segment"]["start"],
                -item["similarity"],
            )
        )

        selected = preferred[0]

        return (
            selected["segment"],
            selected["similarity"],
        )

    reliable = [
        item
        for item in scored
        if (
            item["similarity"]
            >= minimum_similarity
        )
    ]

    if not reliable:
        return (
            None,
            0.0,
        )

    reliable.sort(
        key=lambda item: (
            -item["similarity"],
            item["duration"],
            -item["segment"]["start"],
        )
    )

    selected = reliable[0]

    return (
        selected["segment"],
        selected["similarity"],
    )


# =========================================================
# LONG TIMELINE CLEAN
# =========================================================

def clean_long_timeline(
    segments: list[dict],
) -> list[dict]:
    if not segments:
        return []

    ordered = sorted(
        [
            dict(item)
            for item in segments
        ],
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    cleaned = []

    for item in ordered:
        if not cleaned:
            cleaned.append(
                item
            )

            continue

        previous = (
            cleaned[-1]
        )

        if (
            item["start"]
            < previous["end"]
        ):
            previous["end"] = (
                item["start"]
            )

            if (
                previous["end"]
                - previous["start"]
                < 0.50
            ):
                cleaned.pop()

        cleaned.append(
            item
        )

    return cleaned


# =========================================================
# LONG INITIAL RECONSTRUCTION
# =========================================================

def reconstruct_long_segment(
    original_segment: dict,
    candidates: list[dict],
) -> list[dict]:
    if not candidates:
        return []

    original_text = (
        original_segment["text"]
    )

    anchor, anchor_similarity = (
        choose_long_anchor(
            original_segment,
            candidates,
        )
    )

    if anchor is None:
        print(
            "  -> anchor 없음"
        )

        return []

    print()
    print(
        f"  -> anchor similarity: "
        f"{anchor_similarity:.3f}"
    )

    print(
        f"  -> anchor: "
        f"{anchor['start']:.2f} "
        f"~ {anchor['end']:.2f} "
        f"{anchor['text']}"
    )

    novel_segments = []

    for candidate in candidates:
        if (
            abs(
                candidate["start"]
                - anchor["start"]
            ) < 0.01
            and
            abs(
                candidate["end"]
                - anchor["end"]
            ) < 0.01
        ):
            continue

        similarity_original = (
            text_similarity(
                original_text,
                candidate["text"],
            )
        )

        if similarity_original >= 0.72:
            continue

        if (
            candidate["start"]
            >= anchor["start"]
        ):
            continue

        novel = dict(
            candidate
        )

        novel["end"] = min(
            novel["end"],
            anchor["start"],
        )

        if (
            novel["end"]
            - novel["start"]
            < 0.50
        ):
            continue

        novel_segments.append(
            novel
        )

    novel_segments = (
        deduplicate_long_candidates(
            novel_segments
        )
    )

    result = (
        novel_segments
        + [dict(anchor)]
    )

    return clean_long_timeline(
        result
    )


# =========================================================
# MICRO RECOVERY
# =========================================================

def build_micro_windows(
    gap_start: float,
    gap_end: float,
    media_duration: float,
) -> list[dict]:
    """
    LONG 내부 누락구간 전용.

    동일 발화가 window의 서로 다른 위치에 놓이도록
    짧고 shifted된 window를 만든다.
    """

    raw = [
        {
            "name": "micro_a",
            "start": max(
                0.0,
                gap_start - 0.5,
            ),
            "end": min(
                media_duration,
                gap_start + 9.5,
            ),
        },
        {
            "name": "micro_b",
            "start": max(
                0.0,
                gap_start + 1.0,
            ),
            "end": min(
                media_duration,
                gap_start + 11.0,
            ),
        },
        {
            "name": "micro_c",
            "start": max(
                0.0,
                gap_start + 2.0,
            ),
            "end": min(
                media_duration,
                gap_end + 2.0,
            ),
        },
    ]

    result = []

    seen = set()

    for item in raw:
        if (
            item["end"]
            <= item["start"]
        ):
            continue

        key = (
            round(
                item["start"],
                2,
            ),
            round(
                item["end"],
                2,
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            {
                "name": item["name"],
                "start": item["start"],
                "end": item["end"],
                "duration": (
                    item["end"]
                    - item["start"]
                ),
            }
        )

    return result


def select_micro_segments(
    raw_segments: list[dict],
    gap_start: float,
    gap_end: float,
) -> list[dict]:
    accepted = []

    for segment in raw_segments:
        start = (
            segment["start"]
        )

        end = (
            segment["end"]
        )

        duration = (
            end - start
        )

        if duration <= 0:
            continue

        midpoint = (
            start + end
        ) / 2.0

        overlap = (
            get_interval_overlap(
                start,
                end,
                gap_start,
                gap_end,
            )
        )

        overlap_ratio = (
            overlap / duration
        )

        if (
            midpoint < gap_start
            or midpoint > gap_end
        ) and overlap_ratio < 0.40:
            continue

        clipped = dict(
            segment
        )

        clipped["start"] = max(
            start,
            gap_start,
        )

        clipped["end"] = min(
            end,
            gap_end,
        )

        if (
            clipped["end"]
            - clipped["start"]
            < 0.50
        ):
            continue

        clipped["source"] = (
            "recovery_micro"
        )

        accepted.append(
            clipped
        )

    return accepted


def score_micro_result(
    segments: list[dict],
    gap_start: float,
    gap_end: float,
) -> float:
    if not segments:
        return -999.0

    duration = max(
        gap_end - gap_start,
        0.1,
    )

    coverage = (
        calculate_region_coverage(
            segments,
            gap_start,
            gap_end,
        )
    )

    char_count = sum(
        len(
            normalize_recovery_text(
                item["text"]
            )
        )
        for item in segments
    )

    text_density = min(
        1.0,
        (
            char_count / duration
        ) / 4.0,
    )

    return (
        coverage * 0.60
        + text_density * 0.40
    )


def recover_micro_gap(
    normalized_audio: str,
    gap: dict,
    whisper_engine,
    recovery_dir: str,
    media_duration: float,
    minimum_audio_activity: float = 0.15,
) -> list[dict]:
    gap_start = (
        gap["start"]
    )

    gap_end = (
        gap["end"]
    )

    print()
    print(
        f"  [MICRO GAP] "
        f"{gap_start:.2f} "
        f"~ {gap_end:.2f} "
        f"({gap['duration']:.2f}s)"
    )

    activity = (
        get_audio_activity_ratio(
            wav_file=normalized_audio,
            start_time=gap_start,
            end_time=gap_end,
        )
    )

    print(
        f"    activity: "
        f"{activity:.2f}"
    )

    if (
        activity
        < minimum_audio_activity
    ):
        print(
            "    -> no activity, skip"
        )

        return []

    windows = (
        build_micro_windows(
            gap_start,
            gap_end,
            media_duration,
        )
    )

    attempts = []

    safe_start = (
        make_safe_filename_number(
            gap_start
        )
    )

    for index, window in enumerate(
        windows,
        start=1,
    ):
        print(
            f"    [micro attempt {index}] "
            f"{window['start']:.2f} "
            f"~ {window['end']:.2f}"
        )

        clip_file = os.path.join(
            recovery_dir,
            (
                f"micro_"
                f"{safe_start}_"
                f"{index:02d}.wav"
            ),
        )

        extract_audio_clip(
            input_file=normalized_audio,
            output_file=clip_file,
            start_time=window["start"],
            duration=window["duration"],
        )

        raw = (
            whisper_engine
            .transcribe_file(
                audio_file=clip_file,
                global_offset=window["start"],
                source="recovery_micro",
                audio_duration=window["duration"],
                strict_timestamp=True,
            )
        )

        for item in raw:
            print(
                f"      RAW "
                f"{item['start']:.2f} "
                f"~ {item['end']:.2f} "
                f"{item['text']}"
            )

        accepted = (
            select_micro_segments(
                raw,
                gap_start,
                gap_end,
            )
        )

        score = (
            score_micro_result(
                accepted,
                gap_start,
                gap_end,
            )
        )

        print(
            f"      accepted: "
            f"{len(accepted)}"
        )

        print(
            f"      score   : "
            f"{score:.3f}"
        )

        attempts.append(
            {
                "index": index,
                "segments": accepted,
                "score": score,
            }
        )

        # Micro는 긴 전체 coverage보다
        # '실제 문장을 찾았는지'가 중요.
        if (
            accepted
            and score >= 0.65
        ):
            print(
                "      -> good micro recovery, "
                "early stop"
            )

            break

    attempts.sort(
        key=lambda item: (
            item["score"],
            len(item["segments"]),
        ),
        reverse=True,
    )

    if not attempts:
        return []

    best = (
        attempts[0]
    )

    if not best["segments"]:
        print(
            "    -> micro recovery 실패"
        )

        return []

    print(
        f"    -> micro best attempt: "
        f"{best['index']}"
    )

    return (
        deduplicate_long_candidates(
            best["segments"]
        )
    )


# =========================================================
# LONG + MICRO RECOVERY
# =========================================================

def recover_suspicious_long_segments(
    normalized_audio: str,
    primary_segments: list[dict],
    whisper_engine,
    recovery_dir: str,
    media_duration: float,
    minimum_duration: float = 12.0,
    micro_gap_threshold: float = 2.5,
) -> tuple[
    list[dict],
    int,
    list[dict],
]:
    """
    반환:
        repaired_segments
        long_recovery_count
        handled_micro_intervals

    handled_micro_intervals는 이후 일반 GAP recovery에서
    다시 시도하지 않도록 사용.
    """

    suspicious = (
        detect_suspicious_long_segments(
            primary_segments,
            minimum_duration=minimum_duration,
        )
    )

    print()
    print(
        f"Suspicious long segments: "
        f"{len(suspicious)}"
    )

    replacements = {}

    success_count = 0

    handled_micro_intervals = []

    for segment in suspicious:
        print()
        print(
            "[LONG SEGMENT]"
        )

        print(
            f"  original: "
            f"{segment['start']:.2f} "
            f"~ {segment['end']:.2f} "
            f"({segment['duration']:.2f}s)"
        )

        print(
            f"  text    : "
            f"{segment['text']}"
        )

        candidates = (
            collect_long_candidates(
                normalized_audio=normalized_audio,
                original_segment=segment,
                whisper_engine=whisper_engine,
                recovery_dir=recovery_dir,
                media_duration=media_duration,
            )
        )

        reconstructed = (
            reconstruct_long_segment(
                original_segment=segment,
                candidates=candidates,
            )
        )

        if not reconstructed:
            print(
                "  -> LONG reconstruction 실패"
            )

            print(
                "  -> 기존 PRIMARY 유지"
            )

            continue

        # -------------------------------------------------
        # LONG reconstruction 후 내부 빈 구간 탐지
        # -------------------------------------------------

        micro_gaps = (
            find_uncovered_intervals(
                segments=reconstructed,
                region_start=segment["start"],
                region_end=segment["end"],
                minimum_gap=micro_gap_threshold,
            )
        )

        print(
            f"  -> internal micro gaps: "
            f"{len(micro_gaps)}"
        )

        micro_segments = []

        for micro_gap in micro_gaps:
            # 이 구간은 성공/실패 여부와 무관하게
            # 이미 전용 알고리즘으로 처리했으므로
            # 일반 GAP recovery에서는 제외.
            handled_micro_intervals.append(
                {
                    "start": (
                        micro_gap["start"]
                    ),
                    "end": (
                        micro_gap["end"]
                    ),
                }
            )

            recovered_micro = (
                recover_micro_gap(
                    normalized_audio=normalized_audio,
                    gap=micro_gap,
                    whisper_engine=whisper_engine,
                    recovery_dir=recovery_dir,
                    media_duration=media_duration,
                )
            )

            micro_segments.extend(
                recovered_micro
            )

        final_reconstructed = (
            reconstructed
            + micro_segments
        )

        final_reconstructed = (
            clean_long_timeline(
                final_reconstructed
            )
        )

        print()
        print(
            "  -> FINAL LONG TIMELINE"
        )

        for item in final_reconstructed:
            print(
                f"     "
                f"{item['start']:.2f} "
                f"~ {item['end']:.2f} "
                f"[{item.get('source', 'unknown')}] "
                f"{item['text']}"
            )

        key = (
            segment["start"],
            segment["end"],
            segment["text"],
        )

        replacements[
            key
        ] = final_reconstructed

        success_count += 1

    final_segments = []

    for segment in primary_segments:
        key = (
            segment["start"],
            segment["end"],
            segment["text"],
        )

        if key in replacements:
            final_segments.extend(
                replacements[key]
            )

        else:
            final_segments.append(
                segment
            )

    final_segments.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )

    return (
        final_segments,
        success_count,
        handled_micro_intervals,
    )