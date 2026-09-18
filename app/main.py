import json
import os
import shutil
import time

from app.media import (
    get_media_duration,
    normalize_audio,
)

from app.stt import (
    WhisperEngine,
)

from app.filters import (
    remove_primary_repetitions,
    remove_merge_boundary_duplicates,
)

from app.recovery import (
    recover_suspicious_long_segments,
    detect_transcript_gaps,
    recover_gaps,
)

from app.alignment import (
    get_alignment_target_indices,
    attach_word_timestamps_v3,
)

from app.alignment_secondary import (
    build_secondary_passes as build_alignment_secondary_passes,
)

from app.omission_probe import (
    build_omission_probe_passes,
)

from app.omission_v2 import (
    detect_omission_candidates_v2,
    apply_verified_omissions,
)

from app.omission_realign import (
    realign_applied_omissions,
)

from app.correction import (
    correct_segments,
)

from app.subtitle import (
    format_subtitle_segments,
)

from app.srt import (
    export_srt,
    seconds_to_srt_time,
)


# =========================================================
# PATH
# =========================================================

INPUT_FILE = "input/test.mp4"

NORMALIZED_AUDIO = "work/normalized.wav"

RECOVERY_DIR = "work/recovery"
ALIGN_DIR = "work/alignment"

# Stable alignment secondary ASR
ALIGNMENT_SECONDARY_DIR = (
    "work/alignment_secondary"
)

# New omission-only local ASR
OMISSION_PROBE_DIR = (
    "work/omission_probe"
)

OMISSION_VERIFY_DIR = (
    "work/omission_v2"
)

OMISSION_REALIGN_DIR = (
    "work/omission_realign"
)


OUTPUT_RAW_JSON = (
    "output/raw_transcript.json"
)

OUTPUT_CORRECTED_JSON = (
    "output/corrected_transcript.json"
)

OUTPUT_CORRECTION_LOG = (
    "output/correction_log.json"
)

OUTPUT_REVIEW_CANDIDATES = (
    "output/review_candidates.json"
)

OUTPUT_OMISSION_CANDIDATES = (
    "output/omission_candidates.json"
)

OUTPUT_OMISSION_LOG = (
    "output/omission_log.json"
)

OUTPUT_OMISSION_REVIEW = (
    "output/omission_review.json"
)

OUTPUT_OMISSION_REALIGN_LOG = (
    "output/omission_realign_log.json"
)

OUTPUT_ALIGNMENT_SECONDARY_PASSES = (
    "output/secondary_passes.json"
)

OUTPUT_OMISSION_PROBE_PASSES = (
    "output/omission_probe_passes.json"
)

OUTPUT_RAW_SRT = (
    "output/result_raw.srt"
)

OUTPUT_CORRECTED_SRT = (
    "output/result_corrected.srt"
)


# =========================================================
# MODEL
# =========================================================

WHISPER_MODEL = "turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "int8_float16"


# =========================================================
# RECOVERY
# =========================================================

MIN_RECOVERY_GAP = 3.0
MIN_AUDIO_ACTIVITY_RATIO = 0.15

GOOD_RECOVERY_COVERAGE = 0.80
MAX_INTERNAL_RECOVERY_GAP = 2.0
MAX_RECOVERY_ATTEMPTS = 4

SUSPICIOUS_SEGMENT_DURATION = 12.0
MICRO_GAP_THRESHOLD = 2.5


# =========================================================
# SUBTITLE / ALIGNMENT
# =========================================================

SUBTITLE_MAX_DURATION = 6.0
SUBTITLE_TARGET_CHARS = 24
SUBTITLE_MAX_CHARS = 42

ALIGNMENT_MAX_BATCH_DURATION = 30.0
ALIGNMENT_MAX_TARGET_GAP = 3.0


# =========================================================
# DIRECTORY
# =========================================================

def prepare_work_directory():

    os.makedirs(
        "work",
        exist_ok=True,
    )

    os.makedirs(
        "output",
        exist_ok=True,
    )

    for directory in [
        RECOVERY_DIR,
        ALIGN_DIR,
        ALIGNMENT_SECONDARY_DIR,
        OMISSION_PROBE_DIR,
        OMISSION_VERIFY_DIR,
        OMISSION_REALIGN_DIR,
    ]:

        if os.path.exists(
            directory
        ):

            shutil.rmtree(
                directory
            )

        os.makedirs(
            directory,
            exist_ok=True,
        )


# =========================================================
# JSON
# =========================================================

def save_json(
    data,
    output_file: str,
):

    directory = os.path.dirname(
        output_file
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# MAIN
# =========================================================

def main():

    prepare_work_directory()

    total_start = (
        time.perf_counter()
    )

    # =====================================================
    # MEDIA
    # =====================================================

    media_duration = (
        get_media_duration(
            INPUT_FILE
        )
    )

    print()

    print(
        f"Media duration     : "
        f"{media_duration:.2f} sec"
    )

    # =====================================================
    # AUDIO NORMALIZE
    # =====================================================

    normalize_start = (
        time.perf_counter()
    )

    normalize_audio(
        INPUT_FILE,
        NORMALIZED_AUDIO,
    )

    normalize_time = (
        time.perf_counter()
        -
        normalize_start
    )

    # =====================================================
    # MODEL
    # =====================================================

    whisper = (
        WhisperEngine(
            model_name=(
                WHISPER_MODEL
            ),
            device=(
                WHISPER_DEVICE
            ),
            compute_type=(
                WHISPER_COMPUTE_TYPE
            ),
        )
    )

    # =====================================================
    # PRIMARY STT
    # =====================================================

    primary_start = (
        time.perf_counter()
    )

    raw_primary = (
        whisper.transcribe_primary(
            audio_file=(
                NORMALIZED_AUDIO
            ),
            media_duration=(
                media_duration
            ),
        )
    )

    primary_time = (
        time.perf_counter()
        -
        primary_start
    )

    (
        primary_segments,
        primary_filtered,
    ) = (
        remove_primary_repetitions(
            raw_primary
        )
    )

    # =====================================================
    # LONG + MICRO RECOVERY
    # =====================================================

    long_start = (
        time.perf_counter()
    )

    (
        repaired_primary,
        long_recovery_count,
        handled_micro_intervals,
    ) = (
        recover_suspicious_long_segments(
            normalized_audio=(
                NORMALIZED_AUDIO
            ),
            primary_segments=(
                primary_segments
            ),
            whisper_engine=(
                whisper
            ),
            recovery_dir=(
                RECOVERY_DIR
            ),
            media_duration=(
                media_duration
            ),
            minimum_duration=(
                SUSPICIOUS_SEGMENT_DURATION
            ),
            micro_gap_threshold=(
                MICRO_GAP_THRESHOLD
            ),
        )
    )

    long_time = (
        time.perf_counter()
        -
        long_start
    )

    # =====================================================
    # GAP DETECTION
    # =====================================================

    gaps = (
        detect_transcript_gaps(
            segments=(
                repaired_primary
            ),
            media_duration=(
                media_duration
            ),
            min_gap=(
                MIN_RECOVERY_GAP
            ),
            ignored_intervals=(
                handled_micro_intervals
            ),
        )
    )

    # =====================================================
    # GAP RECOVERY
    # =====================================================

    gap_start = (
        time.perf_counter()
    )

    recovered_segments = (
        recover_gaps(
            normalized_audio=(
                NORMALIZED_AUDIO
            ),
            gaps=(
                gaps
            ),
            whisper_engine=(
                whisper
            ),
            recovery_dir=(
                RECOVERY_DIR
            ),
            media_duration=(
                media_duration
            ),
            minimum_active_ratio=(
                MIN_AUDIO_ACTIVITY_RATIO
            ),
            good_coverage_ratio=(
                GOOD_RECOVERY_COVERAGE
            ),
            maximum_internal_gap=(
                MAX_INTERNAL_RECOVERY_GAP
            ),
            max_attempts=(
                MAX_RECOVERY_ATTEMPTS
            ),
        )
    )

    gap_time = (
        time.perf_counter()
        -
        gap_start
    )

    # =====================================================
    # MERGE
    # =====================================================

    merged_segments = (
        repaired_primary
        +
        recovered_segments
    )

    merged_segments.sort(
        key=lambda item: (
            item[
                "start"
            ],
            item[
                "end"
            ],
        )
    )

    (
        final_raw_segments,
        merge_filtered,
    ) = (
        remove_merge_boundary_duplicates(
            merged_segments
        )
    )

    # =====================================================
    # ALIGNMENT TARGETS
    # =====================================================

    alignment_target_indices = (
        get_alignment_target_indices(
            segments=(
                final_raw_segments
            ),
            max_cue_duration=(
                SUBTITLE_MAX_DURATION
            ),
            max_chars=(
                SUBTITLE_MAX_CHARS
            ),
        )
    )

    # =====================================================
    # ALIGNMENT SECONDARY ASR
    #
    # Full-audio word timeline, one ASR pass.
    # Omission detection is independent and uses omission_probe.py.
    # =====================================================

    secondary_start = (
        time.perf_counter()
    )

    alignment_secondary_passes = (
        build_alignment_secondary_passes(
            segments=(
                final_raw_segments
            ),
            alignment_target_indices=(
                alignment_target_indices
            ),
            normalized_audio=(
                NORMALIZED_AUDIO
            ),
            whisper=(
                whisper
            ),
            work_dir=(
                ALIGNMENT_SECONDARY_DIR
            ),
            media_duration=(
                media_duration
            ),
            max_batch_duration=(
                ALIGNMENT_MAX_BATCH_DURATION
            ),
            max_target_gap=(
                ALIGNMENT_MAX_TARGET_GAP
            ),
        )
    )

    secondary_time = (
        time.perf_counter()
        -
        secondary_start
    )

    save_json(
        alignment_secondary_passes,
        OUTPUT_ALIGNMENT_SECONDARY_PASSES,
    )

    # =====================================================
    # SUBTITLE WORD ALIGNMENT
    # =====================================================

    alignment_start = (
        time.perf_counter()
    )

    aligned_segments = (
        attach_word_timestamps_v3(
            segments=(
                final_raw_segments
            ),
            secondary_passes=(
                alignment_secondary_passes
            ),
            normalized_audio=(
                NORMALIZED_AUDIO
            ),
            whisper=(
                whisper
            ),
            alignment_dir=(
                ALIGN_DIR
            ),
            media_duration=(
                media_duration
            ),
            max_cue_duration=(
                SUBTITLE_MAX_DURATION
            ),
            max_chars=(
                SUBTITLE_MAX_CHARS
            ),
        )
    )

    alignment_time = (
        time.perf_counter()
        -
        alignment_start
    )

    # =====================================================
    # OMISSION-SPECIFIC LOCAL ASR
    #
    # This is now computationally independent from the
    # alignment secondary passes.
    # =====================================================

    omission_probe_start = (
        time.perf_counter()
    )

    omission_probe_passes = (
        build_omission_probe_passes(
            segments=(
                aligned_segments
            ),
            normalized_audio=(
                NORMALIZED_AUDIO
            ),
            whisper=(
                whisper
            ),
            work_dir=(
                OMISSION_PROBE_DIR
            ),
            media_duration=(
                media_duration
            ),
        )
    )

    omission_probe_time = (
        time.perf_counter()
        -
        omission_probe_start
    )

    save_json(
        omission_probe_passes,
        OUTPUT_OMISSION_PROBE_PASSES,
    )

    # =====================================================
    # OMISSION DETECTION
    #
    # omission_v2.py remains unchanged.
    # It receives omission-specific evidence in the same
    # pass schema that it already understands.
    # =====================================================

    omission_start = (
        time.perf_counter()
    )

    omission_candidates = (
        detect_omission_candidates_v2(
            segments=(
                aligned_segments
            ),
            secondary_passes=(
                omission_probe_passes
            ),
            normalized_audio=(
                NORMALIZED_AUDIO
            ),
            whisper_engine=(
                whisper
            ),
            verify_dir=(
                OMISSION_VERIFY_DIR
            ),
            media_duration=(
                media_duration
            ),
        )
    )

    save_json(
        omission_candidates,
        OUTPUT_OMISSION_CANDIDATES,
    )

    # =====================================================
    # OMISSION AUTO APPLY
    # =====================================================

    (
        canonical_segments,
        omission_log,
        omission_review,
    ) = (
        apply_verified_omissions(
            segments=(
                aligned_segments
            ),
            candidates=(
                omission_candidates
            ),
        )
    )

    save_json(
        omission_log,
        OUTPUT_OMISSION_LOG,
    )

    save_json(
        omission_review,
        OUTPUT_OMISSION_REVIEW,
    )

    omission_time = (
        time.perf_counter()
        -
        omission_start
    )

    # =====================================================
    # OMISSION LOCAL RE-ALIGNMENT
    # =====================================================

    omission_realign_start = (
        time.perf_counter()
    )

    (
        canonical_segments,
        omission_realign_log,
    ) = (
        realign_applied_omissions(
            segments=(
                canonical_segments
            ),
            normalized_audio=(
                NORMALIZED_AUDIO
            ),
            whisper_engine=(
                whisper
            ),
            work_dir=(
                OMISSION_REALIGN_DIR
            ),
            media_duration=(
                media_duration
            ),
        )
    )

    omission_realign_time = (
        time.perf_counter()
        -
        omission_realign_start
    )

    save_json(
        omission_realign_log,
        OUTPUT_OMISSION_REALIGN_LOG,
    )

    # =====================================================
    # RAW CANONICAL JSON
    # =====================================================

    save_json(
        canonical_segments,
        OUTPUT_RAW_JSON,
    )

    # =====================================================
    # RAW SUBTITLE
    # =====================================================

    raw_subtitle_segments = (
        format_subtitle_segments(
            canonical_segments,
            max_cue_duration=(
                SUBTITLE_MAX_DURATION
            ),
            target_chars=(
                SUBTITLE_TARGET_CHARS
            ),
            max_chars=(
                SUBTITLE_MAX_CHARS
            ),
        )
    )

    export_srt(
        raw_subtitle_segments,
        OUTPUT_RAW_SRT,
    )

    # =====================================================
    # CORRECTION
    # =====================================================

    (
        corrected_segments,
        correction_log,
        review_candidates,
    ) = (
        correct_segments(
            canonical_segments
        )
    )

    save_json(
        corrected_segments,
        OUTPUT_CORRECTED_JSON,
    )

    save_json(
        correction_log,
        OUTPUT_CORRECTION_LOG,
    )

    save_json(
        review_candidates,
        OUTPUT_REVIEW_CANDIDATES,
    )

    # =====================================================
    # CORRECTED SUBTITLE
    # =====================================================

    corrected_subtitle_segments = (
        format_subtitle_segments(
            corrected_segments,
            max_cue_duration=(
                SUBTITLE_MAX_DURATION
            ),
            target_chars=(
                SUBTITLE_TARGET_CHARS
            ),
            max_chars=(
                SUBTITLE_MAX_CHARS
            ),
        )
    )

    export_srt(
        corrected_subtitle_segments,
        OUTPUT_CORRECTED_SRT,
    )

    # =====================================================
    # APPLIED OMISSIONS LOG
    # =====================================================

    print()

    print(
        "========== APPLIED OMISSIONS =========="
    )

    if not omission_log:

        print(
            "No omissions applied."
        )

    else:

        for item in (
            omission_log
        ):

            print()

            print(
                f"["
                f"{seconds_to_srt_time(item['candidate_start'])}"
                f"]"
            )

            print(
                f"  MODE   : "
                f"{item['mode']}"
            )

            print(
                f"  TEXT   : "
                f"{item['inserted_text']}"
            )

            probability = (
                item.get(
                    "probability"
                )
            )

            if probability is not None:

                print(
                    f"  CONF   : "
                    f"{float(probability):.3f}"
                )

            print(
                f"  VERIFY : "
                f"{item['verification']}"
            )

            print(
                f"  BEFORE : "
                f"{item['before']}"
            )

            print(
                f"  AFTER  : "
                f"{item['after']}"
            )

    # =====================================================
    # OMISSION REALIGN LOG
    # =====================================================

    print()

    print(
        "========== OMISSION REALIGN =========="
    )

    if not omission_realign_log:

        print(
            "No omission re-alignment targets."
        )

    else:

        for item in (
            omission_realign_log
        ):

            print()

            print(
                f"["
                f"{seconds_to_srt_time(item['start'])}"
                f"]"
            )

            print(
                f"  SUCCESS : "
                f"{item['success']}"
            )

            print(
                f"  SIM     : "
                f"{item['alignment_similarity']:.3f}"
            )

            print(
                f"  WORDS   : "
                f"{item['word_count']}"
            )

            print(
                f"  TEXT    : "
                f"{item['text']}"
            )

            print(
                f"  ALIGN   : "
                f"{item['alignment_text']}"
            )

    # =====================================================
    # CORRECTIONS
    # =====================================================

    print()

    print(
        "========== CORRECTIONS =========="
    )

    if not correction_log:

        print(
            "No automatic corrections."
        )

    else:

        for item in (
            correction_log
        ):

            print()

            print(
                f"["
                f"{seconds_to_srt_time(item['start'])}"
                f"]"
            )

            print(
                f"  RAW : "
                f"{item['raw_text']}"
            )

            print(
                f"  FIX : "
                f"{item['corrected_text']}"
            )

            for rule in (
                item.get(
                    "rules",
                    [],
                )
            ):

                print(
                    f"  RULE: "
                    f"{rule['from']} "
                    f"-> "
                    f"{rule['to']}"
                )

    # =====================================================
    # REVIEW CANDIDATES
    # =====================================================

    print()

    print(
        "========== REVIEW CANDIDATES =========="
    )

    if not review_candidates:

        print(
            "No review candidates."
        )

    else:

        for item in (
            review_candidates
        ):

            print()

            print(
                f"["
                f"{seconds_to_srt_time(item['start'])}"
                f"] "
                f"{item['type']}"
            )

            print(
                f"  suspect   : "
                f"{item['suspect']}"
            )

            if item.get(
                "candidate"
            ):

                print(
                    f"  candidate : "
                    f"{item['candidate']}"
                )

            print(
                f"  text      : "
                f"{item.get('raw_text', '')}"
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    total_time = (
        time.perf_counter()
        -
        total_start
    )

    print()

    print(
        "========================================"
    )

    print(
        "         Processing Summary"
    )

    print(
        "========================================"
    )

    print(
        f"Model              : "
        f"{WHISPER_MODEL}"
    )

    print(
        f"Device             : "
        f"{WHISPER_DEVICE}"
    )

    print(
        f"Compute type       : "
        f"{WHISPER_COMPUTE_TYPE}"
    )

    print(
        f"Media duration     : "
        f"{media_duration:.2f} sec"
    )

    print(
        f"Audio normalize    : "
        f"{normalize_time:.2f} sec"
    )

    print(
        f"Primary STT        : "
        f"{primary_time:.2f} sec"
    )

    print(
        f"Long+Micro         : "
        f"{long_time:.2f} sec"
    )

    print(
        f"Gap recovery       : "
        f"{gap_time:.2f} sec"
    )

    print(
        f"Alignment secondary: "
        f"{secondary_time:.2f} sec"
    )

    print(
        f"Word alignment     : "
        f"{alignment_time:.2f} sec"
    )

    print(
        f"Omission probe ASR : "
        f"{omission_probe_time:.2f} sec"
    )

    print(
        f"Omission v2        : "
        f"{omission_time:.2f} sec"
    )

    print(
        f"Omission realign   : "
        f"{omission_realign_time:.2f} sec"
    )

    print(
        f"Omissions applied  : "
        f"{len(omission_log)}"
    )

    print(
        f"Omission realigned : "
        f"{sum(1 for item in omission_realign_log if item['success'])}"
    )

    print(
        f"Corrections        : "
        f"{len(correction_log)}"
    )

    print(
        f"Review candidates  : "
        f"{len(review_candidates)}"
    )

    print(
        f"Alignment passes   : "
        f"{len(alignment_secondary_passes)}"
    )

    print(
        f"Omission probes    : "
        f"{len(omission_probe_passes)}"
    )

    print(
        f"Raw segments       : "
        f"{len(canonical_segments)}"
    )

    print(
        f"Raw subtitle cues  : "
        f"{len(raw_subtitle_segments)}"
    )

    print(
        f"Corrected cues     : "
        f"{len(corrected_subtitle_segments)}"
    )

    print(
        f"Total processing   : "
        f"{total_time:.2f} sec"
    )

    if total_time > 0:

        print(
            f"Speed              : "
            f"{media_duration / total_time:.2f}"
            f"x realtime"
        )

    # =====================================================
    # OUTPUT PATH
    # =====================================================

    print()

    print(
        f"RAW JSON           : "
        f"{OUTPUT_RAW_JSON}"
    )

    print(
        f"Corrected JSON     : "
        f"{OUTPUT_CORRECTED_JSON}"
    )

    print(
        f"Correction log     : "
        f"{OUTPUT_CORRECTION_LOG}"
    )

    print(
        f"Review candidates  : "
        f"{OUTPUT_REVIEW_CANDIDATES}"
    )

    print(
        f"Omission candidates: "
        f"{OUTPUT_OMISSION_CANDIDATES}"
    )

    print(
        f"Omission log       : "
        f"{OUTPUT_OMISSION_LOG}"
    )

    print(
        f"Omission review    : "
        f"{OUTPUT_OMISSION_REVIEW}"
    )

    print(
        f"Omission realign   : "
        f"{OUTPUT_OMISSION_REALIGN_LOG}"
    )

    print(
        f"Alignment passes   : "
        f"{OUTPUT_ALIGNMENT_SECONDARY_PASSES}"
    )

    print(
        f"Omission probes    : "
        f"{OUTPUT_OMISSION_PROBE_PASSES}"
    )

    print(
        f"RAW SRT            : "
        f"{OUTPUT_RAW_SRT}"
    )

    print(
        f"Corrected SRT      : "
        f"{OUTPUT_CORRECTED_SRT}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
