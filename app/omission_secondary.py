import copy
import re


# =========================================================
# PHASE 3-C.2
# FULL-AUDIO SECONDARY WORD TIMELINE
# =========================================================
#
# Goal
# ----
# Replace the 18-batch alignment secondary ASR with a single
# full-audio word-timestamp pass.
#
# Important:
# - Primary STT is NOT changed.
# - Omission detection is NOT changed.
# - omission_probe.py keeps handling omission-specific local ASR.
# - alignment.py keeps consuming the same secondary_passes schema.
#
# Why this experiment?
# --------------------
# Phase 3-C.1 showed that removing coverage hurts subtitle quality:
#
#   global words        701 -> 611
#   token match ratio   90.8% -> 79.2%
#
# So this phase preserves ~100% audio coverage and changes only:
#
#   18 ASR calls -> 1 ASR call
#
# Expected benefit:
# - remove repeated batch extraction overhead
# - remove repeated decoder startup/teardown
# - remove inter-batch overlap
# - simplify global word timeline
#
# Regression risk:
# - Whisper may segment a long full-audio word pass differently.
# - Therefore this must be measured against the stable baseline.
# =========================================================


# =========================================================
# TEXT HELPERS
# =========================================================

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


# =========================================================
# CONTEXT
# =========================================================

def all_segment_indices(
    segments: list[dict],
) -> list[int]:

    return list(
        range(
            len(
                segments
            )
        )
    )


# =========================================================
# FULL-AUDIO SECONDARY PASS
# =========================================================

def build_secondary_passes(
    segments: list[dict],
    alignment_target_indices: list[int],
    normalized_audio: str,
    whisper,
    work_dir: str,
    media_duration: float,
    max_batch_duration: float,
    max_target_gap: float,
) -> list[dict]:
    """
    Build one full-audio secondary word-timestamp pass.

    Signature intentionally matches the previous batch-based
    implementation so main.py does not need to change.

    max_batch_duration and max_target_gap are intentionally unused
    in Phase 3-C.2; they are kept only for interface compatibility.
    """

    del work_dir
    del max_batch_duration
    del max_target_gap

    context_segment_indices = (
        all_segment_indices(
            segments
        )
    )

    alignment_target_indices = sorted(
        set(
            int(
                index
            )
            for index in alignment_target_indices
            if (
                0
                <=
                int(
                    index
                )
                <
                len(
                    segments
                )
            )
        )
    )

    print()

    print(
        "Starting full-audio alignment secondary ASR..."
    )

    print(
        f"  total segments       : "
        f"{len(segments)}"
    )

    print(
        f"  alignment targets    : "
        f"{len(alignment_target_indices)}"
    )

    print(
        f"  secondary passes     : 1"
    )

    print(
        f"  full audio coverage  : "
        f"{media_duration:.2f} sec"
    )

    # =====================================================
    # IMPORTANT
    #
    # This is intentionally ONE call over normalized.wav.
    #
    # Primary uses its own transcribe_primary() path with
    # word_timestamps disabled, so Primary segmentation remains
    # unchanged.
    # =====================================================

    words = (
        whisper.transcribe_words(
            audio_file=(
                normalized_audio
            ),
            global_offset=(
                0.0
            ),
            audio_duration=(
                media_duration
            ),
        )
    )

    # =====================================================
    # SANITY FILTER
    # =====================================================
    #
    # Keep only valid words inside media duration.
    # Do not mutate text/probabilities.
    # =====================================================

    filtered_words = []

    invalid_words = 0

    for word in words:

        try:
            start = float(
                word.get(
                    "start",
                    0.0,
                )
            )

            end = float(
                word.get(
                    "end",
                    start,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid_words += 1
            continue

        text = str(
            word.get(
                "word",
                "",
            )
        ).strip()

        if not text:

            invalid_words += 1
            continue

        if end < 0.0:

            invalid_words += 1
            continue

        if start > media_duration:

            invalid_words += 1
            continue

        # Clamp only tiny timestamp overflow.
        start = max(
            0.0,
            min(
                start,
                media_duration,
            ),
        )

        end = max(
            start,
            min(
                end,
                media_duration,
            ),
        )

        cleaned = copy.deepcopy(
            word
        )

        cleaned[
            "start"
        ] = (
            start
        )

        cleaned[
            "end"
        ] = (
            end
        )

        filtered_words.append(
            cleaned
        )

    filtered_words.sort(
        key=lambda item: (
            float(
                item.get(
                    "start",
                    0.0,
                )
            ),
            float(
                item.get(
                    "end",
                    0.0,
                )
            ),
        )
    )

    print(
        f"  words detected       : "
        f"{len(words)}"
    )

    print(
        f"  valid words          : "
        f"{len(filtered_words)}"
    )

    print(
        f"  invalid words        : "
        f"{invalid_words}"
    )

    if filtered_words:

        first_word = (
            filtered_words[
                0
            ]
        )

        last_word = (
            filtered_words[
                -1
            ]
        )

        print(
            f"  first word           : "
            f"{float(first_word['start']):.2f} "
            f"{first_word.get('word', '')}"
        )

        print(
            f"  last word            : "
            f"{float(last_word['end']):.2f} "
            f"{last_word.get('word', '')}"
        )

    secondary_pass = {
        "batch_number": (
            1
        ),

        "pass_type": (
            "full_audio_alignment"
        ),

        "start": (
            0.0
        ),

        "end": (
            float(
                media_duration
            )
        ),

        # Preserve schema expected by alignment.py.
        "secondary_target_indices": (
            copy.deepcopy(
                context_segment_indices
            )
        ),

        "alignment_target_indices": (
            copy.deepcopy(
                alignment_target_indices
            )
        ),

        "target_reasons": {
            str(
                index
            ): [
                "alignment_target"
            ]
            for index
            in alignment_target_indices
        },

        "context_segment_indices": (
            copy.deepcopy(
                context_segment_indices
            )
        ),

        "words": (
            copy.deepcopy(
                filtered_words
            )
        ),

        "text": (
            words_to_text(
                filtered_words
            )
        ),
    }

    print()

    print(
        "Full-audio alignment secondary ASR generation:"
    )

    print(
        f"  alignment targets    : "
        f"{len(alignment_target_indices)}"
    )

    print(
        f"  secondary passes     : 1"
    )

    print(
        f"  secondary audio      : "
        f"{media_duration:.2f} sec"
    )

    print(
        f"  words                : "
        f"{len(filtered_words)}"
    )

    return [
        secondary_pass
    ]
