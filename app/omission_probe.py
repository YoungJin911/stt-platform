import copy


# =========================================================
# PURPOSE
# =========================================================
#
# Phase 3-A: architecture-only separation.
#
# The existing secondary ASR passes are currently proven stable for
# subtitle word alignment and omission detection.
#
# This module creates a separate omission-evidence data path WITHOUT
# changing ASR coverage yet.
#
# For this first step, omission probe passes are intentionally a
# deep-copied pass-through of the stable alignment secondary passes.
#
# Why do this first?
# - alignment.py can keep consuming the stable secondary evidence
# - omission_v2.py receives its own logical input
# - later we can replace this implementation with local omission-only
#   ASR windows without changing main.py/alignment.py/omission_v2.py
# - regression should be essentially zero in this phase
#
# IMPORTANT:
# This step is NOT expected to improve processing speed.
# It is a dependency-separation refactor.
# =========================================================


def build_omission_probe_passes(
    segments: list[dict],
    alignment_secondary_passes: list[dict],
) -> list[dict]:
    """
    Build omission-specific evidence passes.

    Phase 3-A compatibility mode:
        copy the already generated alignment secondary passes 1:1.

    The returned schema intentionally remains compatible with
    detect_omission_candidates_v2(), which expects:
        - batch_number
        - start
        - end
        - context_segment_indices
        - words
        - text

    Extra metadata is allowed and ignored by omission_v2.py.
    """

    omission_probe_passes = []

    for secondary_pass in alignment_secondary_passes:

        probe_pass = copy.deepcopy(
            secondary_pass
        )

        probe_pass[
            "probe_source"
        ] = (
            "alignment_secondary_compat"
        )

        omission_probe_passes.append(
            probe_pass
        )

    print()

    print(
        "Omission probe preparation:"
    )

    print(
        f"  mode                : "
        f"compatibility pass-through"
    )

    print(
        f"  canonical segments  : "
        f"{len(segments)}"
    )

    print(
        f"  alignment passes    : "
        f"{len(alignment_secondary_passes)}"
    )

    print(
        f"  omission probes     : "
        f"{len(omission_probe_passes)}"
    )

    print(
        "  extra ASR           : 0"
    )

    return omission_probe_passes
