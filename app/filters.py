import re

from difflib import SequenceMatcher


def normalize_text(
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


def get_text_similarity(
    text_a: str,
    text_b: str,
) -> float:
    a = normalize_text(
        text_a
    )

    b = normalize_text(
        text_b
    )

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def is_tail_duplicate(
    previous_text: str,
    current_text: str,
) -> bool:
    """
    예:

    previous:
        규명할 계획입니다.

    current:
        계획입니다.

    → True
    """

    previous = normalize_text(
        previous_text
    )

    current = normalize_text(
        current_text
    )

    if not previous or not current:
        return False

    if previous == current:
        return True

    if (
        len(current) <= 20
        and
        previous.endswith(
            current
        )
    ):
        return True

    similarity = (
        get_text_similarity(
            previous,
            current,
        )
    )

    if (
        len(current) <= 20
        and
        similarity >= 0.80
    ):
        return True

    return False


def remove_primary_repetitions(
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    전체 Primary STT에서
    바로 앞 segment와 명확히 반복되는
    짧은 hallucination만 제거한다.
    """

    sorted_segments = sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    valid = []
    filtered = []

    for segment in sorted_segments:

        if not valid:
            valid.append(
                segment
            )
            continue

        previous = valid[-1]

        gap = (
            segment["start"]
            - previous["end"]
        )

        if gap <= 0.5:

            if is_tail_duplicate(
                previous["text"],
                segment["text"],
            ):
                removed = dict(
                    segment
                )

                removed["reason"] = (
                    "primary_repetition"
                )

                filtered.append(
                    removed
                )

                continue

        valid.append(
            segment
        )

    return (
        valid,
        filtered,
    )


def remove_merge_boundary_duplicates(
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Primary/Recovery 병합 경계에서
    발생하는 반복 문장을 제거한다.

    예:

    PRIMARY
        규명할 계획입니다.

    RECOVERY
        계획입니다.

    → RECOVERY 제거
    """

    sorted_segments = sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    valid = []
    filtered = []

    for current in sorted_segments:

        if not valid:
            valid.append(
                current
            )
            continue

        previous = valid[-1]

        gap = (
            current["start"]
            - previous["end"]
        )

        if gap <= 0.6:

            if is_tail_duplicate(
                previous["text"],
                current["text"],
            ):
                removed = dict(
                    current
                )

                removed["reason"] = (
                    "merge_boundary_duplicate"
                )

                filtered.append(
                    removed
                )

                continue

        valid.append(
            current
        )

    return (
        valid,
        filtered,
    )


def deduplicate_local_segments(
    segments: list[dict],
) -> list[dict]:
    """
    Long-segment recovery처럼
    overlap window 여러 개에서 나온
    local recovery 결과 중 중복 제거.
    """

    sorted_segments = sorted(
        segments,
        key=lambda item: (
            item["start"],
            item["end"],
        ),
    )

    valid = []

    for current in sorted_segments:

        duplicate = False

        for previous in reversed(
            valid[-5:]
        ):
            time_distance = abs(
                current["start"]
                - previous["start"]
            )

            if time_distance > 3.0:
                break

            similarity = (
                get_text_similarity(
                    previous["text"],
                    current["text"],
                )
            )

            if similarity >= 0.78:
                duplicate = True
                break

            previous_text = normalize_text(
                previous["text"]
            )

            current_text = normalize_text(
                current["text"]
            )

            if (
                len(previous_text) >= 6
                and
                len(current_text) >= 6
            ):
                if (
                    previous_text
                    in current_text
                    or
                    current_text
                    in previous_text
                ):
                    duplicate = True
                    break

        if not duplicate:
            valid.append(
                current
            )

    return valid