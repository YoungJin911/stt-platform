import os


def seconds_to_srt_time(
    seconds: float,
) -> str:

    seconds = max(
        0.0,
        seconds,
    )

    total_ms = round(
        seconds * 1000
    )

    hours = (
        total_ms
        // 3_600_000
    )

    remainder = (
        total_ms
        % 3_600_000
    )

    minutes = (
        remainder
        // 60_000
    )

    remainder %= 60_000

    secs = (
        remainder
        // 1000
    )

    milliseconds = (
        remainder
        % 1000
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def export_srt(
    segments: list[dict],
    output_path: str,
) -> None:

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    valid_segments = []

    for segment in segments:

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        text = (
            segment["text"]
            .strip()
        )

        if not text:
            continue

        if end <= start:

            print(
                f"[SRT SKIP] "
                f"{start:.2f} "
                f"~ {end:.2f}"
            )

            continue

        valid_segments.append(
            segment
        )

    valid_segments.sort(
        key=lambda item: (
            item["start"],
            item["end"],
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        for index, segment in enumerate(
            valid_segments,
            start=1,
        ):

            start = (
                seconds_to_srt_time(
                    segment["start"]
                )
            )

            end = (
                seconds_to_srt_time(
                    segment["end"]
                )
            )

            text = (
                segment["text"]
                .strip()
            )

            file.write(
                f"{index}\n"
            )

            file.write(
                f"{start} --> "
                f"{end}\n"
            )

            file.write(
                f"{text}\n\n"
            )