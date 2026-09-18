from faster_whisper import WhisperModel


def main():
    input_file = "work/chunks/chunk_0005.wav"

    model = WhisperModel(
        "turbo",
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(
        input_file,
        language="ko",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False
    )

    print(
        f"Language: {info.language}"
    )

    print(
        f"Language probability: "
        f"{info.language_probability:.2f}"
    )

    print()

    for segment in segments:
        print(
            f"[{segment.start:.2f}s "
            f"-> {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )


if __name__ == "__main__":
    main()