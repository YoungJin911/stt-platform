from faster_whisper import WhisperModel


class WhisperEngine:

    def __init__(
        self,
        model_name: str = "turbo",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        print(
            f"Loading Whisper model: "
            f"{model_name}"
        )

        print(
            f"Device             : "
            f"{device}"
        )

        print(
            f"Compute type       : "
            f"{compute_type}"
        )

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        print(
            "Whisper model loaded."
        )

    # =====================================================
    # CANONICAL ASR
    # =====================================================

    def transcribe_file(
        self,
        audio_file: str,
        global_offset: float = 0.0,
        source: str = "primary",
        audio_duration: float | None = None,
        strict_timestamp: bool = False,
    ) -> list[dict]:

        segments, info = self.model.transcribe(
            audio_file,
            language="ko",
            beam_size=5,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

        results = []

        for segment in segments:

            text = (
                segment.text
                .strip()
            )

            if not text:
                continue

            local_start = float(
                segment.start
            )

            local_end = float(
                segment.end
            )

            # ---------------------------------------------
            # 기본 timestamp validation
            # ---------------------------------------------

            if local_start < -0.1:
                print(
                    f"[STT INVALID] "
                    f"negative start: "
                    f"{local_start:.2f}"
                )
                continue

            if local_end <= local_start:
                print(
                    f"[STT INVALID] "
                    f"end <= start: "
                    f"{local_start:.2f} "
                    f"~ {local_end:.2f}"
                )
                continue

            # ---------------------------------------------
            # Media duration validation
            # ---------------------------------------------

            if (
                audio_duration is not None
            ):
                # Primary와 Recovery 모두
                # 파일 범위를 크게 벗어나는 hallucination 차단
                hard_tolerance = (
                    1.0
                    if not strict_timestamp
                    else 3.0
                )

                if (
                    local_start
                    > audio_duration
                    + hard_tolerance
                ):
                    print(
                        f"[STT INVALID] "
                        f"segment start exceeds "
                        f"audio duration: "
                        f"{local_start:.2f} "
                        f"> {audio_duration:.2f}"
                    )

                    continue

                if (
                    local_end
                    > audio_duration
                    + hard_tolerance
                ):
                    # Primary에서는 이런 경우를
                    # hallucination으로 간주하고 폐기
                    if not strict_timestamp:

                        print(
                            f"[PRIMARY INVALID] "
                            f"segment exceeds "
                            f"media duration: "
                            f"{local_start:.2f} "
                            f"~ {local_end:.2f} "
                            f"(media "
                            f"{audio_duration:.2f})"
                        )

                        print(
                            f"  text: {text}"
                        )

                        continue

                    # Recovery에서는 기존 정책 유지
                    print(
                        f"[STT INVALID] "
                        f"segment end exceeds "
                        f"audio duration: "
                        f"{local_end:.2f} "
                        f"> {audio_duration:.2f}"
                    )

                    continue

                # -------------------------------------------------
                # 경미한 overflow만 clamp
                # -------------------------------------------------

                local_start = max(
                    0.0,
                    local_start,
                )

                if (
                    local_end
                    > audio_duration
                ):
                    print(
                        f"[STT CLAMP] "
                        f"segment end: "
                        f"{local_end:.2f} "
                        f"-> {audio_duration:.2f}"
                    )

                    local_end = (
                        audio_duration
                    )

            if local_end <= local_start:
                continue

            results.append(
                {
                    "start": (
                        global_offset
                        + local_start
                    ),
                    "end": (
                        global_offset
                        + local_end
                    ),
                    "text": text,
                    "source": source,
                    "words": [],
                }
            )

        return results

    # =====================================================
    # PRIMARY
    # =====================================================

    def transcribe_primary(
        self,
        audio_file: str,
        media_duration: float,
    ) -> list[dict]:

        print()
        print(
            "Starting full-audio "
            "primary STT..."
        )

        results = (
            self.transcribe_file(
                audio_file=(
                    audio_file
                ),
                global_offset=0.0,
                source="primary",
                audio_duration=(
                    media_duration
                ),
                strict_timestamp=False,
            )
        )

        print(
            f"Primary segments detected: "
            f"{len(results)}"
        )

        return results

    # =====================================================
    # WORD ALIGNMENT ASR
    # =====================================================

    def transcribe_words(
        self,
        audio_file: str,
        global_offset: float = 0.0,
        audio_duration: float | None = None,
    ) -> list[dict]:

        segments, info = (
            self.model.transcribe(
                audio_file,
                language="ko",

                # alignment은 text 생성 품질보다
                # timing 추출 속도가 우선
                beam_size=1,

                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=True,
            )
        )

        words = []

        for segment in segments:

            if not segment.words:
                continue

            for word in segment.words:

                if (
                    word.start is None
                    or
                    word.end is None
                ):
                    continue

                local_start = float(
                    word.start
                )

                local_end = float(
                    word.end
                )

                text = str(
                    word.word
                ).strip()

                if not text:
                    continue

                if local_end <= local_start:
                    continue

                if (
                    audio_duration
                    is not None
                ):

                    if (
                        local_start
                        > audio_duration
                        + 1.0
                    ):
                        continue

                    if (
                        local_end
                        > audio_duration
                        + 1.0
                    ):
                        continue

                    local_start = max(
                        0.0,
                        local_start,
                    )

                    local_end = min(
                        audio_duration,
                        local_end,
                    )

                if local_end <= local_start:
                    continue

                probability = None

                if (
                    word.probability
                    is not None
                ):
                    probability = float(
                        word.probability
                    )

                words.append(
                    {
                        "start": (
                            global_offset
                            + local_start
                        ),
                        "end": (
                            global_offset
                            + local_end
                        ),
                        "word": text,
                        "probability": (
                            probability
                        ),
                    }
                )

        return words