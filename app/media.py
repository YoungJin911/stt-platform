import os
import subprocess


def get_media_duration(
    file_path: str,
) -> float:
    """
    ffprobe를 사용해 미디어 길이를 초 단위로 반환한다.
    """

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(
        result.stdout.strip()
    )


def normalize_audio(
    input_file: str,
    output_file: str,
) -> None:
    """
    입력 미디어를 Whisper용 WAV로 변환한다.

    출력:
    - 16 kHz
    - Mono
    - PCM signed 16-bit
    """

    output_dir = os.path.dirname(
        output_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    command = [
        "ffmpeg",
        "-y",
        "-v", "error",

        "-i", input_file,

        "-vn",

        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",

        output_file,
    ]

    subprocess.run(
        command,
        check=True,
    )


def extract_audio_clip(
    input_file: str,
    output_file: str,
    start_time: float,
    duration: float,
) -> None:
    """
    normalized WAV에서 특정 구간만 추출한다.
    Recovery에서 사용한다.
    """

    output_dir = os.path.dirname(
        output_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    command = [
        "ffmpeg",
        "-y",
        "-v", "error",

        "-ss",
        f"{start_time:.3f}",

        "-i",
        input_file,

        "-t",
        f"{duration:.3f}",

        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",

        output_file,
    ]

    subprocess.run(
        command,
        check=True,
    )