from pause_detection import (
    detect_pauses,
    load_audio,
    preprocess_audio,
)
from repetition_detection import (
    detect_repetitions,
    extract_mfcc_frames,
    load_and_preprocess,
    print_results,
)

REPETITION_FILE = "repetition.wav"
PAUSE_FILE = "pause.wav"


def run_repetition_detection():
    print("\n===== REPEAT DETECTION =====")

    audio, sr = load_and_preprocess(REPETITION_FILE)
    mfcc_frames, frame_times = extract_mfcc_frames(audio, sr)
    events = detect_repetitions(mfcc_frames, frame_times)

    print_results(events)


def run_pause_detection():
    print("\n===== PAUSE DETECTION =====")

    y, sr = load_audio(PAUSE_FILE)
    y = preprocess_audio(y)

    pauses, total_pause = detect_pauses(y, sr)

    print("\nPause Segments:")
    for start, end in pauses:
        print(f"[{start:.2f}s – {end:.2f}s]")

    print(f"\nTotal Pause Duration: {total_pause:.2f}s")


if __name__ == "__main__":
    run_repetition_detection()
    run_pause_detection()
