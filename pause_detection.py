import librosa
import numpy as np


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def preprocess_audio(y):
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val


def detect_pauses(y, sr, frame_length=1024, step_size=512, threshold=None):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=step_size)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=step_size)

    if threshold is None:
        threshold = np.mean(rms) * 0.5

    pauses = []
    is_pause = False
    start_time = 0
    min_pause_duration = 0.2

    for i, energy in enumerate(rms):
        if energy < threshold and not is_pause:
            is_pause = True
            start_time = times[i]
        elif energy >= threshold and is_pause:
            end_time = times[i]
            if end_time - start_time >= min_pause_duration:
                pauses.append((start_time, end_time))
            is_pause = False

    if is_pause:
        audio_duration = len(y) / sr
        pauses.append((start_time, audio_duration))

    total_pause = sum(end - start for start, end in pauses)

    return pauses, total_pause


if __name__ == "__main__":
    file_path = "pause.wav"

    y, sr = load_audio(file_path)
    y = preprocess_audio(y)

    pauses, total_pause = detect_pauses(y, sr)

    print("\n--- Pause Detection ---")
    print("Pause Segments:")
    for start, end in pauses:
        print(f"[{start:.2f}s – {end:.2f}s]")

    print(f"\nTotal Pause Duration: {total_pause:.2f}s")
