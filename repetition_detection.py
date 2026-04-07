import librosa
import noisereduce as nr
import numpy as np
from scipy.spatial.distance import cosine

AUDIO_PATH = "repetition.wav"
SIMILARITY_THRESHOLD = 0.87


def load_and_preprocess(filepath):
    audio, sr = librosa.load(filepath, sr=16000, mono=True)
    noise_sample = audio[: int(sr * 0.5)]
    audio = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=sr, prop_decrease=0.75)

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    return audio, sr


def extract_mfcc_frames(audio, sr):
    frame_len = int(0.03 * sr)
    hop_len = int(0.015 * sr)

    mfccs = librosa.feature.mfcc(
        y=audio, n_mfcc=13, n_fft=frame_len, hop_length=hop_len
    ).T

    n_frames = mfccs.shape[0]
    frame_times = np.arange(n_frames) * 0.015

    return mfccs, frame_times


def detect_repetitions(mfcc_frames, frame_times):
    n_frames = len(mfcc_frames)
    similar_flags = np.zeros(n_frames, dtype=bool)
    similarity_scores = np.zeros(n_frames, dtype=float)

    for i in range(n_frames):
        best_sim = 0.0
        for gap in range(1, int(0.5 / 0.015)):
            j = gap + i
            if j >= n_frames:
                break

            if np.all(mfcc_frames[i] == 0) or np.all(mfcc_frames[j] == 0):
                continue

            sim = 1 - cosine(mfcc_frames[i], mfcc_frames[j])
            if sim > best_sim:
                best_sim = sim
        if best_sim >= SIMILARITY_THRESHOLD:
            similar_flags[i] = True
            similarity_scores[i] = best_sim
    events = []
    in_event = False
    event_start = 0

    for i in range(n_frames):
        if similar_flags[i] and not in_event:
            in_event = True
            event_start = i
        elif not similar_flags[i] and in_event:
            in_event = False
            event_end = i - 1
            run_len = event_end - event_start + 1
            avg_sim = float(np.mean(similarity_scores[event_start : event_end + 1]))
            duration = frame_times[event_end] - frame_times[event_start]
            repeat_count = max(2, int(duration / 0.15))
            events.append(
                {
                    "start": float(frame_times[event_start]),
                    "end": float(frame_times[event_end]),
                    "count": repeat_count,
                    "similarity": avg_sim,
                }
            )
    if in_event:
        event_end = n_frames - 1
        run_len = event_end - event_start + 1
        avg_sim = float(np.mean(similarity_scores[event_start : event_end + 1]))
        repeat_count = max(2, round(run_len / 4) + 1)
        events.append(
            {
                "start": float(frame_times[event_start]),
                "end": float(frame_times[event_end]),
                "count": repeat_count,
                "similarity": avg_sim,
            }
        )
    return events


def print_results(events):
    print(f"\nFile: {AUDIO_PATH}")
    print("-" * 40)
    print("Repetitions:")

    if not events:
        print("  No repetition patterns detected.")
        return

    for evt in events:
        print(f"  Time range       : {evt['start']:.2f}s – {evt['end']:.2f}s")
        print(f"  Repetition Count : {evt['count']}")
        print(f"  Avg Similarity   : {evt['similarity']:.3f}")

    print(f"\nTotal Repetition Events: {len(events)}")


audio, sr = load_and_preprocess(AUDIO_PATH)
mfcc_frames, frame_times = extract_mfcc_frames(audio, sr)
events = detect_repetitions(mfcc_frames, frame_times)
print_results(events)
