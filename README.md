# 🎙️ Speech Analysis: Pause & Repetition Detection

A Python-based audio analysis system that detects **pause segments** and **repetition patterns** (stuttering) in speech audio files.

---

## 📁 Project Structure

```
speech_analysis/
├── pause_detection.py      # Detects silent regions in speech
├── repetition_detection.py # Detects stuttered/repeated patterns
├── main.py                 # Runs both detections together
├── pause.wav               # Sample audio for pause detection
└── repetition.wav          # Sample audio for repetition detection
```

---

## ⚙️ Installation

```bash
pip install librosa numpy scipy noisereduce
```

---

## 🚀 Usage

```bash
python3 main.py
```

Or run each module individually:

```bash
python3 pause_detection.py
python3 repetition_detection.py
```

---

## 📌 Approach

The system is split into two independent detection pipelines, both sharing a common preprocessing step.

### Common Preprocessing
- **Load audio** using `librosa` with the original sample rate
- **Normalize amplitude** to the range `[-1, 1]` so energy comparisons are consistent across different recordings
- *(Repetition module)* **Noise reduction** via `noisereduce` using the first 0.5s as a noise profile

---

## ⏸️ Task 1: Pause Detection

**File:** `pause_detection.py`

### How It Works

1. **RMS Energy Extraction** — The audio is split into overlapping frames (`frame_length=1024`, `hop_length=512`). For each frame, the Root Mean Square (RMS) energy is computed. RMS measures the "loudness" of that chunk.

2. **Threshold Calculation** — If no threshold is given, it defaults to `mean(RMS) × 0.5`. Frames below this threshold are considered silent.

3. **Pause Detection Loop** — The code walks frame-by-frame:
   - When energy drops below the threshold → mark as start of a pause
   - When energy rises back above → mark as end of a pause
   - Only pauses lasting **≥ 0.2 seconds** are recorded (to filter out micro-silences between words)

4. **Total Pause Duration** — Sum of all detected pause durations.

### Features Used
| Feature | Purpose |
|---|---|
| RMS Energy | Measures loudness per frame |
| Dynamic threshold | Adapts to each audio file's average loudness |
| Minimum duration filter | Removes noise-level false pauses |

### Sample Output

```
--- Pause Detection ---
Pause Segments:
[0.00s – 0.67s]
[1.60s – 2.69s]
[6.37s – 6.75s]

Total Pause Duration: 2.15s

```

---

## 🔁 Task 2: Repetition Detection

**File:** `repetition_detection.py`

### How It Works

1. **MFCC Feature Extraction** — The audio is divided into overlapping frames (30ms window, 15ms hop). Each frame is converted into **13 MFCC coefficients** — a compact fingerprint of how that sound *sounds*, mimicking how the human ear processes audio.

2. **Cosine Similarity Comparison** — For every frame `i`, it is compared against the next 1–10 frames using cosine similarity:
   - Similarity close to **1.0** → frames sound nearly identical → likely a repetition
   - Frames scoring above the threshold (`0.92` by default) are flagged

3. **Grouping into Events** — Consecutive flagged frames are grouped into a single repetition event. The repeat count is estimated by dividing the run length by the average duration of one repeated unit (~60ms / 4 frames).

4. **Filtering** — Events with fewer than `MIN_REPEAT_COUNT` repetitions are discarded.

### Features Used
| Feature | Purpose |
|---|---|
| MFCC (13 coefficients) | Phoneme-level sound fingerprint |
| Cosine Similarity | Compares sound identity, ignoring volume |
| Overlapping frames | Catches repetitions at any alignment |
| Consecutive grouping | Merges nearby flags into one event |

### Sample Output

```
--- Repetition Detection ---
Repetitions:
  Time range       : 0.00s – 10.39s
  Repetition Count : 69
  Avg Similarity   : 0.998

Total Repetition Events: 1

```


---

## 📝 Notes

- Both modules work independently — you can run either one without the other.
- Adjust `SIMILARITY_THRESHOLD` in `repetition_detection.py` for stricter or looser repetition matching.
- Adjust `threshold` parameter in `detect_pauses()` for different audio environments.
