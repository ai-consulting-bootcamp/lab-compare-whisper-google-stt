# Lab Proof — Whisper STT Implementation

## 1. Lab Objective

The objective of this lab was to implement speech-to-text transcription using OpenAI Whisper, process audio in chunks, generate timestamps and export the transcription in different formats.

## 2. Code and Workflow

**Supporting notebook:** `LAB_Whisper_STT .ipynb`

### Step 1 — Load OpenAI Client and Audio File

```python
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(
    dotenv_path="/Users/ritasousa/Desktop/AI Consulting & Integration/WEEK_3/9. LAB | API Calling to ChatGPT/OpenAI_API/.env"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

audio_file = Path(
    "/Users/ritasousa/Desktop/AI Consulting & Integration/WEEK_6/LAB_Whisper_STT I_mplementation/Arthur.mp3"
)
```

### Step 2 — Basic Transcription

```python
with audio_file.open("rb") as audio:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio
    )

print(transcription.text)
```

### Step 3 — Prompted Transcription

```python
prompt = (
    "This is a reading of the story 'The Story of Arthur the Rat'. "
    "The main character is a young rat. Other characters include "
    "Aunt Josephine. The story mentions rats, hunting, a loft, "
    "a hole in the ground, beams and rafters."
)

with audio_file.open("rb") as audio:
    prompted_transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        prompt=prompt
    )

print(prompted_transcription.text)
```

### Step 4 — Unprompted Transcription

```python
with audio_file.open("rb") as audio:
    unprompted_transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio
    )

print(unprompted_transcription.text)
```

### Step 5 — Audio Chunking

```python
from pydub import AudioSegment
from pathlib import Path

audio = AudioSegment.from_mp3(str(audio_file))

chunk_length_ms = 10 * 60 * 1000  # 10 minutes

chunks_folder = Path("audio_chunks")
chunks_folder.mkdir(exist_ok=True)

chunks = []

for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
    chunk = audio[start:start + chunk_length_ms]
    chunk_path = chunks_folder / f"chunk_{i + 1}.mp3"

    chunk.export(chunk_path, format="mp3")
    chunks.append(chunk_path)

    print(f"Created: {chunk_path} | Duration: {len(chunk) / 1000:.2f} seconds")
```

**Execution result:**

```text
Created: audio_chunks/chunk_1.mp3 | Duration: 228.36 seconds
```

### Step 6 — Transcribing Audio Chunks

```python
from datetime import timedelta

chunk_transcriptions = []

for chunk_path in sorted(chunks_folder.glob("chunk_*.mp3")):
    with chunk_path.open("rb") as audio:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="verbose_json"
        )

    chunk_transcriptions.append(result)

    print(f"Transcribed: {chunk_path.name}")
```

**Execution result:**

```text
Transcribed: chunk_1.mp3
```

### Step 7 — Timestamp Adjustment

```python
timestamped_segments = []

for i, result in enumerate(chunk_transcriptions):
    chunk_offset = i * (chunk_length_ms / 1000)

    for segment in result.segments:
        start = segment.start + chunk_offset
        end = segment.end + chunk_offset

        timestamped_segments.append({
            "start": str(timedelta(seconds=int(start))),
            "end": str(timedelta(seconds=int(end))),
            "text": segment.text
        })

print(f"Total segments: {len(timestamped_segments)}")
print(timestamped_segments[:3])
```

**Execution result:**

```text
Total segments: 59
```

### Step 8 — Combined Timestamped Transcription

```python
final_transcription = ""

for segment in timestamped_segments:
    final_transcription += (
        f"[{segment['start']} - {segment['end']}] "
        f"{segment['text']}\n"
    )

output_file = Path("chunked_transcription.txt")
output_file.write_text(final_transcription, encoding="utf-8")

print(f"Transcription exported to: {output_file.resolve()}")
```

### Step 9 — Export to TXT

```python
human_readable_transcription = ""

for segment in timestamped_segments:
    human_readable_transcription += (
        f"[{segment['start']} - {segment['end']}]\n"
        f"{segment['text'].strip()}\n\n"
    )

txt_file = Path("transcription_readable.txt")
txt_file.write_text(human_readable_transcription, encoding="utf-8")

print(f"TXT exported: {txt_file.resolve()}")
```

### Step 10 — Export to SRT

```python
def format_srt_time(seconds):
    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000
    minutes = (milliseconds % 3_600_000) // 60_000
    seconds_part = (milliseconds % 60_000) // 1000
    milliseconds_part = milliseconds % 1000

    return (
        f"{hours:02}:{minutes:02}:{seconds_part:02},"
        f"{milliseconds_part:03}"
    )


srt_content = ""

for index, segment in enumerate(timestamped_segments, start=1):
    start_seconds = sum(
        int(value) * factor
        for value, factor in zip(
            segment["start"].split(":"),
            [3600, 60, 1]
        )
    )

    end_seconds = sum(
        int(value) * factor
        for value, factor in zip(
            segment["end"].split(":"),
            [3600, 60, 1]
        )
    )

    srt_content += (
        f"{index}\n"
        f"{format_srt_time(start_seconds)} --> "
        f"{format_srt_time(end_seconds)}\n"
        f"{segment['text'].strip()}\n\n"
    )

srt_file = Path("transcription_subtitles.srt")
srt_file.write_text(srt_content, encoding="utf-8")

print(f"SRT exported: {srt_file.resolve()}")
```

### Step 11 — Export to JSON

```python
import json

json_file = Path("transcription_timestamps.json")

json_file.write_text(
    json.dumps(timestamped_segments, indent=4, ensure_ascii=False),
    encoding="utf-8"
)

print(f"JSON exported: {json_file.resolve()}")
```

## 3. Test Input and Execution

**Input file:** `Arthur.mp3`

The audio contains the story *The Story of Arthur the Rat*.

The workflow was executed in a Jupyter notebook using Python, OpenAI Whisper and Pydub.

The audio was split into 10-minute chunks. Since the original audio was shorter than 10 minutes, only one chunk was generated:

- `chunk_1.mp3`
- Duration: 228.36 seconds

The chunk was successfully transcribed using the Whisper API with the `verbose_json` response format.

## 4. Evidence and Results

The workflow successfully produced:

- A basic transcription of the original audio.
- A prompted transcription.
- An unprompted transcription.
- 59 timestamped transcription segments.
- A combined transcription with start and end timestamps.

The prompted and unprompted transcriptions were compared. Their results were nearly identical, with minor punctuation differences.

## 5. Final Outputs

The transcription was exported into three formats:

| File | Description |
|---|---|
| `transcription_readable.txt` | Human-readable transcription with timestamps |
| `transcription_subtitles.srt` | Subtitle file with numbered entries and timestamps |
| `transcription_timestamps.json` | Structured transcription data containing start, end and text fields |

Additional supporting files:

- `Arthur.mp3` — original audio input.
- `chunk_1.mp3` — generated audio chunk.
- `LAB_Whisper_STT .ipynb` — complete code and execution workflow.

## 6. Failure and Limitation

During the initial audio chunking attempt, Pydub could not locate FFmpeg and FFprobe, resulting in a `FileNotFoundError`.

This was resolved by installing FFmpeg through Conda:

```bash
conda install -c conda-forge ffmpeg
```

After installation, the chunking process ran successfully.

One limitation of this test is that the audio was shorter than the configured 10-minute chunk length, so only one chunk was generated. The workflow therefore did not test timestamp adjustment across multiple chunks.

Additionally, timestamps were converted to whole seconds when combining the segments, so the final exports do not preserve sub-second timing precision.

## 7. Conclusion

The lab successfully demonstrated audio transcription with OpenAI Whisper, prompted and unprompted transcription comparison, audio chunking, timestamp generation and export into TXT, SRT and JSON formats.

All three export files were generated successfully.