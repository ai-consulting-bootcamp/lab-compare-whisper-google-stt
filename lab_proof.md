# Whisper STT Evaluation Lab

* **Generated On:** `2026-11-18 15:44:13`
* **Model Evaluated:** `whisper-1` (OpenAI API)
* **Scope:** Two human-verified audio samples

## 🔧 Workflow and input

The notebook transcribes [`voice_tam.m4a`](../audio/voice_tam.m4a) and [`voice_partner.m4a`](../audio/voice_partner.m4a) with the Whisper API, compares each transcript with manually verified ground truth, and calculates WER using lowercase conversion, punctuation removal, whitespace normalization, and `jiwer.process_words()`.

### Run/query and input

```python
with open(audio_path, "rb") as audio_file:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )

measures = jiwer.process_words(
    reference=normalize(ground_truth),
    hypothesis=normalize(result.text),
)
```

The two inputs were the audio files linked above. Their human-verified reference texts were entered in the notebook before evaluation.

---

## 📊 1. WER evidence

| Provider / Model | Audio File | Reference Words | Hypothesis Words | WER (%) | Substitutions | Insertions | Deletions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenAI Whisper (`whisper-1`)** | `voice_tam.m4a` | 62 | 62 | **0.0%** | 0 | 0 | 0 |
| **OpenAI Whisper (`whisper-1`)** | `voice_partner.m4a` | 73 | 73 | **0.0%** | 0 | 0 | 0 |

Both tested recordings produced **0.0% WER**. The complete machine-readable evidence is available in [`wer_comparison.json`](./wer_comparison.json); WER complement is not treated as general accuracy.

---

## 💰 2. Cost analysis
Using `$0.006/minute` as the working Whisper rate, estimated costs are: 10 hours = `$3.60`; 100 hours = `$36.00`; 500 hours = `$180.00`. Verify current pricing on the [OpenAI pricing page](https://platform.openai.com/pricing/).

## 🤝 3. Partner comparison and recommendation
Whisper produced zero measured word errors for both available voices. This is encouraging for a controlled pilot, but two short recordings are insufficient to generalize across accents, pace, overlap, background noise, or domain vocabulary. Test a larger representative meeting set before production deployment.

## ⚠️ 4. Limitation encountered
The assignment example used the older `jiwer.compute_measures()` API. This environment uses `jiwer` 4.x, so the workflow uses `jiwer.process_words()` instead. The small sample size is another limitation.

## Supporting files

- [`wer_comparison.json`](./wer_comparison.json) — final WER evidence and error counts
- [`compare_whisper_vs_google_tts.ipynb`](../compare_whisper_vs_google_tts.ipynb) — complete executable workflow
- [`voice_tam.m4a`](../audio/voice_tam.m4a) — your test input
- [`voice_partner.m4a`](../audio/voice_partner.m4a) — partner test input
