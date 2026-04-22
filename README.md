# Gemini Live API Demos

A collection of hands-on Jupyter notebooks demonstrating the **Gemini 3.1 Live API** (`gemini-3.1-flash-live-preview`) — real-time bidirectional streaming for voice, audio, and multimodal AI applications.

Built for live audience presentation. Each notebook is self-contained, runs in Jupyter, and uses **Python only** with synthetic audio (no external audio files required).

---

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 00 | [Live API Basics](00_live_api_basics.ipynb) | WebSocket session lifecycle, text→text, text→audio, PCM audio→audio |
| 01 | [Audio Streaming](01_audio_streaming.ipynb) | PCM16 format, WAV workflow, chunked streaming simulation |
| 02 | [Transcription](02_transcription.ipynb) | Input & output audio transcription, streaming transcripts, conversation logging |
| 03 | [Tool Use](03_tool_use.ipynb) | Function calling — calculator, weather (mock), multi-tool sessions |
| 04 | [Multilingual](04_multilingual.ipynb) | 20-language support, language switching mid-session, voice + language_code config |
| 05 | [Barge-In / VAD](05_barge_in.ipynb) | Voice Activity Detection, ActivityStart/End events, interrupted turn handling |
| 06 | [Proactive Audio](06_proactive_audio.ipynb) | Automatic vs manual VAD, push-to-talk pattern, sensitivity tuning |
| 07 | [Affective Dialog](07_affective_dialog.ipynb) | Tonal personas, emotion-adaptive prompts, all 5 Gemini voices compared |
| 08 | [Google Search + Sessions](08_google_search_and_sessions.ipynb) | Search grounding, combined search+function calling, GoAway reconnect with backoff |

---

## Setup

### Prerequisites

```bash
pip install google-genai nest_asyncio numpy
```

### API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey) and set it as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

Or paste it directly into the `API_KEY` cell at the top of each notebook.

### Run

```bash
jupyter lab
```

Open any notebook and run all cells.

---

## Model

All notebooks use:

```
gemini-3.1-flash-live-preview
```

via the `google-genai` Python SDK with the AI Studio API key.

---

## Key Concepts

### Session Pattern

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)

async with client.aio.live.connect(model=MODEL, config=config) as session:
    await session.send_realtime_input(text="Hello!")
    async for resp in session.receive():
        if resp.data:              # audio bytes (PCM16, 24kHz)
            ...
        if resp.server_content:   # transcripts
            ...
        if resp.tool_call:        # function call requests
            ...
```

### Critical Rules

- **Never mix** `send_client_content` with `send_realtime_input` — causes WebSocket 1008 policy violation
- Always use `send_realtime_input(text=...)` for text, `send_realtime_input(audio=types.Blob(...))` for audio
- Synthetic audio: PCM16, 16 kHz input; agent audio output at 24 kHz

### Jupyter Async

```python
import nest_asyncio
nest_asyncio.patch()
asyncio.run(my_async_function())
```

---

## Synthetic Audio Helper

All notebooks use this to generate test audio without external files:

```python
import numpy as np

def make_pcm(duration=2.0, rate=16000, freq=440):
    t = np.linspace(0, duration, int(rate * duration))
    return (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16).tobytes()
```

---

## References

- [Gemini Live API Docs](https://ai.google.dev/api/live)
- [google-genai Python SDK](https://github.com/googleapis/python-genai)
- [AI Studio](https://aistudio.google.com)

---

*All demos run entirely in Jupyter with no external audio dependencies.*
