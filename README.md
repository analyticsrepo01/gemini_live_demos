# Gemini Live API — Hands-On Demo Notebooks

> **9 self-contained Jupyter notebooks** walking through every major feature of the Gemini 3.1 Live API — from a first WebSocket session to production-grade session management with Google Search grounding.

Model used throughout: **`gemini-3.1-flash-live-preview`**  
SDK: `google-genai` (Python) via AI Studio API key  
Audio: all synthetic — no microphone or external files required

---

## Quick Start

```bash
pip install google-genai nest_asyncio numpy python-dotenv
```

Add your key to `.env` (never commit this file):
```
GEMINI_API_KEY=your-key-here
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
jupyter lab
```

Open notebook `00` and run all cells. Each notebook is independent.

---

## What is the Gemini Live API?

The Gemini Live API provides **real-time bidirectional streaming** over a persistent WebSocket. Unlike the standard `generateContent` endpoint (one request → one response), the Live API works like a phone call:

```
Standard API:  [send prompt] ──► [wait] ──► [full response]

Live API:      [open session]
                   ├──► send text / audio chunk ──► Gemini receives
                   ◄── audio response streams back ◄── Gemini speaks
                   ├──► send more audio ────────────► continues
                   ◄── tool call request ◄──────────── needs data
                   ├──► tool response ──────────────► continues
                   ◄── final spoken answer ◄────────── done
               [close session]
```

**Key characteristics**:
- Sub-500ms time-to-first-audio on most queries
- Full duplex — send and receive simultaneously
- Persistent context — one session, many turns, no re-sending history
- Native function calling with streaming responses
- 70+ languages, 5 TTS voices, configurable VAD

---

## Notebooks

### 00 — Live API Basics
**`00_live_api_basics.ipynb`**

The entry point. Covers the complete session lifecycle and three fundamental patterns.

| Demo | What happens |
|------|-------------|
| Text → Audio + Transcript | Open a session, send text, receive PCM audio + streamed transcript |
| Text → Audio (raw) | Same but watching each chunk arrive in real time |
| Audio → Audio | Send synthetic PCM (resampled from Demo 2's output), receive spoken summary |
| Multi-turn + Context | 4-turn session: introduce a name in Turn 1, Gemini recalls it in Turn 4 |

**Core pattern**:
```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],               # only AUDIO is supported
    output_audio_transcription=types.AudioTranscriptionConfig(),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

async with client.aio.live.connect(model=MODEL, config=config) as session:
    await session.send_realtime_input(text="Hello!")
    async for resp in session.receive():
        if resp.data:                                       # PCM audio bytes
            audio_chunks.append(resp.data)
        if resp.server_content:
            if resp.server_content.output_transcription:   # streamed text
                print(resp.server_content.output_transcription.text, end="")
            if resp.server_content.turn_complete:
                break
```

**Critical rules learned here**:
- `response_modalities=["AUDIO"]` — TEXT modality is **not** supported by this model
- Never use `send_client_content` — always `send_realtime_input`
- Output audio: PCM16 @ 24 kHz. Input audio: PCM16 @ 16 kHz

---

### 01 — Audio Streaming
**`01_audio_streaming.ipynb`**

**Core capability proved**: The Live API accepts audio from *any* source — not just a live microphone. This notebook answers the three questions every developer asks when they first try to send audio.

| Demo | Developer question it answers | Real-world use case |
|------|------------------------------|---------------------|
| Chord to Gemini | "Can I send audio that isn't human speech?" | Music apps, sound classification, audio QA pipelines |
| WAV workflow | "I have a .wav file at 44.1 kHz — can I send it?" | Batch processing recorded calls, podcasts, meetings |
| Chunked streaming | "How do I wire up a live microphone?" | Voice assistants, real-time call agents, live transcription |

The chunked streaming demo is the most important: it shows the **exact production pattern** for a microphone. Real mic callbacks deliver 512 samples at a time (~32 ms). The demo simulates this with a real-time pacing delay — swap in your actual mic buffer and it works identically.

**The one rule that matters most** — always stream in small chunks. A single large blob causes a `1007 Precondition failed` error:
```python
# ❌ Fails for audio > a few KB
await session.send_realtime_input(audio=types.Blob(data=all_pcm_bytes, ...))

# ✅ Stream in 512-sample chunks — matches real mic callback size
CHUNK_SAMPLES = 512
for i in range(0, len(pcm_bytes), CHUNK_SAMPLES * 2):
    await session.send_realtime_input(
        audio=types.Blob(data=pcm_bytes[i:i+CHUNK_SAMPLES*2],
                         mime_type="audio/pcm;rate=16000")
    )
```

**Audio format**:
```
Input  → PCM16, 16 kHz, mono  (mime: "audio/pcm;rate=16000")
Output ← PCM16, 24 kHz, mono  (resp.data)
```

If your source file is at a different sample rate (44.1 kHz WAV, 48 kHz phone audio), resample before sending — the notebook shows how with numpy in ~5 lines.

**VAD and synthetic audio**: Gemini's Voice Activity Detection only triggers on speech-like audio. For synthetic tones, disable it and bracket manually:
```python
realtime_input_config=types.RealtimeInputConfig(
    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
)
# Then:
await session.send_realtime_input(activity_start=types.ActivityStart())
# ... send audio chunks ...
await session.send_realtime_input(activity_end=types.ActivityEnd())
```

---

### 02 — Transcription
**`02_transcription.ipynb`**

Full transcription pipeline — both directions simultaneously.

| Demo | What happens |
|------|-------------|
| Output transcription | Text → Gemini speech, transcript streams word-by-word alongside audio |
| Input transcription | Send audio, get back what Gemini heard (input) + what it says (output) |
| Conversation logging | 3-turn session logged to structured JSON with timestamps and audio duration |

**Config for both directions**:
```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription=types.AudioTranscriptionConfig(),   # what user said
    output_audio_transcription=types.AudioTranscriptionConfig(),  # what Gemini says
)

# In the receive loop:
if sc.input_transcription:
    print(f"[You said]    {sc.input_transcription.text}")
if sc.output_transcription:
    print(f"[Gemini said] {sc.output_transcription.text}")
```

Transcription streams incrementally — words arrive as they are spoken, not after the turn ends. This enables real-time subtitle rendering.

---

### 03 — Tool Use / Function Calling
**`03_tool_use.ipynb`**

Function calling inside a Live API session, including multi-tool routing and audio output.

| Demo | What happens |
|------|-------------|
| Calculator | `calculate(expression)` — Gemini calls it for arithmetic |
| Weather | `get_weather(city)` — mock API returning stub data |
| Multi-tool | Calculator + weather + clock in one session, audio + transcript saved |

**Tool call flow**:
```python
# Declare tools
tools = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_weather",
        description="Get current weather for a city",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"city": types.Schema(type=types.Type.STRING)},
            required=["city"],
        ),
    )
])]

# In receive loop — CRITICAL: break immediately on tool_call
if resp.tool_call:
    for fc in resp.tool_call.function_calls:
        result = my_function(fc.name, dict(fc.args))
        await session.send_tool_response(function_responses=[
            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
        ])
    # Do NOT wait for turn_complete here — server waits for tool response first
    break  # re-enter receive loop after sending response
```

> **Deadlock warning**: If you wait for `turn_complete` before sending `send_tool_response`, the server also waits — neither side moves. Always `break` on `tool_call` and re-enter the receive loop.

---

### 04 — Multilingual + Singapore English
**`04_multilingual.ipynb`**

70+ language support, mid-session language switching, and a Singlish demo.

| Demo | What happens |
|------|-------------|
| 5 languages | Same question ("capital of France?") asked in EN/HI/ES/FR/JP |
| Language switching | Single session, model follows language changes automatically |
| Language detection | Send greetings in 3 scripts — model detects and responds in each |
| Voice + language_code | Pair BCP-47 code with a TTS voice for correct accent |
| en-SG + Singlish | Three variations: formal SG English, casual Singlish, food aunty mode |

**Language + voice config**:
```python
speech_config=types.SpeechConfig(
    language_code="en-SG",   # BCP-47 code sets TTS accent
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
    ),
)
```

**Singlish system prompt** (the model naturally uses lah/lor/sia/wah):
```python
system_instruction=(
    "You are a friendly Singaporean. Respond ONLY in Singlish — "
    "the colloquial English spoken in Singapore. "
    "Use particles naturally: lah, lor, leh, sia, hor, can or not, aiyo, wah."
)
```

Live output: *"Wah, Singapore food best one lah! Must try chicken rice first, confirm tasty! Or if you want spicy, laksa lor."*

**Selected language codes**:
| Language | Code | Language | Code |
|---|---|---|---|
| English (US) | `en-US` | English (SG) | `en-SG` |
| Hindi | `hi-IN` | Japanese | `ja-JP` |
| Mandarin | `cmn-CN` | Arabic | `ar-XA` |
| French | `fr-FR` | Spanish | `es-ES` |

---

### 05 — Barge-In / Voice Activity Detection
**`05_barge_in.ipynb`**

How Gemini detects when a user interrupts a long response.

| Demo | What happens |
|------|-------------|
| Text barge-in signals | Gemini starts a long story; user sends a follow-up text mid-response |
| Audio interruption | `activity_start` + audio blob + `activity_end` triggers `sc.interrupted` |
| Turn coverage | Automatic vs activity-only turn scoping |
| Full barge-in loop | State machine: IDLE → SPEAKING → INTERRUPTED → LISTENING |

**Detecting interruption**:
```python
async for resp in session.receive():
    if resp.server_content:
        if resp.server_content.interrupted:
            # Gemini stopped speaking — user took the floor
            print("Barge-in detected!")
            break
```

**Manual barge-in via activity signals**:
```python
await session.send_realtime_input(activity_start=types.ActivityStart())
# stream audio chunks...
await session.send_realtime_input(activity_end=types.ActivityEnd())
# then send follow-up
await session.send_realtime_input(text="Actually, just tell me the summary.")
```

---

### 06 — Proactive Audio / Push-to-Talk
**`06_proactive_audio.ipynb`**

Manual VAD — the client controls exactly when "speech" starts and ends, like a PTT radio.

| Demo | What happens |
|------|-------------|
| Automatic VAD | Default mode — Gemini detects speech boundaries itself |
| Manual PTT | Disable VAD, bracket audio with ActivityStart/End explicitly |
| Sensitivity tuning | Adjust VAD prefix/suffix padding for different environments |

**Push-to-talk pattern**:
```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
    ),
)

# "Button pressed"
await session.send_realtime_input(activity_start=types.ActivityStart())

# Stream mic chunks while button is held
for chunk in mic_buffer:
    await session.send_realtime_input(
        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
    )

# "Button released"
await session.send_realtime_input(activity_end=types.ActivityEnd())
```

Use this pattern in noisy environments where auto-VAD would falsely trigger, or in walkie-talkie / call center UIs where push-to-talk is the expected interaction model.

---

### 07 — Affective Dialog / Persona Voices
**`07_affective_dialog.ipynb`**

Shape the emotional register and vocal personality of responses through system prompts and voice selection.

| Demo | What happens |
|------|-------------|
| 4 tonal personas | Warm counsellor, confident coach, curious analyst, calm narrator |
| Emotion-adaptive | System prompt shifts tone based on detected user sentiment |
| All 5 voices | Same text spoken by Puck, Aoede, Kore, Charon, Fenrir — compare output |

**Available voices**:
| Voice | Character |
|-------|-----------|
| `Puck` | Bright, upbeat, friendly |
| `Aoede` | Warm, melodic, natural |
| `Kore` | Clear, professional, neutral |
| `Charon` | Deep, authoritative |
| `Fenrir` | Expressive, dynamic |

**Persona via system instruction**:
```python
system_instruction=(
    "You are a warm, empathetic counsellor. "
    "Speak gently and use encouraging language. "
    "Acknowledge emotions before offering advice."
)
```

The voice carries the persona — pair `Aoede` (warm) with a counsellor prompt, `Charon` (deep) with an authoritative narrator.

---

### 08 — Google Search + Session Management
**`08_google_search_and_sessions.ipynb`**

Production patterns: real-time grounding, combined tools, and resilient session handling.

| Demo | What happens |
|------|-------------|
| Google Search grounding | Live answers to "current population of Singapore?" and "latest F1 winner?" |
| Search + function calling | Google Search (news) + custom `get_weather` tool in one session |
| GoAway reconnect | Auto-reconnect loop with exponential back-off when session expires |
| Context-preserving sessions | Conversation history injected into new session's system prompt |
| LiveSessionManager | Production-ready class: reconnect, context, backoff, conv log |

**Google Search tool** (server-side — no client round-trip):
```python
tools = [types.Tool(google_search=types.GoogleSearch())]
# The model queries Search internally; client never sees the raw search calls.
# You just get the grounded final answer.
```

**GoAway reconnect pattern**:
```python
while questions_remaining:
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        for q in questions:
            await session.send_realtime_input(text=q)
            r = await collect_response(session)
            if r["go_away"]:
                break   # session expiring — outer loop will reconnect
            log.append({"q": q, "a": r["transcript"]})
```

**Context injection on reconnect**:
```python
# Build system prompt from conversation history
history = "\n".join(f"User: {t['q']}\nAssistant: {t['a']}" for t in log[-5:])
system_instruction = f"You are a helpful assistant.\n\nPREVIOUS CONTEXT:\n{history}"
# Pass to new session config — model picks up the conversation naturally
```

---

## Common Pitfalls & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `1008 policy violation` | `send_client_content` mixed with `send_realtime_input` | Use only `send_realtime_input` |
| `1011 internal error` | `response_modalities=["TEXT"]` — not supported | Use `["AUDIO"]` + `output_audio_transcription` |
| `1007 precondition failed` | Single large audio blob | Send in 512-sample chunks |
| Session hangs after audio | VAD doesn't recognise synthetic tones as speech | Disable VAD + use `ActivityStart`/`ActivityEnd` |
| Tool call deadlock | Waiting for `turn_complete` before sending tool response | `break` on `tool_call`, send response, re-enter loop |
| `nest_asyncio has no attribute 'patch'` | Wrong method name | Use `nest_asyncio.apply()` not `.patch()` |

---

## Architecture Reference

```
┌─────────────────────────────────────────────────────┐
│                  Your Application                    │
│                                                     │
│  send_realtime_input(text=...)                      │
│  send_realtime_input(audio=Blob(...))               │
│  send_realtime_input(activity_start=ActivityStart())│
│  send_tool_response(function_responses=[...])       │
└────────────────────┬────────────────────────────────┘
                     │  WebSocket (bidirectional)
┌────────────────────▼────────────────────────────────┐
│           gemini-3.1-flash-live-preview              │
│                                                     │
│  resp.data                 → PCM16 audio @ 24kHz   │
│  resp.server_content                                │
│    .output_transcription   → streamed spoken text   │
│    .input_transcription    → what model heard       │
│    .turn_complete          → model finished turn    │
│    .interrupted            → user barged in         │
│  resp.tool_call            → function call request  │
│  resp.go_away              → session expiring soon  │
└─────────────────────────────────────────────────────┘
```

---

## Setup Details

### `.env` file (never commit)
```
GEMINI_API_KEY=AIza...
```

### Load in notebooks
```python
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
```

### Jupyter async compatibility
```python
import nest_asyncio
nest_asyncio.apply()   # must be called before any asyncio.run()
asyncio.run(my_coroutine())
```

---

## References

- [Gemini Live API Docs](https://ai.google.dev/api/live)
- [google-genai Python SDK](https://github.com/googleapis/python-genai)
- [AI Studio — get API key](https://aistudio.google.com/apikey)
- [BCP-47 language codes](https://www.iana.org/assignments/language-subtag-registry)
