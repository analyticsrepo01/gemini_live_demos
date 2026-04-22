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
Standard API:  [send prompt] ──► [wait 2-3s] ──► [full response arrives]

Live API:      [open session]
                   ├──► send text / audio ──────► Gemini receives
                   ◄── audio streams back ◄─────── Gemini starts speaking immediately
                   ├──► send more audio ─────────► continues listening
                   ◄── tool call request ◄─────────  needs your data
                   ├──► tool response ───────────► continues
                   ◄── final spoken answer ◄──────── done
               [session stays open — next turn reuses same context]
```

**Key characteristics**:
- Sub-500ms time-to-first-audio on most queries
- Full duplex — send and receive simultaneously
- Persistent context — one session, many turns, no re-sending history
- Native function calling with real-time tool execution
- 70+ languages, 5 TTS voices, configurable VAD

**What you can build with it**:
- Voice AI agents and assistants
- Real-time call center automation
- Live transcription + translation pipelines
- Multimodal interfaces (audio in, audio + text out)
- Conversational agents with tool/API access

---

## Notebooks

---

### 00 — Live API Basics
**`00_live_api_basics.ipynb`**

**Core capability proved**: You can open a persistent voice session with Gemini, send input (text or audio), and get spoken audio + streamed text back — all in real time, with full conversational memory across turns.

**What this unlocks for agent builders**: This is the foundation every other notebook builds on. If you want to build any kind of voice agent — customer support bot, voice assistant, spoken Q&A system — this is the session pattern you will use.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| Text → Audio + Transcript | "How do I get Gemini to speak and give me the text at the same time?" | Any voice agent that needs both audio output and a text log |
| Text → Audio (raw chunks) | "How does the audio actually stream — is it one blob or many chunks?" | Understanding the streaming model before building a real-time player |
| Audio → Audio | "Can I send audio in and get audio back?" | Voice-to-voice agents — mic in, speaker out |
| Multi-turn + Context | "Does Gemini remember what was said earlier in the session?" | Conversational agents where context matters across turns |

The multi-turn demo is the most important one to understand. It introduces a name in Turn 1, asks two unrelated questions in Turns 2 and 3, then asks "what's my name?" in Turn 4. Gemini answers correctly — because it's the same open WebSocket and the server maintains the full conversation context automatically. **You do not need to re-send history on every turn.**

**Core session pattern** — every agent you build starts here:
```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],                          # AUDIO is the only supported modality
    output_audio_transcription=types.AudioTranscriptionConfig(),  # get text alongside audio
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

async with client.aio.live.connect(model=MODEL, config=config) as session:
    await session.send_realtime_input(text="Hello!")

    async for resp in session.receive():
        if resp.data:                                           # PCM16 audio @ 24kHz
            audio_chunks.append(resp.data)
        if resp.server_content:
            if resp.server_content.output_transcription:       # streamed transcript
                print(resp.server_content.output_transcription.text, end="")
            if resp.server_content.turn_complete:
                break                                          # Gemini finished this turn
```

**Three rules you must know before anything else**:
1. `response_modalities=["AUDIO"]` only — TEXT modality causes a silent 1011 error and a hanging session
2. Always use `send_realtime_input` — never `send_client_content` (causes 1008 policy violation)
3. Output audio is PCM16 @ 24 kHz. Input audio must be PCM16 @ 16 kHz

---

### 01 — Audio Streaming
**`01_audio_streaming.ipynb`**

**Core capability proved**: The Live API accepts audio from *any* source — not just a live microphone. This notebook answers the three questions every developer asks when they first try to send audio, and shows the exact production pattern for a real microphone stream.

**What this unlocks for agent builders**: Once you know how to send audio correctly, you can build agents that process recorded calls, analyse audio files, accept live mic input, or stream from any audio source. The chunked streaming demo is the microphone wiring pattern you will copy directly into a production app.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| Chord to Gemini | "Can I send any audio, not just speech?" | Audio analysis agents — classify sounds, describe music, quality check recordings |
| WAV workflow | "I have a .wav file at 44.1 kHz — how do I send it?" | Batch processing agents — analyse recorded calls, meetings, podcasts at scale |
| Chunked streaming | "How do I wire up a live microphone?" | Real-time voice agents — this is the exact mic streaming pattern for production |

The chunked streaming demo is the most production-relevant: real microphone OS callbacks deliver 512 samples (~32ms of audio) at a time. The demo simulates this pacing exactly. To wire up a real mic (e.g. using PyAudio or sounddevice), replace the synthetic chunks with your mic buffer and the pattern is identical.

**The single most important rule when sending audio**:
```python
# ❌ This FAILS for any audio longer than a few hundred milliseconds
await session.send_realtime_input(audio=types.Blob(data=full_audio_bytes, ...))

# ✅ Always stream in 512-sample chunks — matches real mic callback size
CHUNK = 512 * 2  # 512 samples × 2 bytes (int16)
for i in range(0, len(pcm_bytes), CHUNK):
    await session.send_realtime_input(
        audio=types.Blob(data=pcm_bytes[i:i+CHUNK], mime_type="audio/pcm;rate=16000")
    )
```

**Audio format**:
```
Input  → PCM16, 16 kHz, mono   mime_type="audio/pcm;rate=16000"
Output ← PCM16, 24 kHz, mono   arrives in resp.data
```

If your source is at a different rate (44.1 kHz WAV, 48 kHz phone call), resample first. The WAV demo shows how using numpy in ~5 lines.

**Why VAD matters for non-speech audio**: Gemini's built-in Voice Activity Detection only triggers on speech-like waveforms. Tones, music, and other synthetic audio won't trigger it — the session will hang. For any non-speech audio source, disable VAD and bracket manually:
```python
await session.send_realtime_input(activity_start=types.ActivityStart())
# send your chunks
await session.send_realtime_input(activity_end=types.ActivityEnd())
```

---

### 02 — Transcription
**`02_transcription.ipynb`**

**Core capability proved**: You can get a word-by-word text transcript of both sides of a Live API conversation simultaneously — what the user sent *and* what Gemini spoke — without any post-processing step. The transcript arrives streaming, in real time, alongside the audio.

**What this unlocks for agent builders**: Any agent that needs a text record of a voice conversation — call logs, compliance records, searchable transcripts, real-time subtitles, conversation summaries — can be built on this pattern. Two config lines enable full bidirectional transcription.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| Output transcription | "How do I get the text of what Gemini just said?" | Display subtitles while Gemini speaks; build a chat log alongside audio |
| Input transcription | "How do I know what Gemini heard from the audio I sent?" | Verify audio was understood correctly; build a full conversation record |
| Conversation logging | "How do I persist the conversation with timestamps and audio metadata?" | Call center audit trails, CRM logging, compliance recording |

The key insight: transcription is **not** a post-processing step. The text arrives word-by-word in the same streaming loop as the audio, via `resp.server_content.output_transcription`. You can render subtitles in real time, exactly in sync with the spoken audio.

**Enable both directions with two lines**:
```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription=types.AudioTranscriptionConfig(),   # what user said
    output_audio_transcription=types.AudioTranscriptionConfig(),  # what Gemini says
)
```

**Read both in the receive loop**:
```python
async for resp in session.receive():
    if resp.server_content:
        sc = resp.server_content
        if sc.input_transcription and sc.input_transcription.text:
            print(f"[User]   {sc.input_transcription.text}")   # what Gemini heard
        if sc.output_transcription and sc.output_transcription.text:
            print(f"[Gemini] {sc.output_transcription.text}", end="")  # streams word-by-word
        if sc.turn_complete:
            break
```

**The conversation log demo** shows how to structure per-turn records with timestamp, transcript, and audio duration — the data model you'd store in a database for a call center system:
```python
log.append({
    "turn": turn_num,
    "timestamp": datetime.utcnow().isoformat(),
    "user_text": user_text,
    "gemini_transcript": full_transcript,
    "audio_duration_s": (len(audio_bytes) // 2) / 24000,
})
```

---

### 03 — Tool Use / Function Calling
**`03_tool_use.ipynb`**

**Core capability proved**: Gemini can call your functions — APIs, databases, calculators, any service — during a live voice session, get the result, and incorporate it into a spoken answer, all without breaking the audio stream. The model decides which tool to call, when to call it, and how to use the result.

**What this unlocks for agent builders**: This is what turns a voice interface into a *voice agent*. Without tools, Gemini can only use its training data. With tools, it can query your CRM, check inventory, look up prices, execute bookings, run calculations — and speak the result back in natural language. Any API you have can become a voice-accessible agent action.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| Calculator tool | "How do I let Gemini call a function and use the result?" | Understand the basic tool call → response → continue flow |
| Weather tool | "How do I wire up a real API call?" | Any external data source: CRM, inventory, pricing, availability |
| Multi-tool session | "Can Gemini pick the right tool when multiple are available?" | Full agents with multiple capabilities — Gemini routes to the right tool automatically |

**The tool call loop** — the pattern every function-calling agent uses:
```python
async with client.aio.live.connect(model=MODEL, config=config) as session:
    await session.send_realtime_input(text=user_query)

    while True:
        async for resp in session.receive():
            if resp.tool_call:
                # Gemini decided to call a function — execute it
                results = []
                for fc in resp.tool_call.function_calls:
                    output = my_dispatch(fc.name, dict(fc.args))
                    results.append(types.FunctionResponse(
                        id=fc.id, name=fc.name, response=output
                    ))
                await session.send_tool_response(function_responses=results)
                break   # ← CRITICAL: re-enter loop, don't wait for turn_complete

            if resp.server_content and resp.server_content.turn_complete:
                break   # final answer received
```

**The deadlock you will hit if you miss the `break`**: After sending a `tool_call`, the server waits for your tool response before it sends `turn_complete`. If your code is also waiting for `turn_complete` before sending the tool response — neither side moves and the session hangs forever. Always `break` immediately when you see `tool_call`, send your response, then re-enter the receive loop.

**Declaring a tool**:
```python
tools = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_weather",
        description="Get current weather for a city. Use this for any weather question.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"city": types.Schema(type=types.Type.STRING, description="City name")},
            required=["city"],
        ),
    )
])]
```

The description field matters — Gemini uses it to decide when to call the tool. Write it like a docstring for the model, not for a human reader.

---

### 04 — Multilingual + Singapore English
**`04_multilingual.ipynb`**

**Core capability proved**: A single Live API agent handles 70+ languages natively — no translation layer, no separate models, no preprocessing. The model detects language automatically or follows a configured language code, and switches mid-session if the user changes language.

**What this unlocks for agent builders**: You can deploy one agent for a global audience. A customer support agent built in English can serve users in Hindi, Japanese, Arabic, or Spanish without any architectural changes. One system prompt, one model, 70 languages. The Singapore English / Singlish demo shows how to go even further — configuring not just a language but a specific dialect and cultural register.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| 5 languages | "Does the model actually respond correctly in non-English languages?" | Prove multilingual capability before committing to a global rollout |
| Language switching | "If a user switches languages mid-conversation, does the model follow?" | Multilingual support agents — user starts in English, switches to Hindi |
| Language detection | "Do I need to tell the model what language the user is speaking?" | No — the model detects it. Useful for unknown-language-user scenarios |
| Voice + language_code | "How do I configure the TTS voice for a specific language accent?" | Localised voice assistants — correct accent per language |
| en-SG + Singlish | "Can I configure a specific dialect or cultural speech style?" | Hyper-localised agents — regional customer support, local market chatbots |

**Language + voice configuration** — two fields, full localisation:
```python
speech_config=types.SpeechConfig(
    language_code="en-SG",      # BCP-47 code — sets TTS accent
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
    ),
)
```

**Zero-config language detection**: if you don't set `language_code`, the model detects and responds in whatever language the user speaks. Send a Hindi greeting, get a Hindi response. Send Arabic, get Arabic. No routing logic needed.

**Dialect via system instruction**: The Singlish demo shows that `language_code` sets pronunciation, but vocabulary and style come from the system prompt. This pattern works for any dialect, register, or cultural context:
```python
system_instruction=(
    "You are a friendly Singaporean. Respond ONLY in Singlish — "
    "use particles like lah, lor, sia, hor, wah naturally in every response."
)
# Live output: "Wah, Singapore food best one lah! Must try chicken rice, confirm tasty lor."
```

**Key language codes for Asia-Pacific** (most relevant for Singapore-based deployments):
| Language | Code | Language | Code |
|---|---|---|---|
| English (SG) | `en-SG` | English (US) | `en-US` |
| Mandarin | `cmn-CN` | Hindi | `hi-IN` |
| Malay | `ms-MY` | Tamil | `ta-IN` |
| Japanese | `ja-JP` | Korean | `ko-KR` |
| Thai | `th-TH` | Indonesian | `id-ID` |

---

### 05 — Barge-In / Voice Activity Detection
**`05_barge_in.ipynb`**

**Core capability proved**: Gemini detects when a user interrupts a long response and stops speaking immediately, flagging the interruption so your application can react. This is what makes a voice agent feel like a real conversation rather than a turn-taking system.

**What this unlocks for agent builders**: Without barge-in handling, users must wait for the agent to finish speaking before they can respond — frustrating in any real conversation. With barge-in, users can interrupt mid-sentence, change the subject, or ask a follow-up naturally. This is the difference between a robotic IVR system and a natural-feeling voice agent.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| Text barge-in | "What happens to a long response when I send a new input?" | Understand the basic interrupt-and-redirect flow |
| Audio interruption | "How do I detect that the user spoke while Gemini was talking?" | Voice agents — stop playing audio immediately when user speaks |
| Full barge-in loop | "How do I build the complete listen → speak → interrupt state machine?" | Production voice agents — the complete conversation state machine |

**Detecting an interruption in the receive loop**:
```python
async for resp in session.receive():
    if resp.server_content:
        if resp.server_content.interrupted:
            # Gemini stopped mid-sentence — user took the floor
            player.stop()           # stop playing the audio buffer immediately
            audio_buffer.clear()    # discard any buffered audio that wasn't played
            # now listen for what the user says next
            break
        if resp.server_content.output_transcription:
            player.queue(resp.server_content.output_transcription.text)
        if resp.server_content.turn_complete:
            break
```

**The complete conversation state machine**:
```
IDLE ──► user speaks ──► LISTENING
LISTENING ──► turn_complete ──► SPEAKING (play audio)
SPEAKING ──► sc.interrupted ──► INTERRUPTED (stop audio, clear buffer)
INTERRUPTED ──► user speaks ──► LISTENING
```

**Triggering a barge-in programmatically** (for testing without a mic):
```python
# Start a long response
await session.send_realtime_input(text="Tell me a very long story about...")

# 2 seconds later, interrupt:
await session.send_realtime_input(activity_start=types.ActivityStart())
# send short audio chunk
await session.send_realtime_input(activity_end=types.ActivityEnd())
await session.send_realtime_input(text="Actually, just give me the summary.")
# sc.interrupted fires, Gemini pivots to answer the new question
```

---

### 06 — Proactive Audio / Push-to-Talk
**`06_proactive_audio.ipynb`**

**Core capability proved**: You can disable Gemini's automatic Voice Activity Detection and take full control of when audio input "counts" as a user turn — bracketing it explicitly with start and end signals. This gives your application precise control over the conversation flow.

**What this unlocks for agent builders**: Auto-VAD works well in quiet environments, but real deployments are messier. Noisy call centers, hands-free devices in cars, walkie-talkie UIs, accessibility tools where the user has long pauses — all of these need explicit control over when Gemini should respond. Manual activity detection is also the correct pattern when you're sending non-speech audio (audio files, synthetic audio, music) since VAD won't recognise these as speech.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| Automatic VAD | "What does the default mode look like — how does Gemini know when I've stopped speaking?" | Understand the baseline before deciding whether to override it |
| Manual PTT | "How do I build a push-to-talk button that controls exactly when Gemini listens?" | Walkie-talkie agents, call center UIs, accessibility interfaces |
| VAD sensitivity tuning | "In a noisy environment, how do I stop VAD from false-triggering?" | Field agents, contact centers, any deployment with background noise |

**The push-to-talk pattern** — two signals around your audio:
```python
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
    ),
)

# When PTT button is pressed:
await session.send_realtime_input(activity_start=types.ActivityStart())

# Stream audio chunks for as long as button is held:
async for chunk in mic_stream:
    await session.send_realtime_input(
        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
    )
    if not button_held:
        break

# When PTT button is released:
await session.send_realtime_input(activity_end=types.ActivityEnd())
# Gemini now processes everything between ActivityStart and ActivityEnd as one turn
```

This pattern is also required whenever you send synthetic or non-speech audio — VAD won't fire on tones or music, so you must bracket manually. This is why every audio-sending demo in these notebooks uses `ActivityStart`/`ActivityEnd`.

---

### 07 — Affective Dialog / Persona Voices
**`07_affective_dialog.ipynb`**

**Core capability proved**: You can shape the emotional register, personality, and vocal character of a Live API agent entirely through the system prompt and voice selection — no fine-tuning required. The same underlying model delivers completely different experiences depending on how you configure it.

**What this unlocks for agent builders**: Brand voice is a real product requirement. A mental health support agent must sound warm and empathetic. A sales agent should be confident and energetic. A legal assistant should be measured and precise. This notebook shows how to achieve all of these with system prompts alone, and how to match the right TTS voice to the right persona.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| 4 tonal personas | "Can I make the agent sound warm? Professional? Authoritative? Just from a prompt?" | Brand-matched agents — customer service, coaching, healthcare, legal |
| Emotion-adaptive | "Can the agent detect the user's emotional state and adjust its tone?" | Empathetic support agents — escalate calm when user is frustrated |
| All 5 voices | "Which voice fits my use case? What do they actually sound like?" | Voice selection — hear all 5 on the same text before committing |

**Available voices — matched to use cases**:
| Voice | Character | Best for |
|-------|-----------|---------|
| `Puck` | Bright, upbeat, friendly | Consumer apps, retail, casual assistants |
| `Aoede` | Warm, melodic, natural | Healthcare, mental health, HR support |
| `Kore` | Clear, professional, neutral | Enterprise tools, legal, finance |
| `Charon` | Deep, authoritative | Security briefings, formal announcements |
| `Fenrir` | Expressive, dynamic | Sales, coaching, high-energy engagement |

**Defining a persona via system instruction**:
```python
# Warm counsellor
system_instruction=(
    "You are a warm, empathetic counsellor. "
    "Always acknowledge the user's feelings before offering any advice. "
    "Speak gently, use encouraging language, and never rush the conversation."
)
speech_config = voice("Aoede")   # warm, melodic — matches the persona

# Confident sales coach
system_instruction=(
    "You are an energetic sales coach. "
    "Be direct, action-oriented, and use motivating language. "
    "Always end with a clear next step."
)
speech_config = voice("Fenrir")  # expressive, dynamic — matches the energy
```

**Emotion-adaptive pattern** — adjust tone based on what the user says:
```python
system_instruction=(
    "You are a customer support agent. "
    "If the user sounds frustrated or upset, shift to a calm, empathetic tone and slow down. "
    "If the user is happy and engaged, match their energy. "
    "Always prioritise making the user feel heard before solving the problem."
)
```

The model will read sentiment from the content of messages and adjust accordingly — no sentiment analysis API required.

---

### 08 — Google Search + Session Management
**`08_google_search_and_sessions.ipynb`**

**Core capability proved**: Two production-critical capabilities. First: Gemini can ground its answers in real-time Google Search results — giving accurate, up-to-date answers to questions its training data can't answer. Second: you can build resilient sessions that survive connection expiry, reconnect automatically, and preserve conversation context across reconnects.

**What this unlocks for agent builders**: Together these two capabilities close the gap between a demo and a production deployment. Google Search means your agent isn't limited by a knowledge cutoff — it can answer "what's the latest news on X?" or "what is the current price of Y?". Session management means your agent handles the ~15-minute session limit gracefully instead of crashing.

| Demo | Developer question it answers | Agent use case |
|------|------------------------------|---------------|
| Google Search grounding | "Can the agent answer questions about current events, prices, recent news?" | Any agent needing fresh data: news, prices, sports, weather, company info |
| Search + function calling | "Can I combine Google Search with my own APIs in one session?" | Hybrid agents — grounded facts from Search + proprietary data from your APIs |
| GoAway reconnect | "What happens when the session expires after 15 minutes?" | Production agents — handle session limits without dropping the conversation |
| Context-preserving sessions | "After a reconnect, does the agent remember the earlier conversation?" | Long-running agents — inject history into the new session's system prompt |
| LiveSessionManager class | "Is there a production-ready wrapper for all of this?" | Copy this class into your production agent — handles reconnect, backoff, and context |

**Google Search — one line to enable, no client round-trips**:
```python
tools = [types.Tool(google_search=types.GoogleSearch())]
# The model calls Search internally on the server.
# You never see the raw search queries or results.
# You get the final synthesised, grounded answer.
config = types.LiveConnectConfig(tools=tools, ...)
```
Real output: *"The current population of Singapore is approximately 5.9 million as of 2024..."* — accurate, cited, no knowledge cutoff.

**Combining Search with your own functions** — both tools active in the same session:
```python
tools = [
    types.Tool(google_search=types.GoogleSearch()),   # real-time web data
    types.Tool(function_declarations=[get_weather]),  # your own API
]
# Gemini decides which tool to use for each part of the query.
# "Search for AI news AND get Singapore weather" → uses both, in one response.
```

**GoAway — graceful session expiry handling**:
```python
# resp.go_away fires ~60 seconds before the session closes
# Use it to finish the current turn and reconnect
async for resp in session.receive():
    if resp.go_away:
        save_state()    # persist anything you need
        break           # exit the session — outer loop reconnects

# On reconnect, inject previous conversation as context:
history = "\n".join(f"User: {t['q']}\nAssistant: {t['a']}" for t in log[-5:])
new_config = types.LiveConnectConfig(
    system_instruction=f"Previous conversation:\n{history}\n\nContinue from here.",
    ...
)
```

**`LiveSessionManager`** — the production-ready class from the notebook:
```python
mgr = LiveSessionManager(config=base_config, max_reconnects=5)

# Handles reconnect, exponential backoff, and context injection automatically
answer = await mgr.ask("What did we discuss earlier?")
# Returns the transcript — agent remembers even after a reconnect
```

---

## Common Pitfalls & Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Session hangs, 1011 error | `response_modalities=["TEXT"]` — not supported | Use `["AUDIO"]` + `output_audio_transcription` for text |
| 1008 policy violation | `send_client_content` used in a Live session | Replace with `send_realtime_input` everywhere |
| 1007 precondition failed | Sending audio as a single large blob | Stream in 512-sample chunks |
| Session hangs after sending audio | VAD doesn't recognise synthetic/non-speech audio | Disable VAD + bracket with `ActivityStart`/`ActivityEnd` |
| Tool call session hangs forever | Waiting for `turn_complete` before sending tool response | `break` on `tool_call`, send response, re-enter loop |
| `nest_asyncio` error | `.patch()` doesn't exist in this version | Use `nest_asyncio.apply()` |
| Agent loses context after reconnect | New session starts fresh with no history | Inject last N turns into new session's `system_instruction` |

---

## Architecture Reference

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Voice Agent                         │
│                                                             │
│  SEND:                                                      │
│    send_realtime_input(text=...)          ← text turns      │
│    send_realtime_input(audio=Blob(...))   ← mic / audio     │
│    send_realtime_input(activity_start=ActivityStart())      │
│    send_realtime_input(activity_end=ActivityEnd())          │
│    send_tool_response(function_responses=[...])             │
│                                                             │
│  RECEIVE:                                                   │
│    resp.data                         → PCM16 audio 24kHz   │
│    resp.server_content                                      │
│      .output_transcription.text      → Gemini's words      │
│      .input_transcription.text       → what model heard    │
│      .turn_complete                  → Gemini finished      │
│      .interrupted                    → user barged in       │
│    resp.tool_call                    → call your function   │
│    resp.go_away                      → reconnect soon       │
└────────────────────────┬────────────────────────────────────┘
                         │  Persistent WebSocket
┌────────────────────────▼────────────────────────────────────┐
│              gemini-3.1-flash-live-preview                   │
│                                                             │
│  Built-in: VAD · 70+ languages · 5 TTS voices              │
│  Tools:    Google Search · Your function declarations       │
│  Context:  Full conversation history for session lifetime   │
└─────────────────────────────────────────────────────────────┘
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
client  = genai.Client(api_key=API_KEY)
```

### Jupyter async compatibility
```python
import nest_asyncio
nest_asyncio.apply()          # must be first — patches Jupyter's event loop
asyncio.run(my_coroutine())   # then works normally
```

---

## References

- [Gemini Live API Docs](https://ai.google.dev/api/live)
- [google-genai Python SDK](https://github.com/googleapis/python-genai)
- [AI Studio — get API key](https://aistudio.google.com/apikey)
- [BCP-47 language codes](https://www.iana.org/assignments/language-subtag-registry)
