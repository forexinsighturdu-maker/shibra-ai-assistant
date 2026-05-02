import asyncio
import logging
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import uvicorn

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shibra-live")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

app = FastAPI(title="SHIBRA Live Voice Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are SHIBRA AI Assistant for Shahid Mehmood.

Main behaviour:
- Speak in natural Urdu/Hindi mix, with a Pakistani Urdu style and accent.
- Keep replies short, clear, and conversational.
- User is talking by voice, so respond like a live phone-call assistant.
- Do not give long lectures unless the user asks.
- If user asks coding, explain briefly and give practical steps.
- Address the user respectfully as "sir" sometimes, but do not overuse it.
""".strip()


HTML_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SHIBRA AI Assistant - Live Urdu Voice</title>

  <style>
    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: #eaf8ff;
      background:
        radial-gradient(circle at center, rgba(0, 216, 255, 0.18), transparent 34%),
        radial-gradient(circle at 10% 10%, rgba(255, 215, 100, 0.12), transparent 28%),
        linear-gradient(135deg, #06131d 0%, #081a26 48%, #02070c 100%);
      overflow-x: hidden;
    }

    .wrap {
      width: min(1450px, 96vw);
      margin: 0 auto;
      padding: 28px 0 18px;
    }

    .grid-top,
    .grid-main {
      display: grid;
      gap: 22px;
    }

    .grid-top {
      grid-template-columns: repeat(3, 1fr);
      align-items: start;
    }

    .grid-main {
      grid-template-columns: 1fr 1.22fr 1fr;
      align-items: center;
      margin-top: 24px;
    }

    .card {
      border: 1px solid rgba(121, 221, 255, 0.35);
      background: rgba(7, 24, 36, 0.74);
      box-shadow: 0 0 28px rgba(0, 195, 255, 0.12);
      border-radius: 18px;
      padding: 19px;
      min-height: 150px;
      backdrop-filter: blur(10px);
    }

    .card.active {
      border-color: rgba(255, 215, 105, 0.8);
      box-shadow: 0 0 34px rgba(255, 215, 105, 0.22);
    }

    .card h3 {
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: .35px;
    }

    .card p {
      margin: 0 0 13px;
      color: #b9d9e8;
      line-height: 1.45;
      font-size: 14px;
    }

    .badge {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(255, 215, 105, 0.12);
      border: 1px solid rgba(255, 215, 105, 0.35);
      color: #ffe59a;
      font-size: 13px;
      margin: 3px 4px 0 0;
    }

    .orb {
      width: min(355px, 78vw);
      height: min(355px, 78vw);
      margin: 12px auto;
      border-radius: 50%;
      border: 2px solid rgba(121, 221, 255, 0.88);
      background:
        radial-gradient(circle, rgba(255, 215, 105, 0.20), transparent 43%),
        linear-gradient(145deg, rgba(10, 40, 55, 0.96), rgba(3, 10, 16, 0.96));
      box-shadow:
        0 0 42px rgba(0, 220, 255, 0.34),
        inset 0 0 42px rgba(255, 215, 105, 0.13);
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      position: relative;
    }

    .orb::before,
    .orb::after {
      content: "";
      position: absolute;
      border-radius: 50%;
      border: 1px solid rgba(255,255,255,.15);
      inset: 18px;
    }

    .orb::after {
      inset: 38px;
      border-color: rgba(255, 215, 105, .18);
    }

    .orb h1 {
      color: #f4e3a1;
      font-size: 38px;
      margin: 0 0 5px;
    }

    .orb h2 {
      color: #eaf8ff;
      font-size: 28px;
      margin: 0;
    }

    .orb p {
      color: #94c9dc;
      font-size: 13px;
      margin: 8px 0 0;
    }

    .status-bar {
      border: 1px solid rgba(255, 215, 105, 0.45);
      border-radius: 999px;
      padding: 11px 18px;
      text-align: center;
      color: #cbefff;
      background: rgba(5, 18, 26, 0.72);
      margin-top: 22px;
    }

    .call-panel {
      margin: 25px auto 0;
      border: 1px solid rgba(0, 220, 255, 0.42);
      background: rgba(3, 16, 25, 0.80);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 0 30px rgba(0, 220, 255, 0.15);
    }

    .call-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: stretch;
    }

    button {
      border: none;
      border-radius: 16px;
      padding: 15px 18px;
      font-size: 17px;
      font-weight: 700;
      cursor: pointer;
      color: #06131d;
      background: linear-gradient(135deg, #ffe59a, #60e9ff);
      box-shadow: 0 0 22px rgba(96, 233, 255, .18);
    }

    button.secondary {
      background: rgba(255,255,255,.08);
      color: #eaf8ff;
      border: 1px solid rgba(255,255,255,.18);
    }

    button:disabled {
      opacity: .5;
      cursor: not-allowed;
    }

    .live-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #777;
      display: inline-block;
      margin-right: 8px;
      box-shadow: none;
    }

    .live-dot.on {
      background: #63ffb0;
      box-shadow: 0 0 16px rgba(99,255,176,.75);
    }

    .log {
      margin-top: 16px;
      background: rgba(255,255,255,.04);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 16px;
      padding: 14px;
      min-height: 125px;
      max-height: 245px;
      overflow-y: auto;
      color: #cfeaf5;
      font-size: 14px;
      line-height: 1.55;
    }

    .small {
      color: #9fc9da;
      font-size: 13px;
      margin-top: 10px;
    }

    @media (max-width: 980px) {
      .grid-top,
      .grid-main,
      .call-row {
        grid-template-columns: 1fr;
      }

      .card {
        min-height: auto;
      }
    }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="grid-top">
      <div class="card">
        <h3>👤 PERSONALIZED GREETING</h3>
        <p>Assalamu Alaikum sir Shahid Mehmood, main Shibra hazir hoon.</p>
        <span class="badge">Personal Assistant</span>
      </div>

      <div class="card active">
        <h3>🎙️ LIVE URDU VOICE CALL</h3>
        <p>No recording. Direct live call mode: Listen → Understand → Speak.</p>
        <span class="badge">Active Feature</span>
      </div>

      <div class="card">
        <h3>📷 LIVE VISION & OBJECT DETECTION</h3>
        <p>Camera analysis, document scanner, object detection — next feature.</p>
        <span class="badge">Coming Soon</span>
      </div>
    </div>

    <div class="grid-main">
      <div>
        <div class="card">
          <h3>💬 QUICK ACTIONS & MESSAGING</h3>
          <p>WhatsApp direct, Telegram shortcut, quick reply automation.</p>
          <span class="badge">Coming Soon</span>
        </div>

        <br>

        <div class="card">
          <h3>💻 EXPERT CODING & WRITING</h3>
          <p>Python scripting, Pine Script, YouTube scripts, automation coding.</p>
          <span class="badge">Coming Soon</span>
        </div>
      </div>

      <div class="orb">
        <div>
          <h1>🎙️ SHIBRA</h1>
          <h2>AI ASSISTANT</h2>
          <p>INTELLIGENT COMMAND CENTER</p>
          <p>FOR SHAHID MEHMOOD</p>
        </div>
      </div>

      <div>
        <div class="card">
          <h3>🎥 VIDEO REPORTING</h3>
          <p>Automatic content summary, scene interpretation, reporting dashboard.</p>
          <span class="badge">Coming Soon</span>
        </div>

        <br>

        <div class="card">
          <h3>⚙️ SMART AUTOMATION</h3>
          <p>Browser automation, device controls, command-based workflow actions.</p>
          <span class="badge">Coming Soon</span>
        </div>
      </div>
    </div>

    <div class="status-bar">
      STATUS: <b id="statusText">READY</b> &nbsp; | &nbsp; LOCATION: <b>PAKISTAN</b>
    </div>

    <div class="call-panel">
      <h2><span id="dot" class="live-dot"></span>Live Voice Call with SHIBRA</h2>

      <div class="call-row">
        <button id="startBtn">📞 Start Live Call</button>
        <button id="stopBtn" class="secondary" disabled>⛔ End Call</button>
      </div>

      <div class="small">
        Tip: Urdu mein naturally baat karein. AI response audio speaker se ayega. Headphones use karein to echo kam hoga.
      </div>

      <div id="log" class="log">System ready. Start Live Call press karein.</div>
    </div>
  </div>

<script>
let ws = null;
let audioContext = null;
let micStream = null;
let workletNode = null;
let nextPlayTime = 0;
let playingSources = [];

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const logBox = document.getElementById("log");
const statusText = document.getElementById("statusText");
const dot = document.getElementById("dot");

function log(msg) {
  const time = new Date().toLocaleTimeString();
  logBox.innerHTML += `<div><b>${time}</b> — ${msg}</div>`;
  logBox.scrollTop = logBox.scrollHeight;
}

function setLive(on, text) {
  dot.classList.toggle("on", on);
  statusText.textContent = text;
}

function downsampleBuffer(buffer, sampleRate, outSampleRate) {
  if (outSampleRate === sampleRate) return buffer;

  const ratio = sampleRate / outSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);

  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;

    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }

    result[offsetResult] = accum / count;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;

  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return buffer;
}

function playPCM24k(arrayBuffer) {
  if (!audioContext) return;

  const pcm16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(pcm16.length);

  for (let i = 0; i < pcm16.length; i++) {
    float32[i] = pcm16[i] / 32768;
  }

  const audioBuffer = audioContext.createBuffer(1, float32.length, 24000);
  audioBuffer.getChannelData(0).set(float32);

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);

  const now = audioContext.currentTime;
  nextPlayTime = Math.max(now, nextPlayTime);
  source.start(nextPlayTime);
  nextPlayTime += audioBuffer.duration;

  playingSources.push(source);

  source.onended = () => {
    playingSources = playingSources.filter(s => s !== source);
  };
}

function stopPlayback() {
  playingSources.forEach(s => {
    try { s.stop(); } catch(e) {}
  });

  playingSources = [];

  if (audioContext) {
    nextPlayTime = audioContext.currentTime;
  }
}

async function startCall() {
  try {
    startBtn.disabled = true;
    stopBtn.disabled = false;

    setLive(true, "CONNECTING");
    log("Connecting to SHIBRA...");

    const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${wsProtocol}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      log("Connected. Mic permission maang raha hai...");

      audioContext = new (window.AudioContext || window.webkitAudioContext)();

      await audioContext.audioWorklet.addModule("/pcm-processor.js");

      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1
        },
        video: false
      });

      const source = audioContext.createMediaStreamSource(micStream);
      workletNode = new AudioWorkletNode(audioContext, "pcm-worklet-processor");

      workletNode.port.onmessage = (event) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const downsampled = downsampleBuffer(
          event.data,
          audioContext.sampleRate,
          16000
        );

        const pcm = floatTo16BitPCM(downsampled);
        ws.send(pcm);
      };

      source.connect(workletNode);

      const mute = audioContext.createGain();
      mute.gain.value = 0;

      workletNode.connect(mute);
      mute.connect(audioContext.destination);

      ws.send(
        "Assalamu Alaikum. Live call start ho gayi hai. Sir ko Urdu Pakistani accent mein short greeting dein."
      );

      setLive(true, "LIVE CALL CONNECTED");
      log("Live call started. Ab directly Urdu mein baat karein.");
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === "user" && msg.text) {
            log(`You: ${msg.text}`);
          }

          if (msg.type === "gemini" && msg.text) {
            log(`SHIBRA: ${msg.text}`);
          }

          if (msg.type === "interrupted") {
            stopPlayback();
            log("Interrupted — SHIBRA stopped speaking.");
          }

          if (msg.type === "error") {
            log(`Error: ${msg.error}`);
          }
        } catch(e) {
          log(event.data);
        }
      } else {
        playPCM24k(event.data);
      }
    };

    ws.onerror = () => {
      log("WebSocket error. API key/model/server check karein.");
    };

    ws.onclose = () => {
      log("Call disconnected.");
      cleanup();
    };

  } catch (err) {
    log("Start error: " + err.message);
    cleanup();
  }
}

function cleanup() {
  setLive(false, "READY");

  startBtn.disabled = false;
  stopBtn.disabled = true;

  stopPlayback();

  if (workletNode) {
    try { workletNode.disconnect(); } catch(e) {}
    workletNode = null;
  }

  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }

  ws = null;
}

startBtn.onclick = startCall;

stopBtn.onclick = () => {
  log("Ending call...");
  cleanup();
};
</script>
</body>
</html>
"""


PCM_PROCESSOR_JS = r"""
class PCMWorkletProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];

    if (input && input.length > 0) {
      const channelData = input[0];
      this.port.postMessage(channelData.slice(0));
    }

    return true;
  }
}

registerProcessor("pcm-worklet-processor", PCMWorkletProcessor);
"""


@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.get("/pcm-processor.js")
async def pcm_processor():
    return PlainTextResponse(
        PCM_PROCESSOR_JS,
        media_type="application/javascript"
    )


async def gemini_live_session(websocket: WebSocket):
    if not GEMINI_API_KEY:
        await websocket.send_json({
            "type": "error",
            "error": "GEMINI_API_KEY missing. .env file mein key add karein."
        })
        await websocket.close()
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    audio_input_queue: asyncio.Queue[bytes] = asyncio.Queue()
    text_input_queue: asyncio.Queue[str] = asyncio.Queue()

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY"
        ),
    )

    async with client.aio.live.connect(model=MODEL, config=config) as session:

        async def receive_from_browser():
            try:
                while True:
                    message = await websocket.receive()

                    if message.get("bytes") is not None:
                        await audio_input_queue.put(message["bytes"])

                    elif message.get("text"):
                        await text_input_queue.put(message["text"])

            except WebSocketDisconnect:
                logger.info("Browser disconnected")

            except Exception as exc:
                logger.error("receive_from_browser error: %s", exc)

        async def send_audio_to_gemini():
            try:
                while True:
                    chunk = await audio_input_queue.get()

                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=chunk,
                            mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                        )
                    )

            except asyncio.CancelledError:
                pass

            except Exception as exc:
                logger.error("send_audio_to_gemini error: %s", exc)

        async def send_text_to_gemini():
            try:
                while True:
                    text = await text_input_queue.get()
                    await session.send_realtime_input(text=text)

            except asyncio.CancelledError:
                pass

            except Exception as exc:
                logger.error("send_text_to_gemini error: %s", exc)

        async def receive_from_gemini():
            try:
                async for response in session.receive():
                    server_content = getattr(response, "server_content", None)

                    if not server_content:
                        continue

                    if getattr(server_content, "interrupted", False):
                        await websocket.send_json({"type": "interrupted"})

                    model_turn = getattr(server_content, "model_turn", None)

                    if model_turn and getattr(model_turn, "parts", None):
                        for part in model_turn.parts:
                            inline_data = getattr(part, "inline_data", None)

                            if inline_data and inline_data.data:
                                await websocket.send_bytes(inline_data.data)

                    input_tr = getattr(
                        server_content,
                        "input_transcription",
                        None
                    )

                    if input_tr and getattr(input_tr, "text", None):
                        await websocket.send_json({
                            "type": "user",
                            "text": input_tr.text
                        })

                    output_tr = getattr(
                        server_content,
                        "output_transcription",
                        None
                    )

                    if output_tr and getattr(output_tr, "text", None):
                        await websocket.send_json({
                            "type": "gemini",
                            "text": output_tr.text
                        })

                    if getattr(server_content, "turn_complete", False):
                        await websocket.send_json({"type": "turn_complete"})

            except asyncio.CancelledError:
                pass

            except Exception as exc:
                logger.error(
                    "receive_from_gemini error: %s\n%s",
                    exc,
                    traceback.format_exc()
                )

                await websocket.send_json({
                    "type": "error",
                    "error": str(exc)
                })

        tasks = [
            asyncio.create_task(receive_from_browser()),
            asyncio.create_task(send_audio_to_gemini()),
            asyncio.create_task(send_text_to_gemini()),
            asyncio.create_task(receive_from_gemini()),
        ]

        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        await gemini_live_session(websocket)

    except Exception as exc:
        logger.error(
            "Gemini session failed: %s\n%s",
            exc,
            traceback.format_exc()
        )

        try:
            await websocket.send_json({
                "type": "error",
                "error": str(exc)
            })
        except Exception:
            pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
