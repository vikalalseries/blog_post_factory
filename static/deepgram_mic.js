/* Voice typing for a text box. One plain file, no libraries. Works on Vercel.
 *
 *   const mic = createMic({
 *     input:    document.querySelector("#topic"),    // <textarea> or <input>
 *     button:   document.querySelector("#micBtn"),   // the mic toggle button
 *     onError:  msg => toast(msg),                   // optional
 *     onChange: () => autoGrow(),                    // optional, called after the text changes
 *   });
 *   mic.active()  -> true while recording (or finishing the last piece)
 *   mic.stop()    -> stop and keep the text      mic.cancel() -> stop and restore the old text
 *
 * Tap the button to start, tap again to stop. Esc cancels. While recording the button has
 * class "rec" and aria-pressed="true". The button is hidden if the browser cannot record.
 * The microphone only works on localhost or HTTPS.
 *
 * Two modes, chosen automatically:
 *  1) LIVE: POST /api/voice-token gives a 60 s temporary token and the browser streams straight
 *     to Deepgram over a WebSocket (the real API key never reaches the browser).
 *  2) PIECES (fallback, works with any Deepgram key): if the token request fails, short pieces
 *     are cut at natural pauses and each one is POSTed to /api/transcribe.
 */
function createMic({ input, button, onError = () => {}, onChange = () => {} }) {
  const idle = { stop() {}, cancel() {}, active: () => false };
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder)) {
    button.hidden = true; button.style.display = "none"; return idle;
  }

  const LIVE_URL = "wss://api.deepgram.com/v1/listen?model=nova-3&language=multi&interim_results=true&smart_format=true&punctuate=true";
  const MIN_PIECE_MS = 2500, QUIET_MS = 350, MAX_PIECE_MS = 6500, QUIET_LEVEL = 4, MIN_BYTES = 1000;
  let session = null;        // the current recording, or null
  let pieceModeOnly = false; // token failed once: don't ask again until the page reloads

  const ui = on => {
    button.classList.toggle("rec", on);
    button.setAttribute("aria-pressed", String(on));
    button.setAttribute("aria-label", on ? "Stop recording" : "Speak instead of typing");
  };
  const fail = (s, msg) => { if (!s.errShown) { s.errShown = true; onError(msg); } };   // one error per session
  const show = (s, interim = "", force = false) => {
    if (s.cancelled || (s.done && !force)) return;       // results that arrive after the session ended are ignored
    const finished = s.parts.filter(Boolean).join(" ");
    input.value = [s.base, finished, interim].filter(Boolean).join(" ");
    onChange(); input.scrollTop = input.scrollHeight;
  };

  // Release everything and (unless cancelled) leave the text in the box. Safe to call twice.
  function finish(s) {
    if (s.done) return;
    s.done = true;
    clearInterval(s.timer); clearTimeout(s.closeTimer);
    try { s.rec && s.rec.state !== "inactive" && s.rec.stop(); } catch {}
    try { s.ws && s.ws.readyState < 2 && s.ws.close(); } catch {}
    try { s.ctx && s.ctx.state !== "closed" && s.ctx.close(); } catch {}
    s.stream.getTracks().forEach(t => t.stop());          // turns the browser's recording light off
    if (session === s) session = null;
    ui(false);
    if (!s.cancelled) { show(s, "", true); input.focus(); }
  }

  // ---------- MODE 1: live streaming straight to Deepgram ----------
  function startLive(s, token) {
    let opened = false;
    const ws = s.ws = new WebSocket(LIVE_URL, ["bearer", token]);
    ws.onopen = () => {
      opened = true;
      try {
        const rec = s.rec = new MediaRecorder(s.stream);
        rec.ondataavailable = e => { if (e.data.size && ws.readyState === 1) ws.send(e.data); };
        rec.start(250);                                     // 250 ms chunks = live
      } catch { fail(s, "Could not start recording."); finish(s); }
    };
    ws.onmessage = e => {
      let m; try { m = JSON.parse(e.data); } catch { return; }
      if (m.type !== "Results") return;
      const text = ((m.channel && m.channel.alternatives && m.channel.alternatives[0] && m.channel.alternatives[0].transcript) || "").trim();
      if (!text) return;
      if (m.is_final) { s.parts.push(text); show(s); } else show(s, text);
    };
    ws.onerror = () => {
      if (s.done) return;
      if (!opened && !s.stopping) {                         // could not connect: use short pieces instead
        pieceModeOnly = true; s.ws = null; startPieces(s);
      } else { fail(s, "The speech connection failed."); finish(s); }
    };
    ws.onclose = () => { if (s.mode === "live") finish(s); };
    s.mode = "live";
  }
  function stopLive(s) {
    try { s.rec && s.rec.state !== "inactive" && s.rec.stop(); } catch {}
    const ws = s.ws;
    if (ws && ws.readyState === 1) {
      try { ws.send(JSON.stringify({ type: "Finalize" })); ws.send(JSON.stringify({ type: "CloseStream" })); } catch {}
      s.closeTimer = setTimeout(() => finish(s), 4000);     // close ourselves if Deepgram has not
    } else finish(s);
  }

  // ---------- MODE 2: short pieces cut at pauses, each sent to /api/transcribe ----------
  function startPieces(s) {
    s.mode = "pieces"; s.pending = 0; s.next = 0; s.recorderDone = false;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    let analyser = null, buf = null;
    if (Ctx) {
      try {
        s.ctx = new Ctx();
        const src = s.ctx.createMediaStreamSource(s.stream);
        analyser = s.ctx.createAnalyser(); analyser.fftSize = 1024;
        src.connect(analyser); buf = new Uint8Array(analyser.fftSize);
      } catch { analyser = null; }
    }
    let rec = null, pieceStart = 0, quietSince = 0;
    const settled = () => { if (s.recorderDone && s.pending === 0) finish(s); };

    async function send(blob, idx) {
      s.pending++;
      try {
        const r = await fetch("/api/transcribe", { method: "POST", headers: { "Content-Type": blob.type || "audio/webm" }, body: blob });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Transcription failed");
        const { text } = await r.json();
        if (!s.cancelled && text) { s.parts[idx] = text.trim(); show(s); }
      } catch (e) { fail(s, e.message || "Transcription failed"); }
      s.pending--; settled();
    }
    function record() {
      const chunks = [], idx = s.next++;
      rec = s.rec = new MediaRecorder(s.stream);
      rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      rec.onstop = () => {
        const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        if (!s.stopping && !s.cancelled && !s.done) record();   // start the next piece at once: no speech is lost
        else s.recorderDone = true;
        if (!s.cancelled && blob.size >= MIN_BYTES) send(blob, idx); else settled();
      };
      rec.start(); pieceStart = Date.now(); quietSince = 0;
    }
    s.timer = setInterval(() => {
      if (!rec || rec.state !== "recording") return;
      const now = Date.now(), len = now - pieceStart;
      let quiet = false;
      if (analyser) {
        analyser.getByteTimeDomainData(buf);
        let max = 0; for (let i = 0; i < buf.length; i++) max = Math.max(max, Math.abs(buf[i] - 128));
        quiet = max < QUIET_LEVEL;
      }
      if (quiet) { if (!quietSince) quietSince = now; } else quietSince = 0;
      const pause = quietSince && now - quietSince >= QUIET_MS;
      if ((len >= MIN_PIECE_MS && pause) || len >= MAX_PIECE_MS) { try { rec.stop(); } catch {} }
    }, 50);
    record();
  }
  function stopPieces(s) {
    clearInterval(s.timer);
    if (s.rec && s.rec.state === "recording") { try { s.rec.stop(); } catch {} }   // uploads the last piece
    else { s.recorderDone = true; if (s.pending === 0) finish(s); }
  }

  // ---------- start / stop ----------
  async function start() {
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch { onError("Microphone permission was denied."); return; }
    const s = session = { stream, base: input.value.trim(), parts: [], cancelled: false, stopping: false, done: false, errShown: false, mode: "" };
    ui(true);
    if (!pieceModeOnly) {
      try {
        const r = await fetch("/api/voice-token", { method: "POST" });
        const token = r.ok ? (await r.json()).token : null;
        if (!token) throw new Error("no token");
        if (s.done || s.cancelled || s.stopping) { if (!s.done) finish(s); return; }
        startLive(s, token); return;
      } catch { pieceModeOnly = true; }                     // remember: don't retry until reload
    }
    if (s.done || s.cancelled || s.stopping) { if (!s.done) finish(s); return; }
    startPieces(s);
  }
  function stop() {
    const s = session; if (!s || s.stopping) return;
    s.stopping = true; ui(false);
    if (s.mode === "live") stopLive(s);
    else if (s.mode === "pieces") stopPieces(s);
    else finish(s);                                          // stopped before a mode was chosen
  }
  function cancel() {
    const s = session; if (!s) return;
    s.cancelled = true; s.stopping = true;
    input.value = s.base; onChange();                        // restore what was there before
    finish(s);
  }

  button.addEventListener("click", () => (session ? stop() : start()));
  document.addEventListener("keydown", e => { if (e.key === "Escape" && session) cancel(); });
  return { stop, cancel, active: () => !!session };
}
