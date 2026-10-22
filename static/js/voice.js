// Voice capture: browser speech recognition (live transcript) or MediaRecorder (server STT),
// plus a microphone level visualiser shared by both modes.

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

export class Voice {
  constructor({ canvas, onInterim, onFinal, onAudio, onState, onError }) {
    Object.assign(this, { canvas, onInterim, onFinal, onAudio, onState, onError });
    this.active = false;
  }

  get browserSupported() { return Boolean(SR); }

  async start(mode, { continuous = false } = {}) {
    if (this.active) return;
    this.mode = mode === "browser" && SR ? "browser" : "server";
    this.active = true;
    this.onState?.(true, this.mode);
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      this.active = false;
      this.onState?.(false, this.mode);
      this.onError?.("Microphone permission was denied or no microphone is available.");
      return;
    }
    this._visualise();
    if (this.mode === "browser") {
      const rec = new SR();
      rec.lang = "en-IN";
      rec.interimResults = true;
      rec.continuous = continuous;
      this.finalText = "";
      rec.onresult = (ev) => {
        let interim = "";
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const r = ev.results[i];
          if (r.isFinal) this.finalText += r[0].transcript + " ";
          else interim += r[0].transcript;
        }
        this.onInterim?.((this.finalText + interim).trim());
      };
      rec.onerror = (ev) => { if (ev.error !== "no-speech" && ev.error !== "aborted") this.onError?.(`Speech recognition: ${ev.error}`); };
      rec.onend = () => { const t = this.finalText.trim(); this._teardown(); if (t) this.onFinal?.(t); };
      this.rec = rec;
      rec.start();
    } else {
      const chunks = [];
      this.recorder = new MediaRecorder(this.stream);
      this.recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      this.recorder.onstop = () => {
        const blob = new Blob(chunks, { type: this.recorder.mimeType || "audio/webm" });
        this._teardown();
        if (blob.size > 2000) this.onAudio?.(blob);
      };
      this.recorder.start();
    }
  }

  stop() {
    if (!this.active) return;
    if (this.rec) this.rec.stop();
    else if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
    else this._teardown();
  }

  _teardown() {
    this.active = false;
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.audioCtx?.close().catch(() => {});
    this.rec = null; this.recorder = null; this.stream = null; this.audioCtx = null;
    const ctx = this.canvas.getContext("2d");
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.onState?.(false, this.mode);
  }

  _visualise() {
    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = this.audioCtx.createMediaStreamSource(this.stream);
      const an = this.audioCtx.createAnalyser();
      an.fftSize = 256;
      src.connect(an);
      const data = new Uint8Array(an.frequencyBinCount);
      const c = this.canvas;
      const ctx = c.getContext("2d");
      const color = getComputedStyle(document.documentElement).getPropertyValue("--bad").trim() || "#c0362c";
      const draw = () => {
        const dpr = window.devicePixelRatio || 1;
        c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
        an.getByteFrequencyData(data);
        ctx.clearRect(0, 0, c.width, c.height);
        const bars = 48, bw = c.width / bars;
        ctx.fillStyle = color;
        for (let i = 0; i < bars; i++) {
          const v = data[Math.floor((i / bars) * data.length * 0.7)] / 255;
          const h = Math.max(2 * dpr, v * c.height * 0.9);
          ctx.fillRect(i * bw + bw * 0.2, (c.height - h) / 2, bw * 0.6, h);
        }
        this.raf = requestAnimationFrame(draw);
      };
      draw();
    } catch (e) { /* visualiser is optional */ }
  }
}

export function speak(text) {
  if (!window.speechSynthesis || !text) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.replace(/\[[A-Z]\d+\]/g, ""));
  u.lang = "en-IN";
  u.rate = 1.05;
  speechSynthesis.speak(u);
}
