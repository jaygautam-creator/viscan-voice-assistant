import { icon } from "./icons.js";
import { renderOdontogram } from "./odontogram.js";
import { Voice, speak } from "./voice.js";

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const parseDay = (iso) => { const [y, m, d] = iso.slice(0, 10).split("-").map(Number); return new Date(y, m - 1, d); };
const fmtDate = (iso) => { const d = parseDay(iso); return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`; };
const fmtDateLong = (iso) => { const d = parseDay(iso); return `${d.getDate()} ${MONTHS_LONG[d.getMonth()]} ${d.getFullYear()}`; };
const fmtSlot = (iso) => { const d = parseDay(iso); return `${DAYS[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]} · ${iso.slice(11, 16)}`; };
const initials = (name) => name.split(/\s+/).filter((w) => !/^dr\.?$/i.test(w)).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
const hue = (s) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 360, 7);
const avatar = (name, cls = "avatar") => `<span class="${cls}" style="background:hsl(${hue(name)} 42% 42%)">${esc(initials(name))}</span>`;
const modelName = (id) => {
  if (!id) return "";
  const m = id.match(/^gemini-(.+)$/);
  if (m) return "Gemini " + m[1].split("-").map((p) => (/^\d/.test(p) ? p : p[0].toUpperCase() + p.slice(1))).join(" ");
  const c = id.match(/^claude-(\w+)-(\d+)-(\d+)$/);
  if (c) return `Claude ${c[1][0].toUpperCase() + c[1].slice(1)} ${c[2]}.${c[3]}`;
  return id;
};
const engineLabel = (interp) => {
  if (!interp) return "";
  if (interp.startsWith("llm:")) return modelName(interp.split(":").pop());
  const fb = interp.match(/fallback: ([\w_]+)/);
  const why = { circuit_open: "LLM paused", rate_limited: "LLM rate-limited", timeout: "LLM timeout", llm_unavailable: "LLM unavailable" };
  return fb ? `Rules · ${why[fb[1]] || "LLM " + fb[1].replace(/_/g, " ")}` : "Rules engine";
};
const empty = (ic, title, text) => `<div class="pane-empty">${icon(ic)}<b>${title}</b>${text}</div>`;

// ------------------------------------------------------------------ state
const state = {
  clinician: store.get("viscan-clinician", "dr-anand"),
  clinicians: [],
  patients: [],
  health: null,
  sessionId: null,
  context: null,
  lastTurn: null,
  report: null, // { view, cited:Set, summary, answer }
  reportList: [],
  draft: null,
  draftEdit: null,
  draftJson: false,
  schedule: { modality: "CBCT", day: null, slots: [] },
  bookings: [],
  mode: "assistant",
  stt: "browser",
  tts: store.get("viscan-tts", "0") === "1",
  busy: false,
  turns: 0,
};

// ------------------------------------------------------------------ API
async function api(method, url, body, form = false) {
  const headers = { "X-Clinician-Id": state.clinician };
  const opts = { method, headers };
  if (body !== undefined) {
    if (form) opts.body = body;
    else { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  }
  const r = await fetch(url, opts);
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    const d = data?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `Request failed (${r.status})`;
    const e = new Error(msg);
    e.status = r.status;
    throw e;
  }
  return data;
}

function toast(msg, err = false) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}

// ------------------------------------------------------------------ chrome
function paintIcons() {
  $("#mic").innerHTML = icon("mic");
  $("#send").innerHTML = icon("send");
  $("#searchIcon").innerHTML = icon("search");
  $("#newBtn").innerHTML = icon("refresh");
  paintToggles();
}
function paintToggles() {
  const dark = document.documentElement.dataset.theme === "dark";
  $("#themeBtn").innerHTML = icon(dark ? "sun" : "moon");
  $("#ttsBtn").innerHTML = icon(state.tts ? "volume" : "volumeOff");
  $("#ttsBtn").classList.toggle("on", state.tts);
  $("#ttsBtn").setAttribute("aria-pressed", String(state.tts));
}

async function refreshHealth() {
  try {
    state.health = await api("GET", "/api/health");
  } catch { return; }
  const h = state.health;
  const dot = $("#engineDot");
  let label, cls, tip;
  if (!h.llm_model) { label = "Rules engine (offline)"; cls = ""; tip = "No LLM key configured: deterministic interpreter."; }
  else if (h.llm_status === "ready") { label = modelName(h.llm_model); cls = "ok"; tip = `LLM ready: ${h.llm_model}`; }
  else { label = `${modelName(h.llm_model)} · paused`; cls = "warn"; tip = `LLM ${h.llm_status}`; }
  dot.className = "dot " + cls;
  $("#engineLabel").textContent = label;
  $("#enginePill").title = `${tip}\nSpeech-to-text: ${h.stt_provider}`;
  renderSideFoot();
}

function renderSideFoot() {
  const h = state.health;
  const c = state.clinicians.find((x) => x.clinician_id === state.clinician);
  $("#sideFoot").innerHTML = `
    <div class="kv"><span>Clinic</span><span>${esc(c?.clinic_ids?.join(", ") || "")}</span></div>
    <div class="kv"><span>Role</span><span>${esc(c?.role || "")}</span></div>
    <div class="kv"><span>Speech-to-text</span><span>${esc(h?.stt_provider === "local" ? "Whisper (on-device)" : h?.stt_provider || "")}</span></div>
    <div class="kv"><span>Clinic time</span><span>${h ? esc(fmtDate(h.clinic_time) + " " + h.clinic_time.slice(11, 16)) : ""}</span></div>
    <div style="margin-top:4px">${icon("shield")} Synthetic data only. Drafts and bookings always need your confirmation.</div>`;
}

// ------------------------------------------------------------------ sidebar
function renderPatients() {
  const q = $("#patientSearch").value.trim().toLowerCase();
  const active = state.context?.active_patient?.patient_id;
  const list = state.patients.filter((p) => !q || p.display_name.toLowerCase().includes(q) || p.patient_id.toLowerCase().includes(q));
  $("#patientCount").textContent = state.patients.length;
  $("#patientList").innerHTML = list.map((p) => `
    <button class="patient ${p.patient_id === active ? "active" : ""}" data-patient="${esc(p.patient_id)}" title="Open ${esc(p.display_name)}'s latest report">
      ${avatar(p.display_name)}
      <span style="min-width:0">
        <div class="patient-name">${esc(p.display_name)}</div>
        <div class="patient-meta">${esc(p.patient_id)} · b. ${p.birth_year} · ${p.sex}${p.latest_study ? " · " + esc(p.latest_study.split(" ")[0]) : ""}</div>
      </span>
    </button>`).join("") || `<div class="empty-note">No matching patients.</div>`;
}

async function refreshBookings() {
  try { state.bookings = await api("GET", "/api/bookings"); } catch { state.bookings = []; }
  $("#bookingList").innerHTML = state.bookings.length
    ? state.bookings.map((b) => `
      <div class="booking-mini"><span class="when">${esc(fmtSlot(b.start).split(" · ")[0])}<br>${esc(b.start.slice(11, 16))}</span>
      <span><b>${esc(b.modality)}</b> · ${esc(b.patient.display_name)}<br><span style="color:var(--muted)">${esc(b.machine_name)} · ${esc(b.booking_id)}</span></span></div>`).join("")
    : `<div class="empty-note">No scans booked in this session.</div>`;
  if ($("#pane-schedule").classList.contains("on")) renderSchedule();
}

// ------------------------------------------------------------------ conversation
function renderHero() {
  const c = state.clinicians.find((x) => x.clinician_id === state.clinician);
  const hour = state.health ? Number(state.health.clinic_time.slice(11, 13)) : new Date().getHours();
  const greet = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";
  const caps = [
    ["file", "Retrieve reports", "Finds the right patient and study. Asks when a name is ambiguous.", ["Open Priya's latest OPG report", "Open Arjun's latest CBCT"]],
    ["filePen", "Dictate findings", "Turns speech into structured, editable findings. Nothing is invented; uncertainty is flagged.", ["Lower right third molar is horizontally impacted. Possible periapical radiolucency on 36. No caries on 47."]],
    ["quote", "Ask with evidence", "Summaries and answers cite the exact report sentences they rely on.", ["Open Priya Sharma's latest OPG", "Then: Is the wisdom tooth close to the nerve?"]],
    ["calendar", "Book scans", "Checks free machine slots and books only after you confirm.", ["Book a CBCT for Priya Sharma tomorrow morning"]],
  ];
  $("#log").innerHTML = `
    <div class="hero" id="hero">
      <h2>${greet}${c ? ", " + esc(c.display_name) : ""}</h2>
      <p>Speak or type naturally. Every action is checked by clinical safety rules before anything is shown, drafted or booked.</p>
      <div class="caps">${caps.map(([ic, t, d, tries]) => `
        <div class="cap"><div class="cap-head"><span class="cap-icon">${icon(ic)}</span>${t}</div><p>${d}</p>
        ${tries.map((x) => `<button class="try" data-send="${esc(x.replace(/^Then: /, ""))}">“${esc(x)}”</button>`).join("")}</div>`).join("")}
      </div>
    </div>`;
}

function scrollDown() { const c = $("#convo"); requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; }); }

function addUser(text, { voice = null, dictation = false } = {}) {
  $("#hero")?.remove();
  const el = document.createElement("div");
  el.className = "msg-user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div><div class="bubble-meta">${voice ? icon("mic") + " " + esc(voice) : ""}${dictation ? `<span class="badge brand">Dictation</span>` : ""}</div>`;
  $("#log").appendChild(el);
  scrollDown();
  return el;
}

function addThinking(label) {
  const el = document.createElement("div");
  el.className = "msg-bot";
  el.innerHTML = `<div class="bot-avatar">${icon("sparkles")}</div><div class="card"><div class="thinking"><span class="dots"><i></i><i></i><i></i></span><span class="label">${esc(label)}</span></div></div>`;
  $("#log").appendChild(el);
  scrollDown();
  return el;
}

function citeChips(ids) { return ids.map((id) => `<button class="cite" data-cite="${esc(id)}" title="Show in report">${esc(id)}</button>`).join(""); }
function withCites(text) { return esc(text).replace(/\[([FIR]\d+)\]/g, (_, id) => citeChips([id])); }

const STATUS = {
  ok: ["ok", "Done"], needs_clarification: ["warn", "Needs your input"], needs_confirmation: ["warn", "Awaiting confirmation"],
  not_found: ["bad", "Not found"], forbidden: ["bad", "Not permitted"], unsupported: ["info", "Out of scope"], error: ["bad", "Error"],
};
function actionOf(d) {
  if (d.booking || d.slots?.length || d.action === "book_scan") return ["calendar", "Scan booking"];
  if (d.draft || d.action === "dictate") return ["filePen", "Dictation → draft"];
  if (d.summary || d.action === "summarize_report") return ["sparkles", "Summary"];
  if (d.answer || d.action === "ask_report") return ["quote", "Report Q&A"];
  if (d.report || d.action === "open_report") return ["file", "Report retrieval"];
  if (d.candidates?.length) return ["users", "Patient identity"];
  return ["info", d.action === "cancel" ? "Cancelled" : "Assistant"];
}

function botBody(d) {
  let h = "";
  const r = d.report;
  if (d.summary && r) {
    h += `<div class="msg-text lead">Summary of <b>${esc(r.patient.display_name)}</b>'s ${esc(r.modality)} · ${esc(fmtDate(r.study_date))}</div>`;
    h += `<ul class="bullets">${d.summary.bullets.map((b) => `<li>${esc(b.text)} ${citeChips(b.sentence_ids)}</li>`).join("")}</ul>`;
    if (d.summary.dropped?.length) h += `<div class="warn-box">${d.summary.dropped.length} AI statement(s) were dropped because they were not supported by the report.</div>`;
  } else if (d.answer && r) {
    const inline = /\[[FIR]\d+\]/.test(d.answer.text);
    h += `<div class="msg-text lead">${withCites(d.answer.text)}${inline ? "" : " " + citeChips(d.answer.sentence_ids)}</div>`;
    if (!d.answer.answerable) h += `<div class="warn-box" style="background:var(--surface-3);color:var(--ink-2)">${icon("info")} Not stated in ${esc(r.patient.display_name)}'s ${esc(r.modality)} (${esc(fmtDate(r.study_date))}).</div>`;
  } else if (r && d.action !== "book_scan" && !d.slots?.length) {
    h += `<div class="msg-text lead">Opened <b>${esc(r.patient.display_name)}</b>'s <b>${esc(r.modality)}</b> report · ${esc(fmtDate(r.study_date))}</div>`;
    h += `<ul class="bullets">${r.impression.map((s) => `<li>${esc(s.text)} ${citeChips([s.id])}</li>`).join("")}</ul>`;
    h += `<div class="row" style="margin-top:10px"><button class="btn sm" data-tab-open="report">${icon("file")} View full report</button></div>`;
  } else {
    h += `<div class="msg-text">${esc(d.message)}</div>`;
  }

  if (d.candidates?.length) {
    h += `<div class="choice-grid">${d.candidates.map((c) => `
      <button class="choice" data-send="${esc(c.display_name)}">${avatar(c.display_name)}
        <span><div class="choice-title">${esc(c.display_name)} ${c.is_current_patient ? '<span class="badge brand">current</span>' : ""}</div>
        <div class="choice-sub">${esc(c.patient_id)} · b. ${c.birth_year}${c.latest_study ? " · last " + esc(c.latest_study) : ""}</div></span>
      </button>`).join("")}</div>`;
  }
  if (d.slots?.length && d.status === "needs_confirmation") {
    const s = d.slots[0];
    const who = state.context?.active_patient;
    h += `<div class="confirm-box"><div class="row" style="gap:6px;font-weight:600">${icon("shield")} Confirm before booking</div>
      <dl class="confirm-grid"><dt>Patient</dt><dd>${esc(who ? `${who.display_name} (${who.patient_id})` : "-")}</dd>
      <dt>Scan</dt><dd>${esc(s.modality)}</dd><dt>When</dt><dd>${esc(fmtSlot(s.start))}–${esc(s.end.slice(11, 16))}</dd>
      <dt>Where</dt><dd>${esc(s.machine_name)}</dd></dl>
      <div class="row"><button class="btn ok" data-send="yes">${icon("check")} Yes, book it</button><button class="btn" data-send="no">${icon("x")} No</button></div></div>`;
  } else if (d.slots?.length) {
    h += `<div class="choice-grid">${d.slots.map((s) => `
      <button class="choice" data-send="option ${s.option}"><span class="opt">${s.option}</span>
        <span><div class="choice-title">${esc(fmtSlot(s.start))}</div><div class="choice-sub">${esc(s.modality)} · ${esc(s.machine_name)}</div></span>
      </button>`).join("")}</div>`;
  }
  if (d.booking) {
    const b = d.booking;
    h += `<div class="ticket"><span class="ticket-icon">${icon("check")}</span><div><b>${esc(b.modality)} booked · ${esc(fmtSlot(b.start))}</b>
      <div style="font-size:12.5px;color:var(--ink-2)">${esc(b.patient.display_name)} · ${esc(b.machine_name)} · <span class="mono">${esc(b.booking_id)}</span></div>
      ${b.reason ? `<div style="font-size:12px;color:var(--muted);margin-top:2px">${esc(b.reason)}</div>` : ""}</div></div>`;
  }
  if (d.draft) {
    const dr = d.draft;
    const unc = dr.findings.filter((f) => f.certainty === "uncertain").length;
    h += `<div class="ai-box" style="margin-top:10px"><div class="row"><b>${icon("filePen")} Draft ${esc(dr.draft_id)}</b><span class="badge warn">Pending review</span></div>
      <ul class="bullets" style="margin-top:2px">${dr.findings.slice(0, 5).map((f) => `<li><b>${esc(f.site.fdi || f.site.name || "Site?")}</b> · ${esc(f.observation)}${f.certainty === "uncertain" ? ' <span class="badge warn">uncertain</span>' : ""}</li>`).join("")}</ul>
      <div class="row" style="margin-top:8px"><button class="btn sm primary" data-tab-open="draft">${icon("filePen")} Review draft</button>
      <span style="font-size:12px;color:var(--muted)">${dr.findings.length} finding(s) · ${unc} uncertain · ${dr.unparsed_segments.length + dr.rejected_claims.length} to check</span></div></div>`;
  }
  if (d.available_reports?.length) {
    const name = state.patients.find((p) => p.patient_id === d.context?.active_patient?.patient_id)?.display_name;
    h += `<div class="row" style="margin-top:8px">${d.available_reports.slice().reverse().map((a) => `<button class="chip" data-send="Open ${esc(name || "her")}'s ${esc(a.modality)} report from ${esc(fmtDateLong(a.study_date))}">${icon("file")} ${esc(a.modality)} · ${esc(fmtDate(a.study_date))}</button>`).join("")}</div>`;
  }
  const shownCites = Boolean(d.summary || d.answer || (r && d.action === "open_report"));
  if (d.sources?.length && !shownCites) {
    h += `<details class="evidence"><summary>${icon("quote")} Evidence · ${d.sources.length} source sentence(s)</summary>
      ${d.sources.map((s) => `<div class="ev">${citeChips([s.sentence_id])}<span><b>${esc(s.modality)} ${esc(fmtDate(s.study_date))}</b> · ${esc(s.text)}</span></div>`).join("")}</details>`;
  }
  return h;
}

function addBot(d) {
  $$("#log .choice, #log .confirm-box button").forEach((b) => { b.disabled = true; });
  const [cls, label] = STATUS[d.status] || ["", d.status];
  const [aic, alabel] = actionOf(d);
  const el = document.createElement("div");
  el.className = "msg-bot";
  el.innerHTML = `<div class="bot-avatar">${icon("sparkles")}</div><div class="card">
    <div class="card-head"><span class="badge ${cls}">${esc(label)}</span><span class="badge">${icon(aic)} ${esc(alabel)}</span>
      <span class="meta">${d.interpreter ? `<span title="${esc(d.interpreter)}">${icon("cpu")} ${esc(engineLabel(d.interpreter))}</span>` : ""}${d.latency_ms != null ? `<span>${icon("clock")} ${d.latency_ms < 1000 ? Math.round(d.latency_ms) + " ms" : (d.latency_ms / 1000).toFixed(1) + " s"}</span>` : ""}</span></div>
    ${botBody(d)}</div>`;
  $("#log").appendChild(el);
  scrollDown();
}

function addError(msg) {
  const el = document.createElement("div");
  el.className = "msg-bot";
  el.innerHTML = `<div class="bot-avatar" style="background:var(--bad)">${icon("alert")}</div><div class="card"><div class="card-head"><span class="badge bad">Error</span></div><div class="msg-text">${esc(msg)}</div></div>`;
  $("#log").appendChild(el);
  scrollDown();
}

// ------------------------------------------------------------------ sending
function setBusy(b) { state.busy = b; $("#send").disabled = b; }

async function send(text, { voice = null } = {}) {
  text = (text ?? $("#text").value).trim();
  if (!text || state.busy) return;
  $("#text").value = "";
  autosize();
  if (state.mode === "dictation") return sendDictation(text, voice);
  addUser(text, { voice });
  const th = addThinking("Understanding request…");
  setBusy(true);
  try {
    const d = await api("POST", "/api/chat", { text, session_id: state.sessionId });
    th.remove();
    handleTurn(d);
  } catch (e) {
    th.remove();
    addError(e.message);
  } finally { setBusy(false); }
}

async function sendDictation(text, voice) {
  addUser(text, { voice, dictation: true });
  const th = addThinking("Structuring dictated findings…");
  setBusy(true);
  try {
    const pid = state.context?.active_patient?.patient_id || null;
    const draft = await api("POST", "/api/dictations", { transcript: text, patient_id: pid });
    th.remove();
    afterDraft(draft);
  } catch (e) { th.remove(); addError(e.message); } finally { setBusy(false); }
}

function afterDraft(draft) {
  const n = draft.findings.length;
  addBot({
    status: "ok", action: "dictate",
    interpreter: draft.extractor.startsWith("llm") ? "llm:" + draft.extractor.split(":")[2]?.split(" ")[0]
      : draft.extractor.includes("fallback") ? "rules (fallback: llm_unavailable)" : "rules",
    message: `Draft ${draft.draft_id} created with ${n} finding(s).`, draft, context: state.context,
  });
  setDraft(draft);
  showTab("draft");
}

async function sendAudio(blob, filename) {
  const dictation = state.mode === "dictation";
  const u = addUser("Transcribing audio…", { voice: "Server speech-to-text", dictation });
  const th = addThinking("Transcribing with Whisper…");
  setBusy(true);
  const fd = new FormData();
  fd.append("audio", blob, filename);
  try {
    if (dictation) {
      const pid = state.context?.active_patient?.patient_id;
      if (pid) fd.append("patient_id", pid);
      const draft = await api("POST", "/api/dictations/audio", fd, true);
      $(".bubble", u).textContent = draft.transcript_raw;
      th.remove();
      afterDraft(draft);
    } else {
      if (state.sessionId) fd.append("session_id", state.sessionId);
      const d = await api("POST", "/api/chat/audio", fd, true);
      const stt = d.trace.find((t) => t.step === "stt");
      $(".bubble", u).textContent = stt?.detail?.transcript || d.input_text;
      $(".bubble-meta", u).innerHTML = `${icon("mic")} ${esc(stt?.detail?.provider === "local" ? "Whisper" : stt?.detail?.provider || "STT")} · ${stt?.ms ?? "?"} ms`;
      th.remove();
      handleTurn(d);
    }
  } catch (e) {
    th.remove();
    $(".bubble", u).textContent = "(audio could not be transcribed)";
    addError(e.message);
  } finally { setBusy(false); }
}

function handleTurn(d) {
  state.sessionId = d.session_id;
  state.context = d.context;
  state.lastTurn = d;
  state.turns += 1;
  addBot(d);
  if (d.report) {
    const same = state.report?.view.report_id === d.report.report_id;
    const cited = new Set(d.sources.filter((s) => s.report_id === d.report.report_id).map((s) => s.sentence_id));
    state.report = { view: d.report, cited, summary: d.summary || (same && !d.answer ? state.report.summary : null), answer: d.answer || null };
    loadReportList(d.report.patient.patient_id);
    if (!d.slots?.length && !d.booking) showTab("report");
  }
  if (!d.report && state.report && d.sources?.length) {
    const ids = d.sources.filter((s) => s.report_id === state.report.view.report_id).map((s) => s.sentence_id);
    if (ids.length) { state.report.cited = new Set(ids); renderReport(); }
  }
  if (d.draft) { setDraft(d.draft); showTab("draft"); }
  if (d.booking) { refreshBookings(); }
  if (d.slots?.length && state.schedule.modality !== d.slots[0].modality) { state.schedule.modality = d.slots[0].modality; }
  renderContext();
  renderChips();
  renderPatients();
  renderTrace();
  refreshHealth();
  if (state.tts) speak(d.message);
}

// ------------------------------------------------------------------ context + chips
function renderContext() {
  const pill = $("#contextPill");
  const c = state.context;
  pill.classList.toggle("pending", Boolean(c?.pending_question));
  if (c?.pending_question) {
    pill.innerHTML = `${icon("alert")} <span>Waiting for you: ${esc(c.pending_question)}</span>`;
    pill.title = c.pending_question;
  } else if (c?.active_patient) {
    const p = c.active_patient;
    const rep = state.report?.view.report_id === c.active_report_id ? state.report.view : null;
    pill.innerHTML = `${icon("user")} In context: <b>${esc(p.display_name)}</b> · ${esc(p.patient_id)}${rep ? ` · ${esc(rep.modality)} ${esc(fmtDate(rep.study_date))}` : ""}`;
    pill.title = "Follow-ups like “summarize her report” refer to this patient";
  } else {
    pill.innerHTML = `${icon("user")} No patient in context`;
    pill.title = "";
  }
}

function suggestions() {
  const c = state.context;
  if (c?.pending_question) return ["Cancel"];
  if (state.mode === "dictation") {
    return ["Lower right third molar is horizontally impacted.", "Possible periapical radiolucency on tooth 36. No caries on 47.", "Generalised mild horizontal bone loss. Advise CBCT before extraction."];
  }
  const p = c?.active_patient;
  if (!p) return ["Open Priya's latest OPG report", "Open Arjun's latest CBCT", "Book a CBCT for Priya Sharma tomorrow morning"];
  const pr = p.sex === "M" ? "his" : "her";
  const out = [`Summarize ${pr} report`];
  const teeth = Object.keys(state.report?.view.tooth_map || {});
  if (teeth.some((t) => t.endsWith("8"))) out.push("Is the wisdom tooth close to the nerve?");
  out.push("Does it mention bone loss?", `Book the scan recommended in ${pr} report`, `Show me ${pr} previous report`);
  return out;
}

function renderChips() {
  $("#chips").innerHTML = suggestions().map((s) => `<button class="chip" data-send="${esc(s)}">${esc(s)}</button>`).join("");
}

// ------------------------------------------------------------------ panel: report
async function loadReportList(pid) {
  try { state.reportList = await api("GET", `/api/patients/${encodeURIComponent(pid)}/reports`); } catch { state.reportList = []; }
  renderReport();
}

function renderReport() {
  const el = $("#pane-report");
  if (!state.report) {
    el.innerHTML = empty("file", "No report open", "Ask for a report by voice or text, or pick a patient on the left.");
    return;
  }
  const { view: r, cited, summary, answer } = state.report;
  const p = r.patient;
  const studies = state.reportList.slice().reverse().map((s) => `
    <button class="study ${s.report_id === r.report_id ? "on" : ""}" data-send="Open ${esc(p.display_name)}'s ${esc(s.modality)} report from ${esc(fmtDateLong(s.study_date))}">
      <b>${esc(s.modality)}</b>${esc(fmtDate(s.study_date))}</button>`).join("");
  let ai = "";
  if (summary) {
    ai = `<div class="ai-box"><div class="row"><b>${icon("sparkles")} Summary</b><span class="badge brand" title="${esc(summary.generated_by)}">${esc(summary.generated_by.startsWith("llm") ? "AI · grounded" : "Extractive")}</span></div>
      <ul class="bullets" style="margin-top:0">${summary.bullets.map((b) => `<li>${esc(b.text)} ${citeChips(b.sentence_ids)}</li>`).join("")}</ul>
      ${summary.dropped?.length ? `<div class="warn-box"><b>Dropped (not supported by the report):</b><ul>${summary.dropped.map((x) => `<li>“${esc(x.text)}”: ${esc(x.reason)}</li>`).join("")}</ul></div>` : ""}</div>`;
  } else if (answer) {
    ai = `<div class="ai-box"><div class="row"><b>${icon("quote")} ${esc(answer.question)}</b><span class="badge ${answer.answerable ? "brand" : ""}">${answer.answerable ? "Answered from report" : "Not in report"}</span></div>
      <div>${withCites(answer.text)} ${/\[[FIR]\d+\]/.test(answer.text) ? "" : citeChips(answer.sentence_ids)}</div>
      ${answer.dropped?.length ? `<div class="warn-box">AI answer rejected: ${esc(answer.dropped[0].reason)}. Showing the report's own words.</div>` : ""}</div>`;
  }
  const sec = (title, list) => `<div class="section-title">${title}</div>` + list.map((s) => `
    <div class="sent ${cited.has(s.id) ? "cited" : ""}" id="sent-${esc(s.id)}"><span class="sid">${esc(s.id)}</span><span>${esc(s.text)}</span></div>`).join("");
  el.innerHTML = `
    <div class="patient-card">${avatar(p.display_name)}<div style="flex:1;min-width:0"><h3>${esc(p.display_name)}</h3>
      <div class="patient-meta">${esc(p.patient_id)} · b. ${p.birth_year} · ${p.sex === "F" ? "Female" : p.sex === "M" ? "Male" : p.sex}</div></div>
      <span class="badge brand" style="font-size:12px">${esc(r.modality)}</span></div>
    ${studies ? `<div class="study-pills">${studies}</div>` : ""}
    <div class="report-meta">${esc(fmtDate(r.study_date))} · ${esc(r.reported_by)} · <span class="badge ${r.status === "final" ? "ok" : "warn"}">${esc(r.status)}</span><br>Indication: ${esc(r.clinical_indication)}</div>
    ${ai ? `<div style="margin-top:14px">${ai}</div>` : ""}
    <div class="section-title">Teeth referenced <span style="text-transform:none;letter-spacing:0;font-weight:500">· click a tooth to ask about it</span></div>
    <div id="reportOdo"></div>
    ${sec("Findings", r.findings)}${sec("Impression", r.impression)}${sec("Recommendations", r.recommendations)}`;
  const marks = {};
  for (const [fdi, ids] of Object.entries(r.tooth_map || {})) {
    const texts = ids.map((id) => [...r.findings, ...r.impression, ...r.recommendations].find((s) => s.id === id)?.text).filter(Boolean);
    marks[fdi] = { state: "mentioned", cited: ids.some((id) => cited.has(id)), tip: texts.join(" | ") };
  }
  renderOdontogram($("#reportOdo"), {
    marks,
    onClick: (fdi) => send(`What does the report say about tooth ${fdi}?`),
    caption: "FDI numbering",
  });
}

function flashSentence(id) {
  showTab("report");
  const s = $(`#sent-${CSS.escape(id)}`);
  if (!s) return;
  s.scrollIntoView({ behavior: "smooth", block: "center" });
  s.classList.remove("flash");
  void s.offsetWidth;
  s.classList.add("flash", "cited");
}

// ------------------------------------------------------------------ panel: draft
function setDraft(d) {
  state.draft = d;
  state.draftEdit = JSON.parse(JSON.stringify({ findings: d.findings, recommendations: d.recommendations, modality: d.modality }));
  state.draftJson = false;
  $("#draftCount").hidden = d.status !== "pending_clinician_review";
  renderDraft();
}

function draftMarks(findings) {
  const marks = {};
  const rank = { present: 3, uncertain: 4, missing: 2, absent: 1, mentioned: 0 };
  for (const f of findings) {
    const codes = f.site.kind === "tooth" && f.site.fdi ? [f.site.fdi] : f.site.candidates || [];
    for (const code of codes) {
      let st = f.site.kind !== "tooth" ? "mentioned"
        : f.condition === "missing_tooth" && f.presence === "present" ? "missing"
        : f.presence === "absent" || f.condition === "normal_finding" ? "absent"
        : f.certainty === "uncertain" ? "uncertain" : "present";
      const prev = marks[code];
      if (!prev || rank[st] > rank[prev.state]) marks[code] = { state: st, tip: f.observation };
      else prev.tip += " | " + f.observation;
    }
  }
  return marks;
}

function renderDraft() {
  const el = $("#pane-draft");
  const d = state.draft;
  if (!d) {
    el.innerHTML = empty("filePen", "No draft yet", "Dictate findings, e.g. “Lower right third molar is horizontally impacted”. Switch the composer to Dictation mode for longer reports.");
    return;
  }
  const e = state.draftEdit;
  const locked = d.status === "approved";
  const dis = locked ? "disabled" : "";
  const findings = e.findings.map((f, i) => {
    const cls = f.presence === "absent" || f.condition === "normal_finding" ? "absent" : f.certainty === "uncertain" ? "uncertain" : "present";
    const siteInput = f.site.kind === "tooth" || f.site.kind === "unspecified" || f.site.kind === "tooth_unresolved"
      ? `<input class="fdi" data-f="${i}" data-k="fdi" value="${esc(f.site.fdi || "")}" placeholder="${esc((f.site.candidates || []).join("/") || "FDI")}" title="FDI tooth number" ${dis}>`
      : `<input data-f="${i}" data-k="fdi" value="" placeholder="FDI" title="Region site; enter a tooth number to change" ${dis}>`;
    return `<div class="finding ${cls}" id="finding-${i}">
      <div class="f-grid">${siteInput}
        <input data-f="${i}" data-k="observation" value="${esc(f.observation)}" ${dis} aria-label="Observation">
        ${locked ? "" : `<button class="btn ghost sm" data-act="del" data-f="${i}" title="Remove finding">${icon("trash")}</button>`}
      </div>
      <div class="site-name">${esc(f.site.kind === "tooth" ? `${f.site.name || ""}${f.site.universal ? " · Universal #" + f.site.universal : ""}` : f.site.kind === "region" ? f.site.name : f.site.kind === "tooth_unresolved" ? `Unresolved: “${f.site.name}”, which tooth?` : "No site dictated")}</div>
      <div class="f-controls">
        <select data-f="${i}" data-k="presence" ${dis}><option value="present" ${f.presence === "present" ? "selected" : ""}>Present</option><option value="absent" ${f.presence === "absent" ? "selected" : ""}>Absent</option></select>
        <select data-f="${i}" data-k="certainty" ${dis}><option value="definite" ${f.certainty === "definite" ? "selected" : ""}>Definite</option><option value="uncertain" ${f.certainty === "uncertain" ? "selected" : ""}>Uncertain</option></select>
        <span class="badge">${esc(f.condition_label)}</span><span class="badge">${esc(f.origin)}</span>
        ${f.review_flags.map((x) => `<span class="flag">${esc(x)}</span>`).join("")}
      </div>
      ${f.evidence ? `<div class="evidence-q">“${esc(f.evidence)}”</div>` : ""}
    </div>`;
  }).join("");
  const recs = e.recommendations.map((r, i) => `<input class="rec-input" data-r="${i}" value="${esc(r.text)}" ${dis} style="margin-bottom:6px">`).join("");
  el.innerHTML = `
    <div class="draft-head"><div><h3>Draft report</h3>
      <div class="patient-meta">${esc(d.draft_id)} · v${d.version} · ${esc(d.patient ? `${d.patient.display_name} (${d.patient.patient_id})` : "No patient linked")} · ${esc(d.modality || "modality not stated")}</div></div>
      <span class="badge ${locked ? "ok" : "warn"}">${locked ? "Approved" : "Pending clinician review"}</span></div>
    ${locked
      ? `<div class="ok-box" style="margin-top:10px">${icon("checkCircle")} Signed by ${esc(state.clinicians.find((c) => c.clinician_id === d.approved_by)?.display_name || d.approved_by)} · ${esc(new Date(d.approved_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }))}</div>`
      : `<ul class="reasons">${d.review_reasons.map((x) => `<li>${icon("alert")}<span>${esc(x)}</span></li>`).join("")}</ul>`}
    <div class="section-title">Chart</div><div id="draftOdo"></div>
    <div class="section-title">Structured findings <span style="text-transform:none;letter-spacing:0;font-weight:500">· extractor: ${esc(d.extractor)}</span>
      ${state.draftJson ? "" : ""}<span style="margin-left:auto"></span><button class="btn ghost sm" data-act="json">${icon("code")} ${state.draftJson ? "Form" : "JSON"}</button></div>
    ${state.draftJson
      ? `<textarea class="json-edit" id="draftJson" ${dis}>${esc(JSON.stringify(e, null, 2))}</textarea>`
      : `${findings || '<div class="empty-note">No findings.</div>'}${locked ? "" : `<button class="btn sm" data-act="add">${icon("plus")} Add finding</button>`}
         <div class="section-title">Recommendations (as dictated)</div>${recs || '<div class="empty-note">None dictated.</div>'}`}
    ${d.rejected_claims.length ? `<div class="bad-box"><b>${icon("x")} AI suggestions rejected as unsupported by the dictation</b><ul>${d.rejected_claims.map((c) => `<li><b>${esc(c.claim)}</b>: ${esc(c.reason)}</li>`).join("")}</ul></div>` : ""}
    ${d.unparsed_segments.length ? `<div class="warn-box"><b>Not structured, please review</b><ul>${d.unparsed_segments.map((u) => `<li>“${esc(u)}”</li>`).join("")}</ul></div>` : ""}
    <div class="section-title">Readable draft</div><div class="paper">${esc(d.draft_text)}</div>
    ${locked ? "" : `<div class="sticky-actions"><button class="btn" data-act="save">${icon("save")} Save changes</button>
      <button class="btn primary" data-act="approve">${icon("shield")} Approve &amp; sign</button><span class="patient-meta" id="draftMsg"></span></div>`}`;
  renderOdontogram($("#draftOdo"), {
    marks: draftMarks(e.findings),
    onClick: (fdi) => {
      const i = e.findings.findIndex((f) => f.site.fdi === fdi);
      const card = i >= 0 && $(`#finding-${i}`);
      if (card) { card.scrollIntoView({ behavior: "smooth", block: "center" }); card.animate([{ boxShadow: "0 0 0 3px var(--brand)" }, { boxShadow: "none" }], { duration: 1200 }); }
    },
  });
}

function onDraftInput(t) {
  const e = state.draftEdit;
  if (t.dataset.r !== undefined) { e.recommendations[+t.dataset.r].text = t.value; return; }
  if (t.dataset.f === undefined) return;
  const f = e.findings[+t.dataset.f];
  const k = t.dataset.k;
  if (k === "observation") f.observation = t.value;
  else if (k === "presence" || k === "certainty") { f[k] = t.value; renderDraft(); }
  else if (k === "fdi") {
    const v = t.value.trim();
    if (!v) f.site = { kind: "unspecified", fdi: null, universal: null, name: null, candidates: [], source: "clinician" };
    else f.site = { kind: "tooth", fdi: v, universal: null, name: null, candidates: [], source: "clinician" };
  }
}

async function draftAction(act) {
  const d = state.draft;
  const msg = (m) => { const el = $("#draftMsg"); if (el) el.textContent = m; };
  if (act === "json") {
    if (state.draftJson) {
      try { state.draftEdit = JSON.parse($("#draftJson").value); } catch (err) { toast("Invalid JSON: " + err.message, true); return; }
    }
    state.draftJson = !state.draftJson;
    renderDraft();
    return;
  }
  if (act === "add") {
    state.draftEdit.findings.push({
      finding_id: "", condition: "other", condition_label: "Clinician-added", observation: "", presence: "present", certainty: "definite",
      hedge_terms: [], site: { kind: "unspecified", fdi: null, universal: null, name: null, candidates: [], source: "clinician" },
      attributes: [], evidence: "", evidence_span: null, review_flags: [], origin: "clinician",
    });
    renderDraft();
    return;
  }
  if (act === "save") {
    if (state.draftJson) { try { state.draftEdit = JSON.parse($("#draftJson").value); } catch (err) { toast("Invalid JSON: " + err.message, true); return; } }
    try {
      const body = { ...state.draftEdit, expected_version: d.version };
      const nd = await api("PUT", `/api/dictations/${d.draft_id}`, body);
      setDraft(nd);
      toast(`Saved · version ${nd.version}`);
    } catch (err) { msg(err.message); toast(err.message, true); }
    return;
  }
  if (act === "approve") {
    try {
      const nd = await api("POST", `/api/dictations/${d.draft_id}/approve`);
      setDraft(nd);
      toast("Draft approved and signed");
    } catch (err) { msg(err.message); toast(err.message, true); }
  }
}

// ------------------------------------------------------------------ panel: schedule
async function renderSchedule() {
  const el = $("#pane-schedule");
  const now = state.health ? parseDay(state.health.clinic_time) : new Date();
  const days = Array.from({ length: 7 }, (_, i) => { const d = new Date(now); d.setDate(d.getDate() + i); return d; });
  const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  const afterHours = state.health && Number(state.health.clinic_time.slice(11, 13)) >= 17;
  if (!state.schedule.day || !days.some((d) => iso(d) === state.schedule.day)) {
    state.schedule.day = iso(days.find((d, i) => d.getDay() !== 0 && !(i === 0 && afterHours)));
  }
  const s = state.schedule;
  let slots = [];
  let err = "";
  try { slots = await api("GET", `/api/slots?modality=${s.modality}&day=${s.day}`); } catch (e) { err = e.message; }
  const who = state.context?.active_patient;
  el.innerHTML = `
    <div class="section-title">Scan type</div>
    <div class="seg" id="modSeg">${["OPG", "CBCT", "IOPA", "BITEWING", "CEPH"].map((m) => `<button data-mod="${m}" class="${m === s.modality ? "on" : ""}">${m}</button>`).join("")}</div>
    <div class="section-title">Day</div>
    <div class="day-strip">${days.map((d) => `<button class="day ${iso(d) === s.day ? "on" : ""}" data-day="${iso(d)}" ${d.getDay() === 0 ? "disabled title='Closed'" : ""}>${DAYS[d.getDay()]}<b>${d.getDate()}</b>${MONTHS[d.getMonth()]}</button>`).join("")}</div>
    <div class="section-title">Free slots <span style="text-transform:none;letter-spacing:0;font-weight:500">· ${slots.length} available</span></div>
    ${err ? `<div class="bad-box">${esc(err)}</div>` : slots.length ? `<div class="slot-grid">${slots.map((x) => `<button class="slot" data-slot="${esc(x.start)}" title="${esc(x.machine_name)}">${esc(x.start.slice(11, 16))}<small>${esc(x.machine_id)}</small></button>`).join("")}</div>` : `<div class="empty-note">No free ${esc(s.modality)} slots on this day.</div>`}
    <div class="warn-box" style="background:var(--surface-3);color:var(--ink-2)">${icon("shield")} ${who ? `Clicking a slot asks to book it for <b>${esc(who.display_name)}</b>.` : "Open a patient first to book from here."} Nothing is booked until you confirm in the conversation.</div>
    <div class="section-title">Booked in this session</div>
    ${state.bookings.length ? state.bookings.map((b) => `<div class="booking-mini"><span class="when">${esc(fmtSlot(b.start))}</span><span><b>${esc(b.modality)}</b> · ${esc(b.patient.display_name)} · <span class="mono">${esc(b.booking_id)}</span></span></div>`).join("") : '<div class="empty-note">None yet.</div>'}`;
}

// ------------------------------------------------------------------ panel: trace
const STEP_ICON = {
  stt: "mic", interpret: "cpu", resolve_patient: "users", select_candidate: "users", select_report: "file",
  llm_summary: "sparkles", llm_answer: "sparkles", llm_extract: "sparkles", draft: "filePen", booking: "calendar",
  check_slots: "calendar", select_slot: "calendar", confirm_booking: "shield", book: "check",
};
function stepSummary(t) {
  const d = t.detail || {};
  switch (t.step) {
    case "stt": return `${d.provider} (${d.model || ""}) → “${d.transcript}”`;
    case "interpret": {
      if (d.error) return `LLM error: ${d.error} → falling back to rules`;
      const i = d.intent || {};
      const u = d.usage;
      return `${engineLabel(d.engine?.startsWith("llm") ? d.engine : d.engine || "")} → ${i.action}${i.target?.patient_name ? ` · patient “${i.target.patient_name}”` : ""}${i.target?.modality ? ` · ${i.target.modality}` : ""}${u ? ` · ${u.input_tokens} in / ${u.output_tokens} out tokens${u.est_cost_usd ? ` · $${u.est_cost_usd}` : ""}` : ""}`;
    }
    case "resolve_patient": return `${d.status}${d.mention ? ` for “${d.mention}”` : ""}${d.candidates?.length ? ` → ${d.candidates.join(", ")}` : ""}${d.method ? ` (${d.method})` : ""}`;
    case "select_report": return d.rule ? `${d.rule} → ${d.report_id}` : `${d.modality} · ${d.which}${d.date ? " " + d.date : ""} → ${d.chosen || "none found"} (${(d.candidates || []).length} candidates)`;
    case "llm_summary": case "llm_answer": return d.error ? `LLM error: ${d.error} → ${d.fallback}` : `kept ${d.kept ?? ""} · ${d.rejected ? "rejected: " + d.rejected : d.grounding || ""}`;
    case "llm_extract": return d.error ? `LLM error: ${d.error} → rules` : `${d.proposed_findings} proposed → ${d.accepted} accepted, ${d.rejected?.length || 0} rejected`;
    case "draft": return `${d.draft_id} · ${d.findings} finding(s) · ${d.extractor}`;
    case "check_slots": return `${(d.found || []).length} free ${d.modality} slot(s) · ${d.requested}`;
    case "booking": return d.stage === "modality_from_report" ? `scan type from report: ${d.modality} (${d.evidence})` : d.stage;
    case "select_slot": return `“${d.answer}” → ${d.selected || "no match"}`;
    case "confirm_booking": return `explicit yes: ${d.explicit_yes ? "YES" : "no"} · “${d.utterance}”`;
    case "book": return d.error ? `failed: ${d.error}` : `${d.booking_id} · ${d.slot_id}`;
    case "select_candidate": return `“${d.answer}” → ${d.selected || "no match"}`;
    default: return "";
  }
}
function renderTrace() {
  const el = $("#pane-trace");
  const d = state.lastTurn;
  if (!d) { el.innerHTML = empty("activity", "No turns yet", "Every request shows how it was interpreted, validated and executed here."); return; }
  const usage = d.trace.map((t) => t.detail?.usage).filter(Boolean);
  const cost = usage.reduce((a, u) => a + (u.est_cost_usd || 0), 0);
  el.innerHTML = `
    <div class="trace-sum">
      <div class="stat"><b>${d.latency_ms < 1000 ? Math.round(d.latency_ms) + " ms" : (d.latency_ms / 1000).toFixed(2) + " s"}</b><span>Turn latency</span></div>
      <div class="stat"><b>${usage.length}</b><span>LLM calls</span></div>
      <div class="stat"><b>${cost ? "$" + cost.toFixed(4) : "$0"}</b><span>Est. LLM cost</span></div>
    </div>
    <div class="patient-meta" style="margin-bottom:12px">Input: “${esc(d.input_text)}”${d.normalized_text !== d.input_text ? `<br>Normalised: “${esc(d.normalized_text)}”` : ""}</div>
    <div class="timeline">${d.trace.map((t) => `
      <div class="step ${t.detail?.error ? "err" : ""}"><span class="step-dot">${icon(STEP_ICON[t.step] || "layers")}</span>
        <div class="step-title">${esc(t.step.replace(/_/g, " "))}${t.ms != null ? `<span class="ms">${t.ms} ms</span>` : ""}</div>
        <div class="step-sum">${esc(stepSummary(t))}</div>
        <details><summary>details</summary><pre>${esc(JSON.stringify(t.detail, null, 2))}</pre></details></div>`).join("")}
      <div class="step"><span class="step-dot">${icon("check")}</span><div class="step-title">response · ${esc(d.status)}</div><div class="step-sum">${esc(d.message)}</div></div>
    </div>`;
}

// ------------------------------------------------------------------ tabs
function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === name));
  $$(".pane").forEach((p) => p.classList.toggle("on", p.id === "pane-" + name));
  if (name === "schedule") renderSchedule();
}

// ------------------------------------------------------------------ voice
let voice;
function setupVoice() {
  voice = new Voice({
    canvas: $("#wave"),
    onInterim: (t) => { $("#text").value = t; autosize(); },
    onFinal: (t) => send(t, { voice: "Browser speech" }),
    onAudio: (blob) => sendAudio(blob, "recording.webm"),
    onState: (on) => {
      $("#composer").classList.toggle("recording", on);
      $("#mic").classList.toggle("rec", on);
      $("#mic").innerHTML = icon(on ? "stop" : "mic");
      $("#text").placeholder = on ? (state.stt === "browser" ? "Listening… speak now" : "Recording… click again or release Space to send") : placeholder();
    },
    onError: (m) => toast(m, true),
  });
  if (!voice.browserSupported) {
    state.stt = "server";
    $$("#sttSeg button").forEach((b) => b.classList.toggle("on", b.dataset.stt === "server"));
    $("#sttSeg [data-stt=browser]").disabled = true;
    $("#sttSeg [data-stt=browser]").title = "Browser speech recognition is not available in this browser";
  }
}
function toggleMic() {
  if (voice.active) voice.stop();
  else voice.start(state.stt, { continuous: state.mode === "dictation" });
}
function placeholder() {
  return state.mode === "dictation"
    ? `Dictate findings${state.context?.active_patient ? " for " + state.context.active_patient.display_name : ""}… e.g. “Lower right third molar is horizontally impacted”`
    : "Ask, dictate or book… e.g. “Open Priya's latest OPG report”";
}

async function loadSamples() {
  try {
    const m = await (await fetch("/samples/manifest.json")).json();
    $("#sample").insertAdjacentHTML("beforeend", m.map((s) => `<option value="${esc(s.file)}">${esc(s.transcript)}</option>`).join(""));
  } catch { $("#sample").hidden = true; }
}

// ------------------------------------------------------------------ misc UI
function autosize() { const t = $("#text"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 180) + "px"; }

async function switchClinician(id) {
  state.clinician = id;
  store.set("viscan-clinician", id);
  const c = state.clinicians.find((x) => x.clinician_id === id);
  $("#userAvatar").outerHTML = `<span class="avatar" id="userAvatar" style="background:hsl(${hue(c.display_name)} 42% 42%)">${esc(initials(c.display_name))}</span>`;
  resetConversation();
  state.patients = await api("GET", "/api/patients");
  renderPatients();
  refreshBookings();
  renderSideFoot();
  renderHero();
}

function resetConversation() {
  state.sessionId = null; state.context = null; state.lastTurn = null; state.report = null; state.reportList = [];
  state.draft = null; state.draftEdit = null;
  $("#draftCount").hidden = true;
  renderHero(); renderContext(); renderChips(); renderReport(); renderDraft(); renderTrace(); renderPatients();
}

function bind() {
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-send],[data-cite],[data-tab-open],[data-patient],[data-act],[data-slot],[data-day],[data-mod]");
    if (!t) return;
    if (t.dataset.send !== undefined) {
      const v = t.dataset.send;
      if (v === "Cancel") return send("cancel");
      return send(v);
    }
    if (t.dataset.cite) return flashSentence(t.dataset.cite);
    if (t.dataset.tabOpen) return showTab(t.dataset.tabOpen);
    if (t.dataset.patient) {
      const p = state.patients.find((x) => x.patient_id === t.dataset.patient);
      if (state.clinicians.find((c) => c.clinician_id === state.clinician)?.role === "receptionist") {
        return send(`Book a scan for ${p.display_name}`);
      }
      return send(`Open ${p.display_name}'s latest report`);
    }
    if (t.dataset.act) return draftAction(t.dataset.act);
    if (t.dataset.day) { state.schedule.day = t.dataset.day; return renderSchedule(); }
    if (t.dataset.mod) { state.schedule.modality = t.dataset.mod; return renderSchedule(); }
    if (t.dataset.slot) {
      const who = state.context?.active_patient;
      if (!who) return toast("Open or name a patient first. Bookings always go through confirmation.", true);
      const iso = t.dataset.slot;
      const d = parseDay(iso);
      return send(`Book a ${state.schedule.modality} for ${who.display_name} on ${d.getDate()} ${MONTHS_LONG[d.getMonth()]} at ${iso.slice(11, 16)}`);
    }
  });
  $("#pane-draft").addEventListener("input", (e) => onDraftInput(e.target));
  $("#pane-draft").addEventListener("change", (e) => { if (e.target.dataset.k === "fdi") onDraftInput(e.target); });
  $$(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $("#send").addEventListener("click", () => send());
  $("#text").addEventListener("input", autosize);
  $("#text").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  $("#mic").addEventListener("click", toggleMic);
  $("#patientSearch").addEventListener("input", renderPatients);
  $("#newBtn").addEventListener("click", () => { resetConversation(); toast("New conversation started"); });
  $("#clinician").addEventListener("change", (e) => switchClinician(e.target.value));
  $("#themeBtn").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("viscan-theme", next);
    paintToggles();
  });
  $("#ttsBtn").addEventListener("click", () => { state.tts = !state.tts; store.set("viscan-tts", state.tts ? "1" : "0"); paintToggles(); if (!state.tts) speechSynthesis?.cancel(); });
  $$("#modeSeg button").forEach((b) => b.addEventListener("click", () => {
    state.mode = b.dataset.mode;
    $$("#modeSeg button").forEach((x) => x.classList.toggle("on", x === b));
    $("#composer").classList.toggle("dictation", state.mode === "dictation");
    $("#text").placeholder = placeholder();
    renderChips();
  }));
  $$("#sttSeg button").forEach((b) => b.addEventListener("click", () => {
    if (b.disabled) return;
    state.stt = b.dataset.stt;
    $$("#sttSeg button").forEach((x) => x.classList.toggle("on", x === b));
  }));
  $("#sample").addEventListener("change", async (e) => {
    const f = e.target.value;
    e.target.value = "";
    if (!f) return;
    const blob = await (await fetch("/samples/" + f)).blob();
    new Audio(URL.createObjectURL(blob)).play().catch(() => {});
    sendAudio(blob, f);
  });
  // Hold Space to talk (when not typing).
  const typing = (el) => el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && !typing(document.activeElement) && !voice.active) { e.preventDefault(); voice.start(state.stt, { continuous: true }); }
  });
  document.addEventListener("keyup", (e) => { if (e.code === "Space" && voice.active && !typing(document.activeElement)) { e.preventDefault(); voice.stop(); } });
}

// ------------------------------------------------------------------ boot
async function init() {
  paintIcons();
  bind();
  setupVoice();
  await refreshHealth();
  state.clinicians = await api("GET", "/api/clinicians");
  if (!state.clinicians.some((c) => c.clinician_id === state.clinician)) state.clinician = state.clinicians[0].clinician_id;
  $("#clinician").innerHTML = state.clinicians.map((c) => `<option value="${esc(c.clinician_id)}" ${c.clinician_id === state.clinician ? "selected" : ""}>${esc(c.display_name)} · ${esc(c.role)}</option>`).join("");
  await switchClinician(state.clinician);
  loadSamples();
  setInterval(refreshHealth, 20000);
}

init().catch((e) => toast("Failed to start: " + e.message, true));
