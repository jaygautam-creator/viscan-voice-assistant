# Submission note

**Repository contents:** runnable FastAPI service, web UI, synthetic dataset, 82 automated tests, an eval harness, sample voice clips, the [architecture page](ARCHITECTURE.md) and a [documented API demo with real outputs](DEMO.md).

## What works

- **Report retrieval (A).**
  - Typed or spoken requests are interpreted into intent, patient, modality and which report.
  - Patient resolution is deterministic. Ambiguous names (Priya Sharma, Priya Shah; the surname Sharma) trigger a clarification question. The answer can be a surname, "the second one", a birth year or an ID, and the original request then resumes.
  - Likely STT misspellings ("Pria") require confirmation.
  - Missing reports return `not_found` and list what *is* available. Unknown and other-clinic patients are refused without leaking that they exist.
- **Dictation to draft (B).**
  - The output is structured, editable JSON per finding: condition, observation, FDI and Universal tooth, presence, certainty, modifiers, the verbatim evidence span and review flags. A readable draft is rendered from it.
  - Hedges are labelled uncertain. Negations become pertinent negatives. Under-specified teeth stay unresolved and block approval. Unmappable speech is surfaced, never dropped.
  - The LLM's proposed findings are validated against the transcript. Invented diagnoses, quotes, teeth and modifiers are rejected and listed for the clinician. Dropped hedges are restored.
  - The draft has a versioned edit and approve workflow.
- **Follow-up (C).** "Summarize her report" resolves to the report in context. If there is no context, or the pronoun contradicts the active patient, the assistant asks. Every answer returns the source sentences (report ID, section, sentence ID, text), and the UI highlights them.
- **Bonus: report-aware Q&A.** Answers to questions about the selected report are grounded and cite sentence IDs. "Not mentioned" is stated explicitly, with the caveat that absence from a report is not a clinical negative.
- **Bonus: scan-booking agent.** "Book a CBCT for Priya Sharma tomorrow morning" offers real free slots from a synthetic machine diary, then asks for confirmation. It books only on an explicit yes in the clinician's own words, re-checks the slot at commit, and offers alternatives if the slot was taken. "Book the scan recommended in her report" takes the scan type from the report and cites the sentence. Receptionists can book but cannot read reports. Both bonus options are implemented.
- **Clinical web UI.** Patient sidebar, voice-first conversation with rich cards (patient and slot choice, booking confirmation, booking ticket), an interactive FDI tooth chart, a structured draft editor with versioned save and approve, a slot-grid schedule, a pipeline-trace timeline, and light/dark themes. Screenshots are in the README.
- **Voice.** Browser speech recognition, server-side faster-whisper running locally (verified on the bundled clips at about 0.5 to 0.8 s each), an OpenAI STT adapter, and a deterministic mock for CI. All share one normalisation step.
- **Engineering.**
  - LLM interpretation (function calling and structured outputs, via Gemini 3.8 Flash or Claude behind one interface) is separated from business-rule validation.
  - LLM failures fall back to rules, and a circuit breaker stops calling a rate-limited provider.
  - Input validation, clinic- and role-based authorisation, session binding, an IDs-only audit log, and per-call token and cost traces.

## What is incomplete

- **The LLM is only partly measured.** Gemini works live: it accepted the router tools and the extraction schema, returned the correct tool call for "Open Priya's latest OPG report", and produced schema-valid findings with the hedge marked uncertain. A routing call costs about $0.0017 on 3.8 Flash. The full live eval could not complete because the free-tier key allows 20 Gemini 3.8 Flash requests a day. A first run looked like 27/27, but most turns had quietly fallen back to rules. I changed the eval runner to report how many turns the LLM actually served, and added a circuit breaker. With a billing-enabled key, `scripts/run_evals.py --llm` gives the LLM-only score, latency and cost. Claude is covered by a mock-server contract test only.
- **No demo video.** The demo is a documented sequence of real API calls ([DEMO.md](DEMO.md)), which the brief accepts. The web UI is ready for a 3 to 5 minute screen recording.
- **The offline rules interpreter is tuned to the eval set** I wrote, so its 27/27 is optimistic. Unusual phrasings degrade to `unsupported` or a clarification rather than to a wrong patient.
- **Auth is mocked** (`X-Clinician-Id`). Sessions and drafts live in memory.

## What I would improve with more time

1. **Measure the LLM properly.** Run the eval with a billing-enabled key, grow it from real clinician phrasings, and compare Gemini 3.8 Flash at `minimal` and `low` thinking, Flash-Lite and Claude on accuracy, p95 latency and cost per turn.
2. **Stronger identity for voice.** Phonetic and transliteration-aware name matching (Indian names in particular), birth-year or ID confirmation before opening a report from a fuzzy match, and optional speaker verification.
3. **Grounding beyond lexical checks.** Add an entailment or LLM-judge pass for summaries and Q&A, and accept clinically equivalent paraphrases in dictation ("lying on its side" as horizontal) as *suggestions* for the clinician to confirm.
4. **Real-time voice UX.** Streaming STT with partial transcripts, barge-in, and server-side TTS of the `message` field.
5. **Productionise.** OIDC auth, Redis sessions, Postgres for drafts with an immutable audit trail, FHIR `DiagnosticReport` export, PHI-safe structured logging and rate limiting.
6. **Learn from edits.** Turn clinician edits to drafts into new eval cases and vocabulary entries.
7. **Booking depth.** Cancellation and rescheduling, patient notifications, and a real scheduling backend in place of the synthetic diary.
