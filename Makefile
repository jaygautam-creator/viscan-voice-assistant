.PHONY: install install-all run run-voice test eval eval-llm demo lint format audio

install:        ## Core + dev dependencies (offline rules mode works out of the box)
	uv sync

install-all:    ## Also local Whisper STT and the OpenAI STT adapter
	uv sync --all-extras

run:            ## API + web UI on http://127.0.0.1:8000
	uv run uvicorn app.main:app --port 8000 --reload --reload-dir app --reload-dir static

run-voice:      ## Same, with on-device Whisper speech-to-text
	VISCAN_STT_PROVIDER=local uv run uvicorn app.main:app --port 8000 --reload --reload-dir app --reload-dir static

test:           ## Unit + API tests (no network, no keys)
	uv run pytest

eval:           ## Conversation eval set against the rules engine
	VISCAN_AUDIT_LOG= uv run python scripts/run_evals.py

eval-llm:       ## Same eval set against the configured LLM (needs GEMINI_API_KEY or ANTHROPIC_API_KEY)
	VISCAN_AUDIT_LOG= uv run python scripts/run_evals.py --llm

demo:           ## Regenerate docs/DEMO.md against a running server (make run first)
	uv run python scripts/demo.py

lint:           ## Static checks
	uv run ruff check app tests scripts
	uv run ruff format --check app tests scripts

format:         ## Auto-format
	uv run ruff format app tests scripts
	uv run ruff check --fix app tests scripts

audio:          ## Regenerate sample voice clips (macOS `say`)
	uv run python scripts/make_audio_samples.py
