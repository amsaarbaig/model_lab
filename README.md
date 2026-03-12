<p align="center">
  <img src="https://img.shields.io/badge/Model%20Lab-LLM%20Benchmarking-2c3e6b?style=for-the-badge" alt="Model Lab">
</p>

<h1 align="center">Model Lab</h1>

<p align="center">
  <em>A lightweight tool for benchmarking local LLMs served by Ollama — compare models on memory usage, inference speed, and structured-output accuracy.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/flask-3.x-green?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/ollama-local-black?style=flat-square" alt="Ollama">
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License">
</p>

---

## What It Does

Model Lab lets you **quantitatively benchmark Ollama models** side by side. You define questions with exact expected JSON answers, pick models to test, and get a clean comparison of:

| Metric | How It's Measured |
|---|---|
| **RAM Usage** | Actual runtime memory from Ollama's `/api/ps` (not download size) |
| **Eval Rate** | Tokens per second from Ollama response metadata |
| **Accuracy** | Exact JSON match using structured outputs (schema-constrained) |

Results are displayed in the browser and can be exported as a **LaTeX-generated PDF report**.

---

## Features

- **Structured Output Testing** — Questions enforce a JSON schema via Ollama's `format` parameter, so answers are deterministic and programmatically verifiable
- **AI Question Generation** — Describe your use case and let `gemma3:4b` generate benchmark questions, or copy the prompt to use with your preferred LLM
- **Auto Pull & Cleanup** — Models not already downloaded are pulled automatically and deleted after benchmarking
- **PDF Report** — One-click download of a concise LaTeX-typeset comparison report
- **Clean UI** — Light, minimal, academic-style interface
- **Uses Your Existing Ollama** — Connects to the Ollama server already running on your host machine (no extra Ollama container)

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [Ollama](https://ollama.com/) running on your host machine

### Run

```bash
git clone <repo-url> model_lab
cd model_lab
docker compose up --build
```

Open **http://localhost:5192** in your browser.

### Usage

1. **Define questions** — Enter question/answer pairs (answers as JSON), or generate them via AI
2. **Pick models** — Enter Ollama model tags, one per line (e.g. `gemma3:1b`, `phi4-mini:3.8b`)
3. **Run benchmark** — Hit start, wait for results
4. **Download report** — Export a PDF summary

---

## Project Structure

```
model_lab/
├── docker-compose.yml      # Single service, connects to host Ollama
├── Dockerfile              # Python 3.12 + TeXLive for PDF generation
├── requirements.txt        # Flask, flask-cors, requests
├── backend/
│   └── backend.py          # Flask API (benchmark, AI generation, PDF)
└── frontend/
    ├── index.html          # Single-page UI
    └── styles.css          # Light academic theme
```

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API base URL |

---

## ⚠️ Disclaimer

> **This project is a work in progress.** It is provided as-is, with no guarantees of correctness, completeness, or fitness for any particular purpose. Use it at your own risk. Features may change, break, or be removed without notice. The authors accept no liability for any issues arising from the use of this software.

---

<p align="center">
  <sub>Built with Flask, Ollama, and LaTeX</sub>
</p>
