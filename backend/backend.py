import os
import json
import time
import subprocess
import tempfile
import shutil
import re
import requests
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ollama_api(path, method="GET", json_body=None, stream=False, timeout=300):
    """Call the Ollama HTTP API on the host machine."""
    url = f"{OLLAMA_BASE}{path}"
    if method == "POST":
        resp = requests.post(url, json=json_body, stream=stream, timeout=timeout)
    elif method == "DELETE":
        resp = requests.delete(url, json=json_body, timeout=timeout)
    else:
        resp = requests.get(url, stream=stream, timeout=timeout)
    resp.raise_for_status()
    return resp


def list_local_models():
    """Return set of model names currently pulled in Ollama."""
    resp = ollama_api("/api/tags")
    data = resp.json()
    return {m["name"] for m in data.get("models", [])}


def pull_model(model_name):
    """Pull a model via Ollama API. Returns True if it was freshly downloaded."""
    already = model_name in list_local_models()
    resp = ollama_api("/api/pull", method="POST",
                      json_body={"name": model_name}, stream=True, timeout=1800)
    for line in resp.iter_lines():
        pass  # consume stream until done
    return not already


def delete_model(model_name):
    ollama_api("/api/delete", method="DELETE", json_body={"name": model_name})


def get_running_model_ram(model_name):
    """Get actual VRAM/RAM used by the model while it is loaded."""
    try:
        resp = ollama_api("/api/ps")
        data = resp.json()
        for m in data.get("models", []):
            if m.get("name") == model_name or m.get("model") == model_name:
                size_bytes = m.get("size", 0)
                return size_bytes
    except Exception:
        pass
    return None


def derive_schema_from_value(value):
    """Derive a JSON schema from an example JSON value."""
    if isinstance(value, dict):
        props = {}
        for k, v in value.items():
            props[k] = derive_schema_from_value(v)
        return {
            "type": "object",
            "properties": props,
            "required": list(value.keys()),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        if value:
            return {"type": "array", "items": derive_schema_from_value(value[0])}
        return {"type": "array"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    return {"type": "string"}


def json_values_equal(a, b):
    """Compare two parsed JSON values. Strings are compared case-insensitively."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(json_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(json_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def benchmark_model(model_name, questions):
    """
    Run all Q&A pairs against the model using Ollama structured outputs.
    Each question's expected answer is a JSON object; a schema is derived
    and passed via the ``format`` parameter so Ollama forces the model
    to output conforming JSON.  The JSON is then compared exactly.
    """
    # Warm-up so the model is loaded
    ollama_api("/api/generate", method="POST",
               json_body={"model": model_name, "prompt": "Hi",
                          "stream": False, "format": "json"},
               timeout=600)

    ram_bytes = get_running_model_ram(model_name)

    results = []
    total_eval_rate = 0.0
    eval_count = 0

    for qa in questions:
        q = qa["question"]
        try:
            expected_obj = json.loads(qa["answer"]) if isinstance(qa["answer"], str) else qa["answer"]
        except (json.JSONDecodeError, TypeError):
            expected_obj = {"result": qa["answer"]}

        schema = derive_schema_from_value(expected_obj)

        resp = ollama_api("/api/generate", method="POST",
                          json_body={
                              "model": model_name,
                              "prompt": q + "\n\nRespond with JSON only.",
                              "stream": False,
                              "format": schema,
                              "options": {"num_predict": 1024},
                          }, timeout=600)
        data = resp.json()

        model_answer_raw = data.get("response", "").strip()
        eval_duration = data.get("eval_duration", 0)
        eval_tokens = data.get("eval_count", 0)

        if eval_duration > 0:
            rate = eval_tokens / (eval_duration / 1e9)
            total_eval_rate += rate
            eval_count += 1
        else:
            rate = 0.0

        try:
            model_obj = json.loads(model_answer_raw)
            correct = json_values_equal(expected_obj, model_obj)
        except (json.JSONDecodeError, TypeError):
            correct = False

        results.append({
            "question": q,
            "expected": json.dumps(expected_obj, ensure_ascii=False),
            "model_answer": model_answer_raw,
            "correct": correct,
            "eval_rate": round(rate, 2),
        })

    avg_eval_rate = round(total_eval_rate / eval_count, 2) if eval_count else 0.0
    correct_count = sum(1 for r in results if r["correct"])
    pct_correct = round(100 * correct_count / len(results), 1) if results else 0.0

    return {
        "eval_rate": avg_eval_rate,
        "ram_bytes": ram_bytes,
        "results": results,
        "percent_correct": pct_correct,
    }


# ---------------------------------------------------------------------------
# LaTeX report generation
# ---------------------------------------------------------------------------

def escape_latex(text):
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)
    special = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    # Handle backslash first
    text = text.replace('\\', r'\textbackslash{}')
    for char, replacement in special.items():
        text = text.replace(char, replacement)
    return text


def bytes_to_human(b):
    if b is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def generate_report_pdf(benchmarks, questions):
    """
    Generate a concise PhD-calibre LaTeX PDF report.
    benchmarks: list of dicts {model, eval_rate, ram_bytes, percent_correct, results}
    questions: list of {question, answer}
    Returns path to generated PDF.
    """
    tmpdir = tempfile.mkdtemp()
    n_questions = len(questions)
    n_models = len(benchmarks)

    # Build summary table rows
    summary_rows = ""
    for b in benchmarks:
        ram_str = escape_latex(bytes_to_human(b["ram_bytes"]))
        model_name = escape_latex(b["model"])
        correct_n = sum(1 for r in b.get("results", []) if r.get("correct"))
        summary_rows += (
            f"        {model_name} & {ram_str} & "
            f"{b['eval_rate']:.2f} & "
            f"{correct_n}/{n_questions} & "
            f"{b['percent_correct']:.1f}\\% \\\\\n"
        )

    # Build per-model accuracy breakdown (one line per model, no raw Q&A)
    accuracy_rows = ""
    for b in benchmarks:
        model_name = escape_latex(b["model"])
        correct_n = sum(1 for r in b.get("results", []) if r.get("correct"))
        wrong_n = len(b.get("results", [])) - correct_n
        accuracy_rows += (
            f"        {model_name} & {correct_n} & {wrong_n} & "
            f"{b['percent_correct']:.1f}\\% \\\\\n"
        )

    latex_src = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{caption}

\definecolor{headerblue}{RGB}{40,60,100}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{headerblue}{\textsc{Model Lab}}}
\fancyhead[R]{\textcolor{headerblue}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}

\title{\textcolor{headerblue}{\textbf{Model Lab --- LLM Benchmark Report}}}
\author{Generated by Model Lab}
\date{\today}

\begin{document}
\maketitle
\thispagestyle{fancy}

\begin{abstract}
This report presents a comparative evaluation of """ + str(n_models) + r""" large language models
served via Ollama, tested against """ + str(n_questions) + r""" structured-output benchmark questions.
Each model was assessed on runtime memory consumption, inference speed,
and accuracy. The results facilitate informed model selection for deployment
under given resource and accuracy constraints.
\end{abstract}

\section{Methodology}

Models were loaded sequentially via the Ollama REST API. For each model:
\begin{enumerate}
    \item The model was pulled (downloaded) if not already present.
    \item A warm-up inference was performed to ensure the model was fully loaded.
    \item Runtime RAM usage was recorded via the \texttt{/api/ps} endpoint.
    \item Each question was sent with a JSON schema constraint (structured output).
          The model's JSON response was compared against the expected answer using
          exact value matching.
    \item Eval rate (tokens/s) was recorded from Ollama response metadata.
    \item If the model was freshly downloaded, it was deleted afterwards.
\end{enumerate}

\noindent A total of \textbf{""" + str(n_questions) + r"""} questions were used, each requiring a
deterministic JSON answer.

\section{Results}

\begin{table}[h!]
\centering
\caption{Comparative summary of evaluated models.}
\begin{tabular}{@{} l r r r r @{}}
    \toprule
    \textbf{Model} & \textbf{RAM} & \textbf{Eval Rate (tok/s)} & \textbf{Correct} & \textbf{Accuracy} \\
    \midrule
""" + summary_rows + r"""    \bottomrule
\end{tabular}
\end{table}

\begin{table}[h!]
\centering
\caption{Accuracy breakdown per model.}
\begin{tabular}{@{} l r r r @{}}
    \toprule
    \textbf{Model} & \textbf{Correct} & \textbf{Incorrect} & \textbf{Accuracy} \\
    \midrule
""" + accuracy_rows + r"""    \bottomrule
\end{tabular}
\end{table}

\section{Conclusion}

The benchmark results above enable direct comparison of model resource requirements
and performance characteristics. Researchers and practitioners should weigh the
trade-offs between model size, inference speed, and accuracy when selecting a model
for their specific use case.

\end{document}
"""

    tex_path = os.path.join(tmpdir, "report.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_src)

    # Compile LaTeX to PDF (run twice for TOC)
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
            capture_output=True, timeout=120
        )

    pdf_path = os.path.join(tmpdir, "report.pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"PDF generation failed. LaTeX log:\n{proc.stdout.decode(errors='replace')}")

    return pdf_path


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/models", methods=["GET"])
def api_list_models():
    """List models currently available in Ollama."""
    try:
        models = list(list_local_models())
        models.sort()
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


def build_generate_prompt(use_case):
    """Build the prompt used to generate benchmark Q&A pairs."""
    return (
        "You are an expert AI evaluator. A user wants to benchmark language models "
        "for the following use case:\n\n"
        f"\"{use_case}\"\n\n"
        "Generate exactly 10 benchmark test cases. Each test case is a JSON object with:\n"
        "- \"question\": A clear question or task prompt for the model. "
        "The question MUST instruct the model to output JSON.\n"
        "- \"answer\": The single exact correct answer as a JSON object with descriptive keys.\n\n"
        "Requirements:\n"
        "- Every answer must be a JSON object (not a raw string or array at the top level).\n"
        "- Each question must have exactly ONE correct, deterministic answer.\n"
        "- Questions should be specific and unambiguous.\n"
        "- Each question should end with an instruction like "
        "'Respond as JSON with the schema: {…}'.\n\n"
        "Output ONLY a valid JSON array of these objects. No markdown, no explanation.\n\n"
        "Example:\n"
        '[{"question": "What is the chemical symbol for water? '
        'Respond as JSON with the schema: {\\\"symbol\\\": \\\"<string>\\\"}.", '
        '"answer": {"symbol": "H2O"}}]'
    )


@app.route("/api/generate-questions", methods=["POST"])
def api_generate_questions():
    """Use gemma3:4b to generate structured benchmark questions."""
    data = request.get_json()
    use_case = data.get("use_case", "")
    if not use_case:
        return jsonify({"error": "use_case is required"}), 400

    prompt = build_generate_prompt(use_case)

    try:
        resp = ollama_api("/api/generate", method="POST",
                          json_body={
                              "model": "gemma3:4b",
                              "prompt": prompt,
                              "stream": False,
                              "options": {"temperature": 0.7, "num_predict": 4096}
                          }, timeout=300)
        raw = resp.json().get("response", "")

        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
        else:
            questions = json.loads(raw)

        validated = []
        for q in questions:
            if isinstance(q, dict) and "question" in q and "answer" in q:
                ans = q["answer"]
                if isinstance(ans, str):
                    try:
                        ans = json.loads(ans)
                    except json.JSONDecodeError:
                        ans = {"result": ans}
                if not isinstance(ans, dict):
                    ans = {"result": ans}
                validated.append({
                    "question": str(q["question"]),
                    "answer": json.dumps(ans, ensure_ascii=False),
                })

        return jsonify({"questions": validated})
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned invalid JSON. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/build-prompt", methods=["POST"])
def api_build_prompt():
    """Return the prompt text so the user can copy it to their preferred LLM."""
    data = request.get_json()
    use_case = data.get("use_case", "")
    if not use_case:
        return jsonify({"error": "use_case is required"}), 400
    return jsonify({"prompt": build_generate_prompt(use_case)})


@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    """
    Run the full benchmark pipeline.
    Expects JSON: { models: ["model1", ...], questions: [{question, answer}, ...] }
    """
    data = request.get_json()
    models = data.get("models", [])
    questions = data.get("questions", [])

    if not models:
        return jsonify({"error": "At least one model is required"}), 400
    if not questions:
        return jsonify({"error": "At least one question-answer pair is required"}), 400

    existing_models = list_local_models()
    all_results = []

    for model_name in models:
        was_downloaded = False
        try:
            # Pull if needed
            if model_name not in existing_models:
                pull_model(model_name)
                was_downloaded = True
            else:
                # Still call pull to ensure latest, but mark not downloaded
                was_downloaded = pull_model(model_name)

            # Benchmark
            result = benchmark_model(model_name, questions)
            result["model"] = model_name

            all_results.append(result)

        except Exception as e:
            all_results.append({
                "model": model_name,
                "error": str(e),
                "eval_rate": 0,
                "ram_bytes": None,
                "percent_correct": 0,
                "results": [],
            })
        finally:
            # Delete if we downloaded it for this benchmark
            if was_downloaded:
                try:
                    delete_model(model_name)
                except Exception:
                    pass

    # Generate PDF report
    try:
        pdf_path = generate_report_pdf(all_results, questions)
        # Store the path for download
        app.config["LAST_REPORT_PATH"] = pdf_path
    except Exception as e:
        return jsonify({
            "benchmarks": all_results,
            "report_error": str(e),
        })

    return jsonify({
        "benchmarks": all_results,
        "report_ready": True,
    })


@app.route("/api/report", methods=["GET"])
def api_download_report():
    """Download the last generated PDF report."""
    pdf_path = app.config.get("LAST_REPORT_PATH")
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({"error": "No report available. Run a benchmark first."}), 404
    return send_file(pdf_path, mimetype="application/pdf",
                     as_attachment=True, download_name="model_lab_report.pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5192, debug=False)
