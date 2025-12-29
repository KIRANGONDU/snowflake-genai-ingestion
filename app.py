
# app_flask.py
import os
# If python-dotenv causes issues in your environment, skip auto .env load:
os.environ["FLASK_SKIP_DOTENV"] = "true"

from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename

from snowflake1 import run_load
from pipelines import run_classic_pipeline, run_genai_pipeline  # adjust if your module is named differently

# ==== NEW: background jobs ====
import uuid
import time
from concurrent.futures import ThreadPoolExecutor

# ==== NEW: data & LLM ====
import pandas as pd
import json
import csv

# ---------- Config ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")  # root folder for uploads
INPUT_DIR = os.path.join(UPLOAD_ROOT, "input")   # where uploaded CSVs go
OUTPUT_DIR = os.path.join(UPLOAD_ROOT, "output") # pipeline output folder
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

# ==== NEW: background job infra ====
EXECUTOR = ThreadPoolExecutor(max_workers=3)
JOBS = {}  # job_id -> {"status": "pending|running|done|error", "message": str, "created": ts}

def start_job(fn, *args, **kwargs):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending", "message": "Queued...", "created": time.time()}
    def wrapper():
        JOBS[job_id]["status"] = "running"
        try:
            msg = fn(*args, **kwargs)
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["message"] = msg
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["message"] = f" Error: {e}"
    EXECUTOR.submit(wrapper)
    return job_id

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Template (PROFESSIONAL UI) ----------

TEMPLATE = """
<!doctype html>
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Customer Data Standardizer</title>

  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

  <!-- Bootstrap 5 CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <!-- Bootstrap Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">

  <style>
    :root {
      --panel-h: 58vh;
      --card-radius: 14px;
      --shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
      --muted: #6b7280;
    }

    html, body { height: 100%; }

    body {
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif;
      background-color: #f7f9fc;
      color: #111827;
      padding-block: 16px;
    }

    .container-fluid { max-width: 1200px; }

    /* Header */
    .app-header { padding: 8px 0; }
    .app-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .app-subtitle { color: var(--muted); font-size: 0.9rem; margin-left: 4px; }

    /* Cards */
    .card {
      border: 1px solid rgba(15, 23, 42, 0.06);
      border-radius: var(--card-radius);
      box-shadow: var(--shadow);
      overflow: hidden;
      background: var(--bs-body-bg);
    }
    .card-header {
      border: none;
      background: var(--bs-body-bg);
      padding: 12px 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .card-header .title { font-weight: 600; font-size: 1rem; }

    .panel-body { max-height: var(--panel-h); overflow: auto; padding: 16px; }
    .panel-footer {
      padding: 10px 16px;
      border-top: 1px solid rgba(15, 23, 42, 0.08);
      background: var(--bs-body-bg);
    }

    /* Sections */
    .section-title {
      font-weight: 600;
      font-size: 0.95rem;
      margin: 10px 0 8px;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    }

    /* Form controls */
    .form-label, .form-check-label, .input-group-text, .form-control, .form-select, small { font-size: 0.92rem; }
    .input-group-text { background: transparent; }
    .form-control, .form-select { border-radius: 10px; }

    /* Tabs */
    .nav-pills .nav-link { padding: 6px 12px; border-radius: 999px; }

    /* Chips */
    .chip {
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.85rem;
      transition: all .15s ease;
    }
    .chip:hover { transform: translateY(-1px); }

    /* Buttons */
    .btn { border-radius: 999px; padding: 8px 14px; font-size: 0.92rem; }

    /* Status block */
    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 0.9rem;
      max-height: 28vh;
      overflow: auto;
      border-radius: 12px;
      background: var(--bs-body-bg);
      border: 1px solid rgba(15, 23, 42, 0.08);
    }

    /* Footer info text */
    .foot-note { font-size: 0.85rem; color: var(--muted); }

    /* Smooth theme transitions */
    * { transition: background-color .2s ease, color .2s ease, border-color .2s ease; }
  </style>
</head>
<body>
  <div class="container-fluid px-2">
    <!-- Header -->
    <div class="app-header d-flex align-items-center justify-content-between mb-3">
      <div class="d-flex flex-column">
        <div class="app-title">
          <i class="bi bi-sliders2 text-primary"></i>
          Customer Data Standardizer
        </div>
        <div class="app-subtitle">Upload, run pipeline, connect Snowflake & ask questions — all in one screen.</div>
      </div>
      <button id="themeToggle" class="btn btn-outline-secondary">
        <i class="bi bi-moon-stars me-1"></i> Theme
      </button>
    </div>

    <!-- Single form covers both panels -->
    <form method="post" class="needs-validation" novalidate enctype="multipart/form-data">
      <div class="row g-3">
        <!-- LEFT: Console (Upload + Pipeline + Snowflake in tabs) -->
        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-header">
              <i class="bi bi-terminal"></i>
              <span class="title">Console</span>
            </div>

            <div class="panel-body">
              <!-- Tabs -->
              <ul class="nav nav-pills" id="consoleTabs" role="tablist">
                <li class="nav-item" role="presentation">
                  <button class="nav-link active" id="tab-upload-pipeline" data-bs-toggle="tab" data-bs-target="#pane-upload-pipeline" type="button" role="tab">
                    <i class="bi bi-cloud-arrow-up me-1"></i> Upload & Pipeline
                  </button>
                </li>
                <li class="nav-item" role="presentation">
                  <button class="nav-link" id="tab-snowflake" data-bs-toggle="tab" data-bs-target="#pane-snowflake" type="button" role="tab">
                    <i class="bi bi-snow me-1"></i> Snowflake
                  </button>
                </li>
              </ul>

              <div class="tab-content mt-3">
                <!-- Upload & Pipeline -->
                <div class="tab-pane fade show active" id="pane-upload-pipeline" role="tabpanel" aria-labelledby="tab-upload-pipeline">
                  <div class="section-title">Upload CSV Files</div>
                  <div class="mb-3">
                    <label for="csv-files" class="form-label fw-semibold">Select CSV file(s)</label>
                    <div class="input-group">
                      <span class="input-group-text"><i class="bi bi-filetype-csv"></i></span>
                      <input type="file" class="form-control" id="csv-files" name="files" accept=".csv" multiple required>
                      <div class="invalid-feedback">Please choose at least one CSV file.</div>
                    </div>
                    <small class="text-muted">Multiple files supported. Only .csv are allowed.</small>
                  </div>

                  <div class="section-title">Pipeline Configuration</div>
                  <div class="row g-3">
                    <div class="col-12 col-md-6">
                      <label for="input-folder" class="form-label fw-semibold">Input Folder</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-folder2"></i></span>
                        <input type="text" class="form-control" id="input-folder" name="input" value="{{input}}" required>
                        <div class="invalid-feedback">Provide a valid input folder.</div>
                      </div>
                      <small class="text-muted">Uploads are saved here automatically.</small>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="output-folder" class="form-label fw-semibold">Output Folder</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-folder-check"></i></span>
                        <input type="text" class="form-control" id="output-folder" name="output" value="{{output}}" required>
                        <div class="invalid-feedback">Provide a valid output folder.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="outfile" class="form-label fw-semibold">Output File</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-file-earmark-text"></i></span>
                        <input type="text" class="form-control" id="outfile" name="outfile" value="{{outfile}}" required>
                        <div class="invalid-feedback">Provide an output file name.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="mode" class="form-label fw-semibold">Mode</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-cpu"></i></span>
                        <select class="form-select" id="mode" name="mode" required>
                          <option value="classic" {% if mode=='classic' %}selected{% endif %}>Classic</option>
                          <option value="genai" {% if mode=='genai' %}selected{% endif %}>GenAI</option>
                        </select>
                        <div class="invalid-feedback">Select a mode.</div>
                      </div>
                    </div>

                    <div class="col-12 d-flex align-items-center gap-2">
                      <div class="form-check form-switch mb-0">
                        <input class="form-check-input" type="checkbox" id="use-llm" name="use_llm" {% if use_llm %}checked{% endif %}>
                        <label class="form-check-label" for="use-llm"><i class="bi bi-robot me-1 text-primary"></i> Use LLM (GenAI)</label>
                      </div>
                      <small class="text-muted">Enable GenAI for the pipeline when needed.</small>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="provider" class="form-label fw-semibold">Provider</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-cloud"></i></span>
                        <input type="text" class="form-control" id="provider" name="provider" value="{{provider}}" placeholder="Gemini / OpenAI / Claude" required>
                        <div class="invalid-feedback">Specify a provider.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="llm-model" class="form-label fw-semibold">LLM Model</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-cpu-fill"></i></span>
                        <input type="text" class="form-control" id="llm-model" name="llm_model" value="{{llm_model}}" placeholder="gemini-2.5-flash" required>
                        <div class="invalid-feedback">Provide a model name.</div>
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Snowflake -->
                <div class="tab-pane fade" id="pane-snowflake" role="tabpanel" aria-labelledby="tab-snowflake">
                  <div class="section-title">Snowflake Connection</div>
                  <div class="row g-3">
                    <div class="col-12 col-md-6">
                      <label for="sf-account" class="form-label fw-semibold">Account</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-building"></i></span>
                        <input type="text" class="form-control" id="sf-account" name="sf_account" value="{{sf_account}}" required>
                        <div class="invalid-feedback">Provide your Snowflake account.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="sf-user" class="form-label fw-semibold">User</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-person"></i></span>
                        <input type="text" class="form-control" id="sf-user" name="sf_user" value="{{sf_user}}" required>
                        <div class="invalid-feedback">Provide your Snowflake username.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="sf-password" class="form-label fw-semibold">Password</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-shield-lock"></i></span>
                        <input type="password" class="form-control" id="sf-password" name="sf_password" value="" placeholder="••••••••" required>
                        <div class="invalid-feedback">Password is required.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="sf-warehouse" class="form-label fw-semibold">Warehouse</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-hdd-network"></i></span>
                        <input type="text" class="form-control" id="sf-warehouse" name="sf_warehouse" value="{{sf_warehouse}}" required>
                        <div class="invalid-feedback">Provide a warehouse.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="sf-database" class="form-label fw-semibold">Database</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-collection"></i></span>
                        <input type="text" class="form-control" id="sf-database" name="sf_database" value="{{sf_database}}" required>
                        <div class="invalid-feedback">Provide a database.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="sf-schema" class="form-label fw-semibold">Schema</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-grid-1x2"></i></span>
                        <input type="text" class="form-control" id="sf-schema" name="sf_schema" value="{{sf_schema}}" required>
                        <div class="invalid-feedback">Provide a schema.</div>
                      </div>
                    </div>

                    <div class="col-12 col-md-6">
                      <label for="sf-table" class="form-label fw-semibold">Table Name</label>
                      <div class="input-group">
                        <span class="input-group-text"><i class="bi bi-table"></i></span>
                        <input type="text" class="form-control" id="sf-table" name="sf_table" value="{{sf_table}}" required>
                        <div class="invalid-feedback">Provide a table name.</div>
                      </div>
                    </div>

                    <div class="col-12 d-flex flex-column gap-1">
                      <div class="form-check mb-0">
                        <input class="form-check-input" type="checkbox" id="sf-create" name="sf_create" {% if sf_create %}checked{% endif %}>
                        <label class="form-check-label" for="sf-create">
                          <i class="bi bi-plus-square-dotted me-1 text-primary"></i> Create table if not exists
                        </label>
                      </div>
                      <div class="form-check mb-0">
                        <input class="form-check-input" type="checkbox" id="sf-load" name="sf_load" {% if sf_load %}checked{% endif %}>
                        <label class="form-check-label" for="sf-load">
                          <i class="bi bi-cloud-upload me-1 text-success"></i> Load to Snowflake after pipeline
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              </div> <!-- /tab-content -->
            </div>

            <div class="panel-footer d-flex justify-content-end gap-2">
              <button type="submit" name="action" value="run" class="btn btn-primary">
                <i class="bi bi-play-circle-fill me-1"></i> Run
              </button>
              <button type="submit" name="action" value="test" class="btn btn-success">
                <i class="bi bi-check2-circle me-1"></i> Test SF
              </button>
              <!-- NEW: concurrent run + ask -->
              <button type="submit" name="action" value="run_both" class="btn btn-outline-primary">
                <i class="bi bi-lightning-charge-fill me-1"></i> Run & Ask
              </button>
            </div>
          </div>
        </div>

        <!-- RIGHT: GenAI Bot -->
        <div class="col-lg-6">
          <div class="card h-100">
            <div class="card-header">
              <i class="bi bi-chat-text"></i>
              <span class="title">GenAI Bot</span>
            </div>

            <div class="panel-body">
              <div class="row g-3">
                <div class="col-12">
                  <label for="question" class="form-label fw-semibold">Your Question</label>
                  <textarea class="form-control" id="question" name="question" rows="3" placeholder="Top 5 cities by customers; Hyderabad count; monthly trend; MoM growth" required></textarea>
                  <div class="invalid-feedback">Please enter a question.</div>
                </div>

                <!-- Suggested chips -->
                <div class="col-12">
                  <div class="d-flex flex-wrap gap-1">
                    <button type="button" class="btn btn-sm btn-outline-secondary chip" data-q="How many customers are from Hyderabad?">Hyderabad count</button>
                    <button type="button" class="btn btn-sm btn-outline-secondary chip" data-q="Top 5 cities by number of customers.">Top 5 cities</button>
                    <button type="button" class="btn btn-sm btn-outline-secondary chip" data-q="Monthly trend of customers from load_date; include MoM growth.">Monthly trend + MoM</button>
                    <button type="button" class="btn btn-sm btn-outline-secondary chip" data-q="Count customers by email domain; show percent share.">Email domains %</button>
                    <button type="button" class="btn btn-sm btn-outline-secondary chip" data-q="Average age by city; top 10 cities.">Avg age by city</button>
                  </div>
                </div>

                <div class="col-12">
                  <div class="alert alert-info py-2 mb-2">
                    <i class="bi bi-lightbulb me-1"></i>
                    <strong>Tip:</strong> Ask analytics like <em>“Female customers in India by city (top 5)”</em> or <em>“Monthly customers with MoM growth”</em>. The bot turns your question into a safe plan and executes it on the latest output CSV.
                  </div>
                </div>

                <div class="col-12 d-flex justify-content-end">
                  <button type="submit" name="action" value="ask" class="btn btn-warning">
                    <i class="bi bi-question-circle-fill me-1"></i> Ask
                  </button>
                </div>
              </div>
            </div>

            <div class="panel-footer d-flex justify-content-between align-items-center">
              <small class="foot-note">Provider: {{provider}} &nbsp;|&nbsp; Model: {{llm_model}}</small>
              <small class="foot-note">Uses latest standardized file: <code>{{outfile}}</code></small>
            </div>
          </div>
        </div>
      </div>

      <!-- Status / Logs -->
      {% if message %}
      <div class="row mt-3">
        <div class="col-12">
          <div class="alert alert-secondary py-2 mb-2 d-flex align-items-center">
            <i class="bi bi-info-circle me-2"></i>
            <span class="fw-semibold">Status / Logs</span>
          </div>
          <pre class="p-3" id="statusBlock">{{message}}</pre>
        </div>
      </div>
      {% endif %}
    </form>
  </div>

  <!-- Bootstrap JS -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

  <!-- Dark Mode Toggle -->
  <script>
    (function() {
      const root = document.documentElement;
      const toggle = document.getElementById('themeToggle');
      const key = 'bs-theme';
      const current = localStorage.getItem(key) || 'light';
      root.setAttribute('data-bs-theme', current);
      toggle.addEventListener('click', () => {
        const next = root.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-bs-theme', next);
        localStorage.setItem(key, next);
      });
    })();
  </script>

  <!-- Client-side Validation -->
  <script>
    (() => {
      'use strict';
      const forms = document.querySelectorAll('.needs-validation');
      Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
          if (!form.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
          }
          form.classList.add('was-validated');
        }, false);
      });
    })();
  </script>

  <!-- Chips -> fill question -->
  <script>
    (function(){
      const ta = document.getElementById('question');
      document.querySelectorAll('.chip').forEach(btn => {
        btn.addEventListener('click', () => {
          ta.value = btn.dataset.q;
          ta.focus();
        });
      });
    })();
  </script>

  <!-- Poll job status: supports two jobs (pipeline + Q&A) -->
  <script>
    (function(){
      const statusBlock = document.getElementById('statusBlock');
      const id1 = "{{ job_id|default('') }}";
      const id2 = "{{ job_id2|default('') }}";
      const jobIds = [id1, id2].filter(Boolean);
      if (jobIds.length === 0 || !statusBlock) return;

      const state = {};
      let tries = 0;

      const labelFor = (id) => {
        if (id === id1 && id2) return "Pipeline";
        if (id === id2) return "Q&A";
        return "Job";
      };

      const render = () => {
        const parts = jobIds.map(id => {
          const s = state[id] || {status: 'pending', message: 'Queued...'};
          const short = id.substring(0, 8);
          return `(${labelFor(id)} • ${short})\\n${s.message}`;
        });
        statusBlock.textContent = parts.join("\\n\\n------------------------------\\n\\n");
      };

      const pollOnce = () => {
        const fetches = jobIds.map(id =>
          fetch(`/job/${id}`)
            .then(r => r.json())
            .then(data => { state[id] = data; })
            .catch(() => { /* ignore transient errors */ })
        );

        Promise.all(fetches).then(() => {
          render();
          const allDone = jobIds.every(id => {
            const s = state[id]?.status;
            return s === 'done' || s === 'error';
          });
          if (allDone) return;
          tries++;
          setTimeout(pollOnce, Math.min(3000, 1000 + 500*tries));
        });
      };

      pollOnce();
    })();
  </script>
</body>
</html>
"""

app = Flask(__name__)

def _snowflake_ctx_from_env():
    """Prefill UI with env vars; user can override in the form. (No saving)"""
    return {
        "sf_account": os.getenv("SNOWFLAKE_ACCOUNT", ""),
        "sf_user": os.getenv("SNOWFLAKE_USER", ""),
        "sf_warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", ""),
        "sf_database": os.getenv("SNOWFLAKE_DATABASE", ""),
        "sf_schema": os.getenv("SNOWFLAKE_SCHEMA", ""),
        "sf_table": "CUSTOMER_MASTER1",
    }

# ==== helpers for Q&A ====
def _latest_output_csv(output_dir: str, outfile: str):
    """Return path to standardized output CSV if present; else None."""
    path = os.path.join(output_dir, outfile)
    return path if os.path.exists(path) else None

def _load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=True,
        na_values=["", "NULL", "null"],
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        escapechar='\\'
    )

def _schema_summary(df: pd.DataFrame, max_uniques=8):
    """Compact schema summary + sample values for prompt."""
    cols = []
    for c in df.columns:
        ser = df[c]
        uniques = [u for u in ser.dropna().unique()[:max_uniques]]
        cols.append({
            "name": c,
            "non_null": int(ser.notna().sum()),
            "sample_values": uniques
        })
    return {"columns": cols, "rows": len(df)}

def _parse_json_strict(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise

def _call_llm_json(provider: str, model: str, system_prompt: str, user_prompt: str) -> dict:
    if provider.lower() != "gemini":
        raise RuntimeError("Only 'gemini' provider is supported in this Q&A path.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is missing. Please set it.")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = {"response_mime_type": "application/json", "temperature": 0.0}
    model = genai.GenerativeModel(model, system_instruction=system_prompt)
    resp = model.generate_content(user_prompt, generation_config=generation_config)
    text = getattr(resp, "text", None)
    if not text:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            raise ValueError("Gemini response had no text payload.")
    return _parse_json_strict(text)

SYSTEM_QA = (
  "You are a data querying planner. Return ONLY JSON describing a SAFE plan "
  "to answer the question on a tabular dataset. Allowed operations:\n"
  "- filters: eq, neq, contains, gt, gte, lt, lte\n"
  "- groupby\n"
  "- aggregates: count, distinct_count, sum, avg, min, max\n"
  "- derived columns: age_from_dob -> 'age', email_domain_from_email -> 'email_domain', month_from_date -> 'load_month'\n"
  "- percent_of_total: compute share per group for a metric\n"
  "- trend: time-series by a date/month column with aggregation\n"
  "- growth: compare consecutive periods (MoM) for a metric and return growth_rate\n"
  "- topn: pick top N rows by a metric\n"
  "- order: sort by a column asc|desc\n"
  "- select: choose columns for preview\n"
  "- chart: optional {'type': 'bar|line|pie', 'x': 'column', 'y': 'column'}\n"
  "Do NOT return Python. Output must be pure JSON keys:\n"
  "{intent, filters, derived, groupby, aggregates, percent, trend, growth, order, topn, select, limit, chart}\n"
  "Constraints: Only use existing dataset columns or derived columns listed above. "
  "If the question requests age, domain, or monthly analysis, include derived entries accordingly."
)

def USER_QA(question: str, schema_summary: dict) -> str:
    return (
        f"Question: {question}\n"
        f"Dataset summary (JSON):\n{json.dumps(schema_summary, ensure_ascii=False)}\n\n"
        "Example output:\n"
        '{'
        '"intent":"aggregation",'
        '"filters":[{"column":"country","op":"eq","value":"India"}],'
        '"derived":[{"name":"age","expr":"age_from_dob","source":"dob"},{"name":"email_domain","expr":"email_domain_from_email","source":"email"}],'
        '"groupby":["city"],'
        '"aggregates":[{"op":"count","column":"customer_id","alias":"customers"},{"op":"distinct_count","column":"email_domain","alias":"unique_domains"}],'
        '"percent":{"of":"customers","alias":"pct_customers"},'
        '"order":[{"column":"customers","dir":"desc"}],'
        '"topn":{"column":"customers","n":5},'
        '"limit":10,'
        '"select":null,'
        '"chart":{"type":"bar","x":"city","y":"customers"}'
        '}\n'
        "Return ONLY JSON for your plan."
    )

from datetime import datetime

def _safe_to_date(s):
    try:
        return pd.to_datetime(str(s), errors="coerce").date()
    except Exception:
        return None

def _age_from_dob(dob):
    d = _safe_to_date(dob)
    if not d: return None
    today = datetime.today().date()
    years = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    return years if years >= 0 and years < 120 else None

def _email_domain(email):
    if pd.isna(email): return None
    s = str(email).strip()
    if "@" not in s: return None
    return s.split("@")[-1].lower()

def _month_from_date(dt):
    d = _safe_to_date(dt)
    if not d: return None
    return f"{d.year:04d}-{d.month:02d}"

def _prepare_derived_columns(data: pd.DataFrame, derived: list):
    """Compute derived columns safely based on plan instructions."""
    if not derived: return data
    df = data.copy()
    for dspec in derived:
        name = dspec.get("name")
        expr = dspec.get("expr")
        src  = dspec.get("source")
        if not name or not expr:
            continue
        if expr == "age_from_dob":
            source_col = src or "dob"
            if source_col in df.columns:
                df[name] = df[source_col].apply(_age_from_dob)
        elif expr == "email_domain_from_email":
            source_col = src or "email"
            if source_col in df.columns:
                df[name] = df[source_col].apply(_email_domain)
        elif expr == "month_from_date":
            source_col = src or "load_date"
            if source_col in df.columns:
                df[name] = df[source_col].apply(_month_from_date)
    return df

from typing import Tuple

def _execute_plan(plan: dict, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """Execute restricted plan with pandas, return (answer_text, preview_df)."""
    data = df.copy()

    intent    = plan.get("intent", "aggregation")
    filters   = plan.get("filters", []) or []
    derived   = plan.get("derived", []) or []
    groupby   = plan.get("groupby", []) or []
    aggregates= plan.get("aggregates", []) or []
    order     = plan.get("order", []) or []
    topn      = plan.get("topn", None)
    select    = plan.get("select", None)
    limit     = plan.get("limit", 20)
    percent   = plan.get("percent", None)
    trend     = plan.get("trend", None)
    growth    = plan.get("growth", None)
    chart     = plan.get("chart", None)  # not rendered, but accepted

    # 1) Derived columns
    data = _prepare_derived_columns(data, derived)

    # 2) Filters
    for f in filters:
        col = f.get("column"); op = f.get("op"); val = f.get("value")
        if not col or col not in data.columns:
            continue
        series = data[col].astype(str)
        if op == "eq":       data = data[series == str(val)]
        elif op == "neq":    data = data[series != str(val)]
        elif op == "contains": data = data[series.str.contains(str(val), case=False, na=False)]
        elif op in {"gt","gte","lt","lte"}:
            try:
                num = pd.to_numeric(series, errors="coerce")
                comp = float(val)
                if op == "gt":   data = data[num >  comp]
                elif op == "gte":data = data[num >= comp]
                elif op == "lt": data = data[num <  comp]
                elif op == "lte":data = data[num <= comp]
            except:
                if op == "gt":   data = data[series >  str(val)]
                elif op == "gte":data = data[series >= str(val)]
                elif op == "lt": data = data[series <  str(val)]
                elif op == "lte":data = data[series <= str(val)]

    answer_text = ""
    preview = None

    # 3) Trend
    if trend and trend.get("by"):
        by_col = trend["by"]
        metric = trend.get("metric", {"op":"count","column":"customer_id","alias":"value"})
        if by_col not in data.columns:
            data["load_month"] = data.get("load_date", pd.Series()).apply(_month_from_date)
            by_col = "load_month"
        op = metric.get("op"); col = metric.get("column"); alias = metric.get("alias","value")
        if op == "count":
            grouped = data.groupby(by_col).agg(**{alias: pd.NamedAgg(column=col, aggfunc="count")}).reset_index()
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            func = {"sum":"sum","avg":"mean","min":"min","max":"max"}.get(op, "count")
            grouped = data.groupby(by_col).agg(**{alias: pd.NamedAgg(column=col, aggfunc=func)}).reset_index()
        grouped = grouped.sort_values(by=by_col)
        preview = grouped
        answer_text = f"Trend by {by_col}: {alias} over time."
        for o in order:
            c = o.get("column"); desc = o.get("dir","desc").lower()=="desc"
            if c in grouped.columns:
                grouped = grouped.sort_values(by=c, ascending=not desc)
                preview = grouped
        if topn and topn.get("column") in preview.columns:
            n = int(topn.get("n", 5)); preview = preview.head(n)

    # 4) Aggregation
    elif groupby and aggregates:
        agg_dict = {}
        for a in aggregates:
            op = a.get("op"); col = a.get("column"); alias = a.get("alias", f"{op}_{col}")
            if op == "count":
                agg_dict[alias] = pd.NamedAgg(column=col, aggfunc="count")
            elif op == "distinct_count":
                agg_dict[alias] = pd.NamedAgg(column=col, aggfunc=lambda x: x.nunique(dropna=True))
            elif op in {"sum","avg","min","max"}:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                func = {"sum":"sum","avg":"mean","min":"min","max":"max"}[op]
                agg_dict[alias] = pd.NamedAgg(column=col, aggfunc=func)

        grouped = data.groupby(groupby).agg(**agg_dict).reset_index()

        if percent and percent.get("of"):
            m = percent["of"]
            alias = percent.get("alias", f"pct_{m}")
            total = grouped[m].sum()
            grouped[alias] = (grouped[m] / total * 100.0).round(2) if total else 0.0

        for o in order:
            col = o.get("column"); direction = o.get("dir","desc").lower()=="desc"
            if col in grouped.columns:
                grouped = grouped.sort_values(by=col, ascending=not direction)

        if topn and topn.get("column") in grouped.columns:
            n = int(topn.get("n", 5))
            grouped = grouped.head(n)

        preview = grouped

        if len(groupby) == 1 and any(a.get("op")=="count" for a in aggregates):
            cnt_alias = next((a.get("alias", f"{a.get('op')}_{a.get('column')}") for a in aggregates if a.get("op")=="count"), None)
            parts = [f"{row[groupby[0]]}: {int(row[cnt_alias]) if pd.notna(row[cnt_alias]) else 0}"
                     for _,row in grouped.iterrows()]
            answer_text = "Counts by " + groupby[0] + " → " + "; ".join(parts)
        else:
            answer_text = f"Aggregated by {', '.join(groupby)} with {len(aggregates)} metric(s)."

    # 5) Growth
    if growth and growth.get("by") and preview is not None:
        by_col = growth["by"]
        metric_col = growth.get("metric")
        alias = growth.get("alias", "growth_rate")
        if by_col in preview.columns and metric_col in preview.columns:
            pr = preview.sort_values(by=by_col).reset_index(drop=True)
            pr[alias] = pr[metric_col].pct_change().round(4)
            preview = pr
            answer_text += f" | Added MoM growth on '{metric_col}' as '{alias}'."

    # 6) Selection-only / preview
    if preview is None:
        if select:
            keep = [c for c in select if c in data.columns]
            preview = data[keep] if keep else data
        else:
            preview = data
        if limit:
            preview = preview.head(int(limit))
        answer_text = f"Showing a preview of {len(preview)} row(s) after filters."

    # cap
    if limit and len(preview) > int(limit):
        preview = preview.head(int(limit))

    return answer_text, preview

# ==== NEW: workers for concurrent run & ask ====
def run_pipeline_worker(input_dir, output_dir, outfile, mode, use_llm, provider, llm_model, sf_cfg, sf_table, do_load, create_if_missing) -> str:
    """
    Runs the pipeline (classic/genai) and optionally loads into Snowflake.
    Returns multiline log string.
    """
    logs = []
    try:
        if mode == "classic":
            path = run_classic_pipeline(input_dir, output_dir, outfile)
        else:
            path = run_genai_pipeline(
                input_dir, output_dir, outfile,
                use_real_genai=use_llm,
                llm_provider=provider,
                llm_model=llm_model,
                api_key=os.getenv("GEMINI_API_KEY")
            )
        logs.append(f" Pipeline completed. Output: {path}")

        if do_load:
            rows = run_load(
                csv_path=path,
                cfg=sf_cfg,
                table_name=sf_table,
                create_if_not_exists=create_if_missing,
            )
            logs.append(f" Loaded {rows} rows into {sf_cfg['database']}.{sf_cfg['schema']}.{sf_table}")
        return "\\n".join(logs)
    except Exception as e:
        raise RuntimeError(f"Pipeline error: {e}")

def answer_question_worker(question: str, provider: str, model: str, output_dir: str, outfile: str) -> str:
    """
    Loads latest standardized CSV, builds a plan via LLM, executes it, and returns a formatted message string.
    """
    path = _latest_output_csv(output_dir, outfile)
    if not path:
        raise RuntimeError("No standardized output found. Please run the pipeline first.")
    df = _load_df(path)
    schema = _schema_summary(df, max_uniques=6)
    plan = _call_llm_json(provider, model, SYSTEM_QA, USER_QA(question, schema))
    answer_text, preview = _execute_plan(plan, df)

    max_rows = 15
    if preview is not None and len(preview) > max_rows:
        preview = preview.head(max_rows)

    table_str = preview.to_string(index=False) if preview is not None else "(no preview)"
    msg = [
        f" Question: {question}",
        f"️ Source: {path}",
        f" Plan: {json.dumps(plan, ensure_ascii=False)}",
        f" Answer: {answer_text}",
        "---- Preview ----",
        table_str
    ]
    return "\\n".join(msg)

def answer_question_after_output_worker(question, provider, model, output_dir, outfile, timeout_sec=180, poll_ms=500) -> str:
    """
    Starts immediately, waits for standardized output to appear/stabilize, then answers.
    Prevents 'no output' errors while still running concurrently with pipeline.
    """
    target = os.path.join(output_dir, outfile)
    elapsed = 0
    last_size = -1

    # wait until file exists and size becomes stable across two checks
    while elapsed < timeout_sec:
        if os.path.exists(target):
            size = os.path.getsize(target)
            if last_size == size and size > 0:
                break
            last_size = size
        time.sleep(poll_ms / 1000.0)
        elapsed += poll_ms / 1000.0

    if not os.path.exists(target):
        raise RuntimeError("Timed out waiting for standardized output. Please re-run.")

    df = _load_df(target)
    schema = _schema_summary(df, max_uniques=6)
    plan = _call_llm_json(provider, model, SYSTEM_QA, USER_QA(question, schema))
    answer_text, preview = _execute_plan(plan, df)

    max_rows = 15
    if preview is not None and len(preview) > max_rows:
        preview = preview.head(max_rows)

    table_str = preview.to_string(index=False) if preview is not None else "(no preview)"
    msg = [
        f" Question: {question}",
        f"️ Source: {target}",
        f" Plan: {json.dumps(plan, ensure_ascii=False)}",
        f" Answer: {answer_text}",
        "---- Preview ----",
        table_str
    ]
    return "\\n".join(msg)

@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {
        "input": INPUT_DIR,
        "output": OUTPUT_DIR,
        "outfile": "Transformed_file.csv",
        "mode": "classic",
        "use_llm": False,
        "provider": "gemini",
        "llm_model": "gemini-1.5-flash",
        "message": ""
    }
    ctx.update(_snowflake_ctx_from_env())
    ctx["sf_create"] = True
    ctx["sf_load"] = False
    ctx["job_id"] = ""
    ctx["job_id2"] = ""  # support a second concurrent job (Q&A)

    if request.method == "POST":
        # Read pipeline form
        ctx.update({
            "input": request.form.get("input") or ctx["input"],
            "output": request.form.get("output") or ctx["output"],
            "outfile": request.form.get("outfile") or ctx["outfile"],
            "mode": request.form.get("mode") or ctx["mode"],
            "use_llm": request.form.get("use_llm") == "on",
            "provider": request.form.get("provider") or ctx["provider"],
            "llm_model": request.form.get("llm_model") or ctx["llm_model"],
        })

        # ---------- Handle File Uploads ----------
        uploaded_files = request.files.getlist("files")
        saved_files = []
        if uploaded_files:
            # Clean the INPUT_DIR before new upload (optional)
            for old in os.listdir(ctx["input"]):
                try:
                    os.remove(os.path.join(ctx["input"], old))
                except Exception:
                    pass

            for f in uploaded_files:
                if f and allowed_file(f.filename):
                    filename = secure_filename(f.filename)
                    dest = os.path.join(ctx["input"], filename)
                    f.save(dest)
                    saved_files.append(dest)

        # Read Snowflake form
        sf_cfg = {
            "account": request.form.get("sf_account") or ctx["sf_account"],
            "user": request.form.get("sf_user") or ctx["sf_user"],
            "password": request.form.get("sf_password") or "",  # do not persist
            "warehouse": request.form.get("sf_warehouse") or ctx["sf_warehouse"],
            "database": request.form.get("sf_database") or ctx["sf_database"],
            "schema": request.form.get("sf_schema") or ctx["sf_schema"],
        }
        sf_table = request.form.get("sf_table") or ctx["sf_table"]
        ctx["sf_create"] = request.form.get("sf_create") == "on"
        ctx["sf_load"]   = request.form.get("sf_load") == "on"

        action = request.form.get("action", "run")

        try:
            if action == "test":
                # Test Snowflake connection only
                import snowflake.connector
                conn = snowflake.connector.connect(
                    user=sf_cfg["user"],
                    password=sf_cfg["password"],
                    account=sf_cfg["account"],
                    warehouse=sf_cfg["warehouse"],
                    database=sf_cfg["database"],
                    schema=sf_cfg["schema"],
                )
                with conn.cursor() as cur:
                    cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_REGION(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
                    info = cur.fetchone()
                conn.close()
                ctx["message"] = " Snowflake connection OK.\\nContext: " + str(info)
                return render_template_string(TEMPLATE, **ctx)

            elif action == "ask":
                question = (request.form.get("question") or "").strip()
                if not question:
                    raise RuntimeError("Please enter a question.")
                # Launch background job that waits for output if needed
                job_id = start_job(
                    answer_question_after_output_worker,
                    question,
                    ctx["provider"],
                    ctx["llm_model"],
                    ctx["output"],
                    ctx["outfile"]
                )
                ctx["job_id"] = job_id
                ctx["job_id2"] = ""
                ctx["message"] = f"🟡 Your question has been queued as job {job_id}. The result will appear here."
                return render_template_string(TEMPLATE, **ctx)

            elif action == "run_both":
                question = (request.form.get("question") or "").strip()
                if not question:
                    raise RuntimeError("Please enter a question for 'Run & Ask'.")

                # Start pipeline job
                pipe_job = start_job(
                    run_pipeline_worker,
                    ctx["input"], ctx["output"], ctx["outfile"],
                    ctx["mode"], ctx["use_llm"], ctx["provider"], ctx["llm_model"],
                    sf_cfg, sf_table, ctx["sf_load"], ctx["sf_create"]
                )

                # Start Q&A job that waits on output file
                ask_job = start_job(
                    answer_question_after_output_worker,
                    question, ctx["provider"], ctx["llm_model"], ctx["output"], ctx["outfile"]
                )

                ctx["job_id"] = pipe_job
                ctx["job_id2"] = ask_job
                ctx["message"] = (
                    f"🟡 Started pipeline (job {pipe_job}) and Q&A (job {ask_job}) concurrently.\\n"
                    f"The Q&A will begin automatically once the standardized output is ready."
                )
                return render_template_string(TEMPLATE, **ctx)

            # ---------- Run pipeline ----------
            if not saved_files and not os.listdir(ctx["input"]):
                raise RuntimeError("No CSV files uploaded. Please choose at least one .csv file and submit.")

            if ctx["mode"] == "classic":
                path = run_classic_pipeline(ctx["input"], ctx["output"], ctx["outfile"])
            else:
                path = run_genai_pipeline(
                    ctx["input"], ctx["output"], ctx["outfile"],
                    use_real_genai=ctx["use_llm"],
                    llm_provider=ctx["provider"],
                    llm_model=ctx["llm_model"],
                    api_key=os.getenv("GEMINI_API_KEY")
                )

            msg = [
                f" Uploaded {len(saved_files)} file(s):",
                *[f"   - {p}" for p in saved_files],
                f" Pipeline completed. Output: {path}"
            ]

            # Optional: load to Snowflake
            if ctx["sf_load"]:
                rows = run_load(
                    csv_path=path,
                    cfg=sf_cfg,
                    table_name=sf_table,
                    create_if_not_exists=ctx["sf_create"],
                )
                msg.append(f" Loaded {rows} rows into {sf_cfg['database']}.{sf_cfg['schema']}.{sf_table}")

            ctx["message"] = "\\n".join(msg)

        except Exception as e:
            ctx["message"] = f" Error: {e}"

    return render_template_string(TEMPLATE, **ctx)

# ==== job status route for polling ====
@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    info = JOBS.get(job_id)
    if not info:
        return jsonify({"status": "error", "message": "Unknown job id"}), 404
    return jsonify({"status": info["status"], "message": info["message"]})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
