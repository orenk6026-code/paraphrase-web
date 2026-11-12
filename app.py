import os, re
from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)
# ---------- Penyedia AI ----------
# Model bawaan bisa usang: ubah lewat kolom Model di UI atau env <PROVIDER>_MODEL (mis. OPENAI_MODEL).
PROVIDERS = {
    "claude":     {"kind": "anthropic", "url": "https://api.anthropic.com/v1/messages", "model": "claude-sonnet-5-5", "env": "ANTHROPIC_API_KEY"},
    "openai":     {"kind": "openai", "url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o-mini", "env": "OPENAI_API_KEY"},
    "gemini":     {"kind": "gemini", "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", "model": "gemini-2.0-flash", "env": "GEMINI_API_KEY"},
    "deepseek":   {"kind": "openai", "url": "https://api.deepseek.com/chat/completions", "model": "deepseek-chat", "env": "DEEPSEEK_API_KEY"},
    "groq":       {"kind": "openai", "url": "https://api.groq.com/openai/v1/chat/completions", "model": "llama-3.3-70b-versatile", "env": "GROQ_API_KEY"},
    "openrouter": {"kind": "openai", "url": "https://openrouter.ai/api/v1/chat/completions", "model": "openai/gpt-4o-mini", "env": "OPENROUTER_API_KEY"},
    "ollama":     {"kind": "openai", "url": "http://localhost:11434/v1/chat/completions", "model": "llama3.1", "env": None},
}

def call_llm(provider, key, model, system, text):
    cfg = PROVIDERS[provider]
    model = model or os.environ.get(provider.upper() + "_MODEL") or cfg["model"]
    key = key or (os.environ.get(cfg["env"], "") if cfg["env"] else "")
    if cfg["env"] and not key:
        raise ValueError("Isi API key untuk " + provider + " di kolom Penyedia AI.")
    kind, url = cfg["kind"], cfg["url"].format(model=model)
    if kind == "anthropic":
        r = requests.post(url, timeout=90, headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 2000, "system": system, "messages": [{"role": "user", "content": text}]})
    elif kind == "gemini":
        r = requests.post(url, timeout=90, headers={"x-goog-api-key": key},
            json={"system_instruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": text}]}]})
    else:
        h = {"Authorization": "Bearer " + key} if key else {}
        r = requests.post(url, timeout=90, headers=h,
            json={"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]})
    if not r.ok:
        raise RuntimeError(provider + " error " + str(r.status_code) + ": " + r.text[:200])
    d = r.json()
    if kind == "anthropic":
        return "".join(b.get("text", "") for b in d["content"])
    if kind == "gemini":
        return "".join(x.get("text", "") for x in d["candidates"][0]["content"]["parts"])
    return d["choices"][0]["message"]["content"]

MODES = {
    "standard": "Parafrasakan dengan kata dan struktur kalimat berbeda, makna tetap sama.",
    "formal": "Parafrasakan dengan gaya akademik/formal.",
    "simple": "Parafrasakan dengan bahasa yang lebih sederhana dan mudah dipahami.",
    "concise": "Parafrasakan dengan lebih ringkas tanpa menghilangkan poin penting.",
}

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/paraphrase")
def paraphrase():
    d = request.get_json(force=True)
    text = (d.get("text") or "").strip()
    if not text:
        return jsonify(error="Teks kosong"), 400
    mode = MODES.get(d.get("mode"), MODES["standard"])
    lang = d.get("lang", "sama dengan teks asli")
    provider = (request.headers.get("X-Provider") or "claude").lower()
    if provider not in PROVIDERS:
        return jsonify(error="Provider tidak dikenal"), 400
    system = (f"Kamu alat parafrasa. {mode} Bahasa output: {lang}. "
              "Jangan menambah fakta baru. Keluarkan hanya hasil parafrasa, tanpa pembuka atau penutup.")
    try:
        out = call_llm(provider, (request.headers.get("X-API-Key") or "").strip(),
                       (request.headers.get("X-Model") or "").strip(), system, text)
        fid = None
        m = get_model()
        if m is not None:
            e = m.encode([text, out], normalize_embeddings=True)
            fid = round(float(e[0] @ e[1]) * 100, 1)
        return jsonify(result=out, fidelity=fid)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=str(e)), 500

def tokens(s):
    return re.findall(r"\w+", s.lower())

def shingles(toks, n):
    return {tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)}

@app.post("/api/similarity")
def similarity():
    d = request.get_json(force=True)
    text = d.get("text") or ""
    sources = [s for s in (d.get("sources") or []) if s.strip()]
    n = int(d.get("n", 4))
    words = re.findall(r"\S+", text)
    norm = [(tokens(w) or [""])[0] for w in words]
    if len(norm) < n or not sources:
        return jsonify(error="Isi teks (min. %d kata) dan minimal 1 sumber pembanding" % n), 400

    flags = [-1] * len(words)  # -1 = tidak cocok, else index sumber
    per_source = []
    for si, src in enumerate(sources):
        sh = shingles(tokens(src), n)
        covered = set()
        for i in range(len(norm) - n + 1):
            if tuple(norm[i:i+n]) in sh:
                covered.update(range(i, i + n))
        for i in covered:
            if flags[i] == -1:
                flags[i] = si
        per_source.append(round(100 * len(covered) / len(words), 1))
    total = round(100 * sum(f != -1 for f in flags) / len(words), 1)
    return jsonify(total=total, per_source=per_source,
                   words=[{"w": w, "s": f} for w, f in zip(words, flags)])


# ---------- Semantik (SBERT) ----------
# paraphrase-multilingual-mpnet-base-v2: model paraphrase multibahasa (50+ bahasa, termasuk Indonesia).
# Batas 128 token, jadi dipakai per kalimat.
EMB_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
_model = None
_model_failed = False

def get_model():
    global _model, _model_failed
    if _model is None and not _model_failed:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMB_NAME)
        except Exception as e:
            print("Model embedding gagal dimuat:", e)
            _model_failed = True
    return _model

def split_sentences(t):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", t) if len(x.strip()) > 15]

@app.post("/api/semantic")
def semantic():
    d = request.get_json(force=True)
    th = float(d.get("threshold", os.environ.get("SEM_TH", 0.80)))
    m = get_model()
    if m is None:
        return jsonify(error="Model embedding belum terpasang. Jalankan: pip install sentence-transformers"), 500
    q = split_sentences(d.get("text") or "")
    refs = []  # (index sumber, kalimat)
    for si, src in enumerate(d.get("sources") or []):
        refs += [(si, x) for x in split_sentences(src)]
    if not q or not refs:
        return jsonify(error="Teks atau sumber terlalu pendek"), 400
    eq = m.encode(q, normalize_embeddings=True)
    er = m.encode([r[1] for r in refs], normalize_embeddings=True)
    sim = eq @ er.T
    items = []
    for i, sent in enumerate(q):
        j = int(sim[i].argmax())
        items.append({"text": sent, "score": round(float(sim[i][j]), 3),
                      "source": refs[j][0], "match": refs[j][1], "flag": bool(sim[i][j] >= th)})
    pct = round(100 * sum(x["flag"] for x in items) / len(items), 1)
    return jsonify(items=items, percent=pct, threshold=th)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
