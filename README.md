# Parafrasa & Cek Kemiripan

Web lokal untuk memparafrasakan teks dengan AI dan memeriksa kemiripannya terhadap sumber yang kamu pilih sendiri.

**Demo (simulasi, tanpa backend):** `demo/index.html` atau halaman GitHub Pages repo ini.

## Fitur
- **Banyak penyedia AI:** Claude, OpenAI, Google Gemini, DeepSeek, Groq, OpenRouter, dan Ollama (lokal, tanpa key). Pilih di kartu Penyedia AI, model bisa diganti manual.
- **Parafrasa**: gaya standar, formal, sederhana, ringkas; bahasa Indonesia atau English.
- **Skor makna terjaga:** cosine similarity SBERT antara teks asli dan hasil parafrasa.
- **Cek kemiripan hibrida:**
  - Leksikal: pencocokan frasa N-kata, teks yang cocok di-highlight per sumber.
  - Semantik: SBERT per kalimat (`paraphrase-multilingual-mpnet-base-v2`) untuk menangkap parafrasa.
- **Fine-tuning** model dengan data sendiri (`train.py`) plus kalibrasi ambang otomatis.

## Menjalankan
Pilih penyedia dan isi key di kartu **Penyedia AI** di UI (dikirim hanya ke backend lokalmu lewat header, tidak disimpan), atau lewat environment variable (lihat `.env.example`). Nama model bawaan bisa sudah usang; ganti lewat kolom Model.

```bash
git clone https://github.com/Huk21/paraphrase-web.git
cd paraphrase-web
pip install -r requirements.txt
python app.py                            # buka http://127.0.0.1:5000
```
Pertama kali dijalankan, model SBERT (sekitar 1 GB) diunduh dari HuggingFace.

## Fine-tuning (opsional)
Siapkan `data/pairs.csv` dengan kolom `sentence1,sentence2,label` (1 = parafrasa, 0 = bukan).
```bash
python train.py --data data/pairs.csv --epochs 4
export EMB_MODEL=model-ft
export SEM_TH=$(cat threshold.txt)
python app.py
```
Butuh minimal beberapa ratus pasang. Kalau skor setelah fine-tune tidak naik, pakai model dasar.

## Batasan
- **Ini bukan Turnitin.** Turnitin punya database global; alat ini hanya membandingkan dengan sumber yang kamu masukkan.
- Skor bergantung pada ambang (default 0.80) dan perlu dikalibrasi dengan data kamu.
- Parafrasa tidak menggantikan sitasi. Cantumkan sumber asli untuk ide yang bukan milikmu.
- Jangan commit API key. File `.env` sudah ada di `.gitignore`.

## Struktur
```
app.py            Flask backend (/api/paraphrase, /api/similarity, /api/semantic)
templates/        UI dashboard (terhubung ke backend)
demo/index.html   UI yang sama dengan hasil simulasi, untuk GitHub Pages
train.py          Fine-tuning SBERT + kalibrasi ambang
data/pairs.csv    Contoh format data
```

## Referensi
- Zhou dkk., *Paraphrase Identification with Deep Learning: A Review of Datasets and Methods*, arXiv 2212.06933
- Vrbanec & Meštrović, *Comparison study of unsupervised paraphrase detection*, Expert Systems, 2023
- Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019

Lisensi: MIT
