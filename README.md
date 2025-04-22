# 📑 SmartBuddy - A NotebookLM-Inspired AI Knowledge Assistant

This is a Streamlit-based app that acts like a personalized AI study buddy. It allows you to upload PDFs, URLs, YouTube links, and raw text — and then chat with the knowledge base.

---

## 🚀 Features

- Upload PDF/TXT/YouTube/URLs
- Chat-based Q&A UI
- Typing animation for bot replies
- Uses local LLM via Ollama
- Code highlighting with language detection
- GPU-ready (Tested on RTX 4080)

---

## ⚒️ Setup Instructions

### 1. Clone the Repository

Open VS Code terminal and run:

```bash
git clone https://github.com/ShriKirupa/ML_Project.git
cd ML_Project
```

---

### 2. Install Python 3.12 (if not installed)

Download from: [https://www.python.org/downloads/release/python-3120/](https://www.python.org/downloads/release/python-3120/)

Make sure to check ✅ **"Add Python to PATH"** during installation.

Then verify:

```bash
python --version
```

---

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

If `pip` gives errors, try updating it:

```bash
python -m pip install --upgrade pip
```

---

### 4.Install Ollama


Install it for windows and run the command
```bash
ollama pull deepseek-r1
```
To verify it runs,
```bash
ollama run deepseek-r1
```
---

### 5. Run the Streamlit App

```bash
streamlit run main.py
```

It will open in your browser (usually at [http://localhost:8501](http://localhost:8501)).

---

### 6. Push Code Updates to GitHub

After making code changes:

```bash
git add .
git commit -m "Your message here"
git push
```

---

---

## ✨ Pro Tips

- Keep your `requirements.txt` updated
- Test your environment by running: `python main.py` or `streamlit run main.py`
- For new models, check Ollama's GPU settings and log outputs

---

Happy learning with SmartBuddy! ✨

