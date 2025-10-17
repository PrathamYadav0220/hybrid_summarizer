# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import io
from PyPDF2 import PdfReader
import docx
import torch
import nltk

# download punkt if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from summarizer import hy_summarizer

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {'pdf', 'txt', 'docx'}

app = Flask(__name__, 
            template_folder=os.path.join(current_dir, 'templates'),
            static_folder=os.path.join(current_dir, 'static'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-secure-key"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# --- Load models once on startup ---
print("Loading BERT tokenizer/model (this may take a while)...")
bert_name = "google-bert/bert-base-cased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
# Use AutoModel to get hidden states (not ForMaskedLM)
bert_model = AutoModel.from_pretrained(bert_name, output_hidden_states=True)

print("Loading T5 tokenizer/model (flan-t5-base)...")
t5_name = "google/flan-t5-base"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_name)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", summary=None)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def extract_text_from_file(file_stream, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == "pdf":
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text
    elif ext == "txt":
        file_stream.seek(0)
        return file_stream.read().decode('utf-8')
    elif ext == "docx":
        # python-docx requires a path or file-like object; create temporary file-like
        file_stream.seek(0)
        doc = docx.Document(file_stream)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return "\n".join(fullText)
    else:
        return ""

@app.route("/summarize", methods=["POST"])
def summarize():
    # Get parameters
    extract_sentences = int(request.form.get("extract_sentences", 5))
    t5_max_len = int(request.form.get("t5_max_length", 150))
    t5_min_len = int(request.form.get("t5_min_length", 30))
    t5_num_beams = int(request.form.get("t5_num_beams", 4))

    # Either text area or file upload
    text_input = request.form.get("text_input", "").strip()
    uploaded_file = request.files.get("file")

    full_text = ""
    if uploaded_file and uploaded_file.filename != "":
        filename = secure_filename(uploaded_file.filename)
        if not allowed_file(filename):
            flash("Unsupported file type.")
            return redirect(url_for('index'))
        # read content
        file_stream = uploaded_file.stream
        full_text = extract_text_from_file(file_stream, filename)
    elif text_input:
        full_text = text_input
    else:
        flash("Please provide a text or upload a file.")
        return redirect(url_for('index'))

    if not full_text.strip():
        flash("Could not extract text from uploaded file or input is empty.")
        return redirect(url_for('index'))

    # Run hybrid summarizer
    try:
        summary = hy_summarizer(
            full_text,
            bert_tokenizer=bert_tokenizer,
            bert_model=bert_model,
            t5_tokenizer=t5_tokenizer,
            t5_model=t5_model,
            device=device,
            extract_sentences=extract_sentences,
            t5_max_length=t5_max_len,
            t5_min_length=t5_min_len,
            t5_num_beams=t5_num_beams
        )
    except Exception as e:
        flash(f"Error during summarization: {str(e)}")
        return redirect(url_for('index'))

    return render_template("index.html", summary=summary, original=full_text[:5000])

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == "__main__":
    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=True)