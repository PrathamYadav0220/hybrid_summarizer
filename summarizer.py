# summarizer.py
import re
import numpy as np
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

# Load models inside this module if you prefer; we'll load from app.py for better control
# But provide functions that accept models/tokenizers as arguments.

# Sentence splitter (basic, uses nltk optionally)
def split_into_sentences(text):
    # simple split with punctuation — fallback if nltk not installed
    import nltk
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except Exception:
        # fallback naive splitting
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    # clean
    sentences = [s.strip() for s in sentences if len(s.strip())>10]
    return sentences

def get_bert_sentence_embeddings(sentences, bert_tokenizer, bert_model, device='cpu', max_len=128):
    """
    For each sentence, compute token embeddings using BERT and return a sentence embedding vector.
    Uses layer N-2 (two layers before last): hidden_states[-3].
    Returns numpy array of shape (n_sentences, hidden_size)
    """
    bert_model.to(device)
    bert_model.eval()
    embeddings = []
    with torch.no_grad():
        for s in sentences:
            toks = bert_tokenizer(s, return_tensors='pt', truncation=True, max_length=max_len, padding='max_length')
            input_ids = toks['input_ids'].to(device)
            attention_mask = toks['attention_mask'].to(device)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layerN)
            # pick N-2 layer:
            if len(hidden_states) >= 3:
                layer = hidden_states[-3]  # shape (1, seq_len, hidden_size)
            else:
                layer = hidden_states[-1]
            # mask out padding tokens when averaging
            mask = attention_mask.unsqueeze(-1)  # (1, seq_len, 1)
            layer = layer * mask
            summed = layer.sum(dim=1)  # (1, hidden_size)
            counts = mask.sum(dim=1).clamp(min=1)
            sent_emb = (summed / counts).squeeze(0).cpu().numpy()
            embeddings.append(sent_emb)
    return np.vstack(embeddings) if embeddings else np.zeros((0, bert_model.config.hidden_size))

def extractive_summary(text, bert_tokenizer, bert_model, num_sentences=5, device='cpu'):
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    n_sent = min(num_sentences, len(sentences))
    emb = get_bert_sentence_embeddings(sentences, bert_tokenizer, bert_model, device=device)
    # if we have less sentences than clusters, just return them
    if len(sentences) <= n_sent:
        return " ".join(sentences)
    # KMeans
    kmeans = KMeans(n_clusters=n_sent, random_state=42, n_init=10)
    kmeans.fit(emb)
    centers = kmeans.cluster_centers_
    selected_idx = []
    for c in centers:
        dists = np.linalg.norm(emb - c, axis=1)
        idx = int(np.argmin(dists))
        selected_idx.append(idx)
    selected_idx = sorted(set(selected_idx))
    selected_sentences = [sentences[i] for i in selected_idx]
    # keep original order
    selected_sentences_sorted = sorted(selected_sentences, key=lambda s: sentences.index(s))
    return " ".join(selected_sentences_sorted)

def abstractive_summary(text, t5_tokenizer, t5_model, device='cpu', max_length=150, min_length=30, num_beams=4):
    """
    T5-based abstractive summarization. Truncates input to T5 max length.
    """
    t5_model.to(device)
    t5_model.eval()
    input_text = "summarize: " + text.strip()
    # encode with truncation
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        summary_ids = t5_model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def hy_summarizer(full_text, bert_tokenizer, bert_model, t5_tokenizer, t5_model,
                     device='cpu', extract_sentences=5, t5_max_length=150, t5_min_length=30, t5_num_beams=4):
    # Stage 1: extractive
    extract = extractive_summary(full_text, bert_tokenizer, bert_model, num_sentences=extract_sentences, device=device)
    if not extract:
        return ""
    # Stage 2: abstractive
    abstr = abstractive_summary(extract, t5_tokenizer, t5_model, device=device, max_length=t5_max_length,
                                min_length=t5_min_length, num_beams=t5_num_beams)
    return abstr
