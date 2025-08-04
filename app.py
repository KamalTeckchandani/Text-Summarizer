import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from newspaper import Article
import torch
import re

# ---- Device setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Long-T5 model ----
model_name = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def bulletify(text):
    """
    Split summary text into sentences and format as a bullet list
    """
    # Basic sentence splitting by period, exclamation, or question
    lines = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty lines and bullets that are just whitespace
    lines = [line.strip() for line in lines if line.strip()]
    # Prepend bullet to each line
    return "\n".join([f"- {line}" for line in lines])

# --- Summarizer function ---
def summarize_text(text, max_input_length=8192, max_output_length=512):
    # Tokenize and move tensors to correct device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate summary
    output = model.generate(
        inputs["input_ids"],
        max_length=max_output_length,
        min_length=30,
        length_penalty=1.5,
        num_beams=4,
        early_stopping=True
    )
    summary_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return bulletify(summary_text)

# --- PDF text extraction ---
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# --- Web URL content extraction ---
def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Summarizer", layout="centered")
st.title("📄 Smart Summarizer – PDF | Text | Web")

# Common inputs
num_points = st.number_input(
    "🔢 Number of summary points (approx.):", min_value=1, max_value=20, value=5
)
notes_input = st.text_area("📝 Notes to include in context (optional):", height=100)

# Tabs for input methods
tab1, tab2, tab3 = st.tabs(["📋 Paste Text", "📄 Upload PDF", "🌐 Enter URL"])

with tab1:
    text_input = st.text_area("Paste your content here:", height=300)
    if st.button("Summarize Text"):
        if text_input:
            with st.spinner("Summarizing..."):
                combined_text = f"{text_input}\n\nAdditional context:\n{notes_input}"
                summary = summarize_text(combined_text, max_output_length=num_points * 40)
                st.success("Summary (Bulleted):")
                st.markdown(summary)
        else:
            st.warning("Please paste some content first.")

with tab2:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text Preview:", extracted_text[:3000], height=300)
        if st.button("Summarize PDF"):
            with st.spinner("Summarizing PDF..."):
                combined_text = f"{extracted_text}\n\nAdditional context:\n{notes_input}"
                summary = summarize_text(combined_text, max_output_length=num_points * 40)
                st.success("Summary (Bulleted):")
                st.markdown(summary)

with tab3:
    url_input = st.text_input("Enter the article URL:")
    if st.button("Summarize Web Content"):
        if url_input:
            try:
                with st.spinner("Extracting and summarizing web content..."):
                    web_text = extract_text_from_url(url_input)
                    combined_text = f"{web_text}\n\nAdditional context:\n{notes_input}"
                    summary = summarize_text(combined_text, max_output_length=num_points * 40)
                    st.success("Summary (Bulleted):")
                    st.markdown(summary)
            except Exception as e:
                st.error(f"Error reading the URL: {e}")
        else:
            st.warning("Please enter a valid URL.")
