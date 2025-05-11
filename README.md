# 📄 Smart Summarizer

**Smart Summarizer** is an AI-powered Streamlit application that summarizes long-form content from various input sources—**text**, **PDF files**, and **web URLs**—into concise, bullet-point summaries. It uses Google's Long-T5 transformer model to handle extended input lengths, and allows users to customize the number of summary points and include additional contextual notes.

---

## 🚀 Features

- 📋 **Text Summarization** – Paste long content and get summarized output
- 📄 **PDF Summarization** – Upload academic papers or reports in PDF format
- 🌐 **Web Article Summarization** – Enter any article/blog URL to extract and summarize its content
- 📝 **Notes Input** – Add personal notes or focus areas to guide the summarizer
- 🔢 **Custom Summary Length** – Choose the number of bullet points in your summary
- ⚡ Powered by `google/long-t5-tglobal-base` model for high-quality results

---

## 🧠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Model**: [Transformers – Long-T5](https://huggingface.co/google/long-t5-tglobal-base)
- **Libraries**:
  - `transformers`, `torch` – for model inference
  - `PyMuPDF` (`fitz`) – for PDF text extraction
  - `newspaper3k` – for extracting article content from URLs
  - `lxml_html_clean` – required by `newspaper3k` for HTML parsing

---

## 🖥️ How to Run

### 1. Clone the repository


git clone https://github.com/yourusername/smart-summarizer.git
cd smart-summarizer

 **Install dependencies**
 pip install -r requirements.txt
 pip install lxml[html_clean]

---

 **Run the app**
 streamlit run app.py

 **Project Structure**
 smart-summarizer/
│
├── app.py               # Streamlit application code
├── requirements.txt     # List of required Python packages
└── README.md            # Project documentation

---

**Use Cases**
Summarizing academic papers

Digesting lengthy blog articles or reports

Creating executive summaries with contextual focus

Automating note-taking and content review

---

📝 **Example Summary Output**
**Input**: 6-page research article (PDF)
**Notes**: "Focus on methodology and key findings"
**Output**:

The study uses a comparative model to evaluate data across two domains.

Key findings highlight a 34% improvement in response accuracy.

The dataset spans 10,000 entries from healthcare and education sectors.

Limitations include small sample size and non-randomized trials.

Future work suggests using a hybrid neural-symbolic approach.


📬 **Contact**

If you like the project or want to collaborate, feel free to connect:

GitHub: (https://github.com/KamalTeckchandani)

LinkedIn: https://www.linkedin.com/in/kamal-teckchandani/
