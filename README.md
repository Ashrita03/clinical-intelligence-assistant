python - <<'PY'
from pathlib import Path

readme = r"""# Clinical Intelligence Assistant

### AI-Powered Medical Document Understanding with OCR, RAG & Conversational AI

[**Live Demo**](https://clinical-intelligence-ai.streamlit.app) | [**GitHub Repository**](https://github.com/Ashrita03/clinical-intelligence-assistant)

Clinical Intelligence Assistant is an end-to-end AI application that transforms medical reports into structured, understandable information using document processing, optical character recognition (OCR), Retrieval-Augmented Generation (RAG), and Qwen language models.

Built with **Python, Streamlit, PyMuPDF, Tesseract OCR, SentenceTransformers, FAISS, and Qwen2.5**, the application enables users to extract laboratory measurements, identify out-of-range results, and ask follow-up questions grounded in uploaded reports.

> **Medical disclaimer:** This is an educational and research prototype, not a clinically validated medical device. It does not provide medical diagnoses or treatment recommendations. Do not upload identifiable or sensitive patient information to the public demo.

## Live Application

**Try it here:** https://clinical-intelligence-ai.streamlit.app

The application is deployed on Streamlit Community Cloud and supports PDF and image uploads, OCR, medical report classification, laboratory extraction, interactive dashboards, and report-grounded conversational questions.

**Deployment:** Streamlit Community Cloud (Linux)  
**Deployment branch:** `feature/linux-deployment`  
**Local development:** Apple Silicon macOS

## Key Features

### Medical Document Processing
- Upload PDF, JPG, JPEG, and PNG reports.
- Extract selectable PDF text using PyMuPDF.
- Apply Tesseract OCR to scanned documents and images.
- Process documents containing different text layouts.

### Medical Report Classification
Classify documents into categories including:
- Laboratory reports
- Dental reports
- Cardiology reports
- Dermatology reports
- Gastroenterology reports
- Radiology and imaging reports
- Discharge summaries
- Clinical consultations

Classification is heuristic and may be inaccurate.

### Structured Laboratory Analysis
- Extract laboratory test names, numeric values, and units.
- Identify supported reference ranges.
- Process multi-column laboratory tables using layout-aware extraction.
- Categorize measurements as Normal, High, or Low based on report-provided reference ranges.
- Display laboratory findings and abnormal-result summaries.

**Verified example:** A 20-measurement CBC report was successfully processed in the deployed application, with 19 measurements classified as Normal and one as High. This is a functional demonstration, not a clinical validation.

### Retrieval-Augmented Generation (RAG)
- Create semantic embeddings using SentenceTransformers.
- Search report content using FAISS.
- Retrieve relevant passages for question answering.
- Display supporting report evidence when available.

### Conversational AI
- Ask natural-language questions about uploaded reports.
- Maintain multi-turn conversations.
- Answer supported laboratory-value questions using explicitly extracted information.
- Use Qwen language models for document-grounded responses.
- Reset conversation history when a different report is uploaded.

AI-generated answers may contain errors and should be verified against the original report.

### Interactive Streamlit Dashboard
- Document upload interface
- Report classification and summary metrics
- Laboratory results table
- Normal, High, and Low indicators
- Conversational AI interface
- Evidence inspection

## System Architecture

```text
Medical Report (PDF / JPG / JPEG / PNG)
                    |
                    v
             Streamlit UI
                    |
                    v
          Document Processing
             /          \
            v            v
         PyMuPDF     Tesseract OCR
             \          /
              v        v
             Extracted Text
                    |
          +---------+---------+
          |                   |
          v                   v
    Classification      Lab Extraction
          |                   |
          v                   v
    Report Metadata     Reference Ranges
          |                   |
          +---------+---------+
                    |
                    v
               Text Chunking
                    |
                    v
        SentenceTransformers
                    |
                    v
               FAISS Index
                    |
                    v
           Evidence Retrieval
                    |
                    v
             Grounded Q&A
                    |
                    v
            Qwen2.5 Model
                    |
                    v
          Conversational Response
