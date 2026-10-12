## Live Demo

**Try the Clinical Intelligence Assistant:**
https://clinical-intelligence-ai.streamlit.app

The deployed application supports medical PDF and image uploads, OCR, laboratory result extraction, report classification, and report-grounded conversational questions.

**Note:** This is an educational and research prototype, not a clinically validated diagnostic tool. Avoid uploading identifiable patient information.


# Clinical Intelligence Assistant

### Local AI-Powered Medical Document Understanding with OCR, RAG, and Conversational AI

Clinical Intelligence Assistant is an end-to-end AI application that transforms medical reports into structured, understandable information using document processing, optical character recognition (OCR), Retrieval-Augmented Generation (RAG), and local large language models.

Built with Python, Streamlit, PyMuPDF, Tesseract OCR, SentenceTransformers, FAISS, and Qwen2.5, the application enables users to explore medical reports and ask follow-up questions grounded in their uploaded documents.

> **Disclaimer:** Educational and research prototype only. Not intended for medical diagnosis, treatment decisions, or professional medical advice.

## Project Status

**Active development — local prototype**

The latest version runs locally on Apple Silicon using MLX. It has not yet been deployed as a publicly accessible version.

An earlier prototype was deployed at:

https://clinical-intelligence-assistant.streamlit.app

The earlier deployment should not be considered representative of the current local implementation.

## Key Features

### 1. Medical Document Processing

- Upload PDF, JPG, JPEG, and PNG medical reports
- Extract selectable text using PyMuPDF
- Apply Tesseract OCR to scanned PDF pages and uploaded images
- Support mixed PDFs containing selectable text and scanned pages
- Extract report information for downstream analysis

### 2. Medical Report Classification

Keyword-based document classification supports:

- Laboratory reports
- Dental reports
- Cardiology reports
- Dermatology and hair reports
- Gastroenterology reports
- Radiology and imaging reports
- Discharge summaries
- Clinical consultations

Reports without sufficient classification evidence are assigned a general category.

### 3. Structured Laboratory Analysis

- Extract laboratory test names
- Identify measured values and units
- Extract supported reference-range formats
- Compare results with document-provided reference ranges
- Categorize detected measurements as Normal, High, or Low
- Display findings in an interactive dashboard

Reference-range classification is not a medical diagnosis.

### 4. Retrieval-Augmented Generation

The document-grounded question-answering pipeline uses:

- SentenceTransformers for semantic embeddings
- FAISS for similarity search
- Retrieved report passages as supporting evidence
- Evidence inspection within the Streamlit interface

### 5. Local Conversational AI

The application uses:

**Qwen2.5-1.5B-Instruct-4bit**, running locally through MLX-LM.

Capabilities include:

- Natural-language questions about uploaded reports
- Multi-turn conversation
- Document-grounded answers
- Deterministic handling of supported laboratory measurements
- General educational explanations
- Conversation reset when a different report is uploaded

The application separates reported findings from general medical information.

### 6. Interactive Streamlit Dashboard

The interface provides:

- Medical document uploads
- Detected document type
- Report overview and summary metrics
- Structured laboratory findings
- Out-of-range measurement indicators
- Conversational AI
- Retrieved evidence inspection
- Educational-use disclaimers

## System Architecture

```text
               Medical Report
            PDF / JPG / JPEG / PNG
                      |
                      v
              Streamlit Interface
                      |
                      v
             Document Processing
                /           \
               v             v
          PyMuPDF         Tesseract OCR
               \             /
                v           v
                Extracted Text
                      |
           +----------+----------+
           |                     |
           v                     v
    Report Classification   Lab Extraction
           |                     |
           v                     v
    Report Metadata       Reference-Range
                              Analysis
           |                     |
           +----------+----------+
                      |
                      v
               Text Chunking
                      |
                      v
          SentenceTransformer Embeddings
                      |
                      v
                  FAISS Index
                      |
                      v
               Evidence Retrieval
                      |
                      v
             Grounded Q&A Pipeline
                      |
                      v
             Local Qwen2.5 via MLX
                      |
                      v
             Conversational Response
```

The application uses separate worker processes for retrieval and language-model operations to improve stability in its local environment.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Web Interface | Streamlit |
| PDF Processing | PyMuPDF |
| OCR | Tesseract, pytesseract, Pillow |
| Language Model | Qwen2.5-1.5B-Instruct-4bit |
| Local Inference | MLX-LM |
| Embeddings | SentenceTransformers |
| Vector Search | FAISS |
| Data Processing | pandas |
| Development | VS Code, Conda, Git |

## Local Installation

### Requirements

- macOS with Apple Silicon for the current MLX-based implementation
- Python 3.11
- Conda
- Homebrew
- Tesseract OCR
- Internet access for initial model downloads

### 1. Clone the repository

```bash
git clone https://github.com/Ashrita03/clinical-intelligence-assistant.git
cd clinical-intelligence-assistant
```

### 2. Create the environment

```bash
conda create -n clinical-ai python=3.11
conda activate clinical-ai
```

### 3. Install Tesseract

```bash
brew install tesseract
```

### 4. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

The dependency file must be kept synchronized with the current implementation before this installation procedure can be considered fully verified.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

http://localhost:8501

## Usage

1. Start the Streamlit application.
2. Upload a supported medical report.
3. Review the detected document type.
4. Explore extracted findings and laboratory measurements.
5. Ask questions about the report.
6. Inspect retrieved evidence when available.
7. Upload a different report to start a new document conversation.

Use fictional or appropriately de-identified reports during development.

## Testing

Initial local tests have successfully demonstrated:

- Text extraction from a standard dental PDF
- OCR extraction from a synthetic scanned laboratory PDF
- JPG medical report processing
- PNG medical report processing
- Laboratory value and reference-range extraction
- Document-grounded chatbot answers about extracted glucose measurements

These tests use synthetic documents and do not establish clinical accuracy or reliability across real-world medical reports.

## Limitations

- OCR accuracy depends on image quality, resolution, and layout.
- OCR may misread laboratory values, units, medication names, and reference ranges.
- Structured laboratory extraction currently supports limited text patterns.
- Keyword-based report classification can produce incorrect categories.
- Language-model responses can be incomplete or inaccurate.
- The current MLX inference backend is specific to compatible Apple Silicon systems.
- The current implementation has not been clinically validated.
- Production-grade privacy, security, and medical-data handling controls have not been established.

**Do not use this prototype to make medical decisions or upload sensitive patient information to an unverified public deployment.**

## Future Improvements

- Broader testing across medical report types
- More robust laboratory table extraction
- OCR confidence checks and extraction validation
- Improved document classification
- Better response quality and evidence attribution
- Automated regression tests
- Privacy and security improvements
- Deployment-compatible inference backend
- Updated public demonstration

## Research and Development Focus

This project demonstrates practical experience with:

- Applied natural language processing
- Retrieval-Augmented Generation
- Local large language models
- Document intelligence and OCR
- Information extraction
- Semantic search and vector databases
- AI application engineering
- Interactive data applications

## License

A license has not yet been specified. See the repository for current usage permissions.

---

**Clinical Intelligence Assistant — An educational AI document intelligence research project.**
