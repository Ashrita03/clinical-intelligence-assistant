# Clinical Intelligence Assistant

An end-to-end AI/ML healthcare research application that combines clinical PDF processing, structured laboratory data extraction, Retrieval-Augmented Generation (RAG), grounded question answering, and machine learning in an interactive Streamlit application.

> **Educational and research prototype only. This application is not intended for medical diagnosis, treatment recommendations, or professional medical advice.**

## Live Application

The application is publicly deployed using Streamlit Community Cloud.

**Live Demo:**  
https://clinical-intelligence-assistant.streamlit.app

## Overview

Clinical Intelligence Assistant demonstrates how machine learning and retrieval-based AI techniques can be integrated into a complete healthcare-oriented application.

Users can upload a clinical PDF report, automatically extract laboratory measurements, review reference-range status, and ask questions grounded in the uploaded report.

The application also includes a separate heart disease classification demonstration built using the UCI Heart Disease dataset.

## Key Features

### Clinical PDF Processing

- Upload clinical reports in PDF format
- Extract text using PyMuPDF
- Detect structured laboratory measurements
- Extract test name, value, unit, and reference range
- Classify extracted values as Normal, High, or Low

### Retrieval-Augmented Generation

- Split report text into overlapping chunks
- Generate semantic embeddings using SentenceTransformers
- Store and search embeddings using FAISS
- Retrieve report evidence relevant to the user's question
- Display retrieved evidence for transparency

### Grounded AI Question Answering

- Answer questions using information from the uploaded report
- Deterministically extract explicit laboratory values when available
- Use FLAN-T5 as a local language-model fallback
- Reject unsupported questions rather than inventing information
- Keep responses grounded in retrieved document evidence

### Machine Learning

A separate classification demonstration uses the UCI Heart Disease dataset.

Models evaluated during development included:

- Logistic Regression
- Random Forest
- Gradient Boosting

Logistic Regression was selected based on cross-validation ROC-AUC performance and consistency.

Evaluation of the selected model produced:

- Accuracy: 86.9%
- Precision: 81.3%
- Recall: 92.9%
- F1 Score: 86.7%
- ROC-AUC: 95.8%

These results are experimental results on the project dataset and do not represent clinical validation.

### Interactive Web Application

The Streamlit interface provides:

- PDF upload
- Structured laboratory results
- Summary metrics
- Extracted report text
- Grounded report Q&A
- Retrieved evidence inspection
- Heart disease classification demonstration
- Safety and research-use disclaimers

## System Architecture

```text
                         Clinical Intelligence Assistant
                                      |
                                      v
                             Streamlit Web Interface
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
             Clinical PDF                       ML Classification
                    |                                   |
                    v                                   v
             PyMuPDF Extraction                 UCI Heart Disease
                    |                                   |
          +---------+---------+                         v
          |                   |                  Preprocessing
          v                   v                         |
 Structured Lab Data      Report Text                  v
          |                   |                 Logistic Regression
          v                   v
 Reference-Range        Text Chunking
 Classification              |
                              v
                    SentenceTransformer
                         Embeddings
                              |
                              v
                            FAISS
                              |
                              v
                    Evidence Retrieval
                              |
                              v
                  Grounded Answer Pipeline
                       /             \
                      v               v
          Explicit Lab Extraction   FLAN-T5
                       \             /
                        v           v
                       Grounded Answer