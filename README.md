FinInsight — An Intelligent Financial Chatbot

FinInsight (also referred to as FinSight in the codebase) is an educational financial-research prototype designed to bridge the gap between qualitative management commentary and quantitative financial indicators. It processes unstructured earnings-call transcripts and structured SEC financial datasets to curate an evidence-based knowledge layer, which is then explored through a retrieval-grounded conversational interface.

Academic Context: This project was developed by B. Shabareshwar Reddy, Nayanojwal Sri Tej, and Sreekar Bharadwaz (Vellore Institute of Technology, Chennai) for CSE4074 – Forensic Analytics.

Note: For a comprehensive dive into the methodology, architectural diagrams, data pipeline evaluations, and experimental models, please refer to the detailed project report included in this repository: FinInsight_Project_Report_with_Figures.docx.

Disclaimer: FinInsight is an academic/educational prototype. It is provided “as is,” without a guarantee of accuracy, completeness, or fitness for a particular purpose. It does not provide investment, legal, tax, or accounting advice.

✨ Key Features

Company Selection: Select one of ten supported technology companies and optionally compare it with a peer.

Curated Financial Facts: View consolidated data from SEC 10-K/10-Q filings, including revenue, net income, operating income, gross profit, assets, liabilities, cash, diluted EPS, and operating cash flow.

Smart Comparisons: Calculates year-over-year changes only when the two values have matching period lengths and corresponding period-end dates one year apart.

Contextual Evidence Retrieval: See up to five transcript sentences selected by keyword overlap with the question, along with their source filename, speaker, sentiment, and category labels.

Traceability: Shows SEC accession numbers, reporting periods, and XBRL tags in the model context for absolute transparency.

LLM Guardrails: The system prompt explicitly directs the model to cite only supplied evidence, report missing company-specific facts instead of hallucinating, and strictly avoid investment recommendations.

🏗️ System Architecture

FinInsight is organized around two primary evidence streams and a conversational retrieval layer.

1. Transcript Stream (Unstructured NLP)

Earnings-call PDFs are ingested and processed to preserve conversational context:

Text Extraction: PyMuPDF extracts raw text.

Speaker-Aware Dynamic Chunking (HSADC): Regular expressions identify Name, Title: patterns to group text by speaker turns (CEO, CFO, Analyst, etc.).

Sentence Segmentation: NLTK splits chunks into sentences.

NLP Classification: Sentences are categorized by rule-based keywords, scored for sentiment via FinBERT, and classified for risk and forward-guidance via BART-MNLI (Zero-Shot).

2. SEC Stream (Structured Financial Data)

Company-reported financial facts are processed from the SEC Financial Statement Data Sets (2025 Q4 release):

Facts are filtered by exact company legal names and exact XBRL tags.

Only consolidated values (excluding segment breakdowns) are kept.

The data is structured into a curated CSV mapping values to metrics, periods, and accession numbers.

3. Evidence Retrieval & Chatbot Prototype (RAG)

Company-Scoped Context: The application builds a context block containing the two most recent values of each financial metric, along with relevant transcript sentences retrieved via keyword overlap.

Generation: A hosted NVIDIA language model (nemotron-3.5-lightning-30b-a3b) processes the prompt and user query to generate a natural-language response complete with inline citations.

(Note: The multi-agent orchestration layer proposed in the theoretical design remains a planned future enhancement).

🛠️ Technologies Used

Document Extraction & Preprocessing

Python: Core programming language.

PyMuPDF (fitz): Extracts text from 228 PDF files.

Regex: Powers the speaker chunking and numerical/temporal expression extraction.

NLTK (punkt): Handles sentence segmentation.

Natural Language Processing (NLP)

Hugging Face Transformers: Framework for model pipelines.

ProsusAI/finbert: Classifies financial sentiment (Positive, Negative, Neutral) and assigns confidence scores across ~29,500 sentences.

facebook/bart-large-mnli: Zero-shot classifier categorizing Forward-Guidance (e.g., Revenue Guidance, CapEx) and Risks (e.g., Competitive Pressure).

Data Processing & Experimental Modeling

pandas & NumPy: Data manipulation and structuring of the massive SEC datasets.

pymannkendall: Experimental temporal trend analysis on extracted metrics.

TOPSIS (via NumPy): Experimental Multi-Criteria Decision Analysis ranking system.

XGBoost & scikit-learn: Experimental regressor prototype.

Conversational Interface

Streamlit: Frontend web application framework.

OpenAI Python Client: API communication format.

NVIDIA Hosted LLM: The inference engine powering the chatbot.

📊 Data and Coverage

The data/ directory contains prepared files used by the application:

financial_facts.csv — 251 curated, consolidated SEC facts for 10 companies.

filings_manifest.csv — Selected filing metadata.

transcript_evidence.csv — 9,221 recognized transcript sentences for 8 companies.

Note: Transcript coverage is incomplete (e.g., AMD and IBM have no transcript records in the prepared export due to formatting constraints during chunking). Data reflects the 2025 Q4 SEC release and is not automatically refreshed by the app.

🚀 Getting Started (Local Setup)

1. Requirements

Python: 3.12 recommended.

API Key: An NVIDIA API key for the LLM endpoint.

2. Installation

Create and activate a virtual environment, then install dependencies:

# Using uv (or standard venv)
uv venv --python 3.12

# macOS/Linux activation
source .venv/bin/activate  

# Windows activation
.venv\Scripts\Activate.ps1

# Install requirements
uv pip install -r requirements.txt


3. Configure API Key

Set your NVIDIA API key as an environment variable (or enter it in the app UI):

# macOS/Linux
export NVIDIA_API_KEY="your-api-key"
export NVIDIA_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b"

# Windows (PowerShell)
$env:NVIDIA_API_KEY = "your-api-key"
$env:NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


(Alternatively, create .streamlit/secrets.toml locally. Never commit this file to Git).

4. Run the App

python -m streamlit run app.py


🧪 Testing & Data Generation

Run the Test Suite:
From the repository root, run the built-in unittest suite to verify data selection, context formatting, and guardrails:

python -m unittest discover -s tests -v


Regenerate Data (Optional):
If you have the raw SEC export and transcript directories, you can rebuild the CSVs:

python prepare_data.py \
  --sec-csv "/path/to/chatbot_data/financial_merged.csv" \
  --transcripts-dir "/path/to/unstructured_final_results" \
  --output-dir "./data"


☁️ Deployment (Streamlit Community Cloud)

Push your repository to GitHub (ensure requirements.txt and the data/ folder are included). Do not commit API keys.

Log in to Streamlit Community Cloud and click "New app".

Select your repository, branch, and set the main file path to app.py.

Navigate to Advanced settings > Secrets and add your credentials:

NVIDIA_API_KEY = "your-api-key"
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


Click Deploy.

⚠️ Known Limitations & Future Enhancements

Current Limitations:

Retrieval Mechanism: Keyword-overlap retrieval may miss relevant sentences that use synonymous phrasing (lacking dense semantic embeddings).

Extraction Brittleness: Regex-based speaker chunking failed on non-standard PDF formats (106 out of 228 PDFs yielded no parsed dialogue).

Experimental Scope: The TOPSIS ranking and XGBoost models are exploratory notebook experiments (XGBoost uses a simulated target variable) and are not used by the live chatbot.

Sentiment Context: FinBERT scores represent the model's confidence in linguistic tone, which does not directly correlate with actual financial health or stock performance.

Future Work:

Implement Semantic Search (Vector Embeddings) for retrieval.

Integrate OCR for image-based PDFs to improve parsing yield.

Deploy the theoretical Multi-Agent Orchestration layer (Transcript Agent, Financial Agent, Ranking Agent).

Align transcript evidence and SEC facts chronologically by precise fiscal quarters.
