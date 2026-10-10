# FinSight — financial research chatbot

Streamlit app that answers questions using the curated SEC filing facts and earnings-call transcript evidence in `data/`. The LLM is used to explain retrieved evidence; it is instructed not to invent missing facts or provide investment recommendations.

## Run locally

1. Use Python 3.12 (the current Community Cloud default) and create an environment:

   ```bash
   uv venv --python 3.12
   uv pip install --python .venv/Scripts/python.exe -r requirements.txt
   ```

   On macOS/Linux, replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and enter your own NVIDIA API key. Never commit the real secrets file.
3. Start the app from the project root:

   ```bash
   .venv/Scripts/python.exe -m streamlit run app.py
   ```

   On macOS/Linux, use `.venv/bin/python -m streamlit run app.py`.

If you need to regenerate the prepared evidence files, run `python prepare_data.py` with the original SEC and transcript export paths. The original large source files are intentionally not needed at app runtime.

## Deploy to Streamlit Community Cloud

1. Create a **private** GitHub repository for this project. The curated transcript evidence is included in `data/`; do not make the repository public unless you have checked the data's sharing rights.
2. From this project folder, initialize Git, review the staged file list, and push the project to your private repository. Do not add `.streamlit/secrets.toml`, notebooks, or original SEC source exports. Keep `.gitignore` in place.
3. Sign in at Streamlit Community Cloud with GitHub and authorize access to the private repository.
4. Choose **Create app**, select the repository and `main` branch, set the app file to `app.py`, and select Python 3.12 in Advanced settings.
5. In the deployment **Secrets** box, paste TOML in this shape, replacing the placeholder with your NVIDIA API key:

   ```toml
   NVIDIA_API_KEY = "YOUR_NVIDIA_API_KEY"
   NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
   ```

6. Deploy, then test company selection, peer comparison, and a question whose answer appears in the SEC/transcript evidence. Check deployment logs if the build or model request fails.

NVIDIA model access, account credits, and free-trial limits are account-specific and may change. Never commit API keys. Verify company filings before relying on generated summaries; this is an educational research tool, not investment advice.

## Data coverage

The app only knows what is present in the committed prepared CSVs. Refresh and regenerate those exports when you want newer filings or transcript analyses; then commit the updated prepared files and let Community Cloud redeploy.
