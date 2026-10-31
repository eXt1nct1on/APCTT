# Deployment

## Hugging Face Spaces (Docker SDK)

1. Create a Space and select **Docker** as the SDK.
2. Push this repository, including `Dockerfile`, to the Space.
3. In **Settings > Variables and secrets**, add `LLM_MODEL` and one provider
   key such as `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or
   `OPENAI_API_KEY`. Never commit `.env` or keys.
4. Optionally set `LLM_FALLBACK_MODEL`, `EMBED_MODEL`, and `DATA_DIR`.
5. The container listens on port `7860`. Without a provider key, the UI
   intentionally opens in demo mode using `docs/example_assessment.json`.

The Space health check uses Streamlit's built-in `/_stcore/health` endpoint.

## Render or Railway

Create a Web Service from the repository, choose Docker deployment, and
provide the same environment variables in the platform's secret manager.
Use the platform-provided `PORT` value if it differs from 7860:

```text
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

For an API-only service, override the container command:

```text
uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

The API exposes `GET /healthz`, while the UI container exposes
`GET /_stcore/health`.

## Local Docker

```text
docker build -t trl-scout .
docker run --rm -p 7860:7860 trl-scout
```

Open `http://localhost:7860`. To run the API instead:

```text
docker run --rm -p 8000:8000 trl-scout uvicorn api.main:app --host 0.0.0.0 --port 8000
```
