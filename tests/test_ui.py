"""Static tests for the Streamlit application contract."""

from pathlib import Path

from app.streamlit_app import _has_provider_key


def test_streamlit_app_contains_required_surfaces() -> None:
    """The UI source exposes demo mode, progress, evidence, gaps, and audit views."""
    source = (Path(__file__).parents[1] / "app" / "streamlit_app.py").read_text(
        encoding="utf-8"
    )
    for text in (
        "Demo mode",
        "st.status",
        "LLM calls used",
        "Evidence",
        "Gaps and roadmap",
        "Audit trail",
        "show_rejected",
        "st.download_button",
    ):
        assert text in source


def test_demo_fallback_when_no_provider_key(monkeypatch) -> None:
    """Deployments without a hosted provider key can start safely in demo mode."""
    monkeypatch.setattr("app.streamlit_app.settings.gemini_api_key", None)
    monkeypatch.setattr("app.streamlit_app.settings.groq_api_key", None)
    monkeypatch.setattr("app.streamlit_app.settings.openrouter_api_key", None)
    monkeypatch.setattr("app.streamlit_app.settings.openai_api_key", None)
    assert _has_provider_key() is False


def test_deployment_files_expose_ui_and_api_commands() -> None:
    """Container documentation covers the hosted UI and API alternatives."""
    root = Path(__file__).parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    deploy = (root / "docs" / "deploy.md").read_text(encoding="utf-8")
    assert "streamlit" in dockerfile
    assert "7860" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "uvicorn api.main:app" in deploy
