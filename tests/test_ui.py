"""Static tests for the Streamlit application contract."""

from pathlib import Path


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
