"""Streamlit user interface for TRL-Scout assessments."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from trl_scout.config import settings
from trl_scout.pipeline.run import assess
from trl_scout.report.builder import render_html
from trl_scout.schemas import Assessment, ProgressEvent

ROOT = Path(__file__).parents[1]
EXAMPLE_PATH = ROOT / "docs" / "example_assessment.json"
POLARITY_COLOURS = {
    "achieved": "#1d7a5a",
    "planned": "#b7791f",
    "hypothetical": "#7651a9",
    "background": "#647779",
}

st.set_page_config(
    page_title="TRL-Scout",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _style() -> None:
    """Apply projector-friendly layout and colour styling."""
    st.markdown(
        """
        <style>
        .block-container { max-width: 1500px; padding-top: 2rem; }
        h1 { font-size: 2.8rem !important; color: #123e43; }
        h2 { color: #123e43; }
        .metric-card { background: #eef6f5; border-left: 5px solid #1f6f78;
                       border-radius: 8px; padding: 1rem 1.2rem; min-height: 115px; }
        .metric-value { color: #1f6f78; font-size: 2.8rem; font-weight: 700; }
        .metric-label { color: #526668; font-size: 1rem; }
        .quote { border-left: 4px solid #1f6f78; background: #f5f8f8;
                 padding: .8rem 1rem; margin: .4rem 0 1rem; font-size: 1.05rem; }
        .badge { border-radius: 12px; color: white; display: inline-block;
                 font-size: .78rem; font-weight: 700; margin-right: .4rem;
                 padding: .15rem .55rem; }
        .gap-card { background: #f5f8f8; border-top: 4px solid #1f6f78;
                    border-radius: 6px; padding: .9rem; margin-bottom: .8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_demo() -> Assessment:
    """Load and validate the bundled example assessment."""
    return Assessment.model_validate_json(EXAMPLE_PATH.read_text(encoding="utf-8"))


def _set_result(assessment: Assessment, html: str | None = None) -> None:
    """Persist the result and generated report in session state."""
    st.session_state.assessment = assessment
    st.session_state.report_html = html or render_html(assessment)
    st.session_state.run_id = assessment.run_id


def _source_controls() -> tuple[str | Path | None, str | None]:
    """Render source controls and return the selected source."""
    samples = ["None"] + [
        str(path.relative_to(ROOT))
        for folder in (ROOT / "eval" / "gold" / "docs", ROOT / "tests" / "fixtures")
        if folder.exists()
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
    ]
    sample = st.selectbox("Bundled sample document", samples)
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown",
        type=["pdf", "docx", "txt", "md"],
    )
    arxiv_id = st.text_input("arXiv ID", placeholder="e.g. 2401.12345")
    url = st.text_input("Document URL", placeholder="https://...")
    text = st.text_area("Paste document text", height=150)
    selected = [value for value in (uploaded, arxiv_id.strip(), url.strip(), text.strip()) if value]
    if sample != "None":
        return ROOT / sample, "sample"
    if len(selected) > 1:
        st.warning("Choose only one source.")
        return None, None
    if uploaded is not None:
        return uploaded, "upload"
    if arxiv_id.strip():
        return arxiv_id.strip(), "arxiv"
    if url.strip():
        return url.strip(), "url"
    if text.strip():
        return text.strip(), "text"
    return None, None


def _run_real(source: Any, source_kind: str) -> None:
    """Run the pipeline once and keep progress visible."""
    status = st.status("Assessment in progress", expanded=True)
    progress_slot = status.empty()
    progress_events: list[ProgressEvent] = []

    def progress(event: ProgressEvent) -> None:
        """Render progress events received from the in-process pipeline."""
        progress_events.append(event)
        progress_slot.write(f"**{event.step}** · {event.status} — {event.detail}")
        status.update(label=f"{event.step}: {event.detail}", state="running")

    try:
        if source_kind == "upload":
            suffix = Path(source.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(source.getvalue())
                source_path = Path(handle.name)
            source = source_path
        result = assess(
            source,
            settings=settings,
            progress=progress,
        )
        _set_result(result)
        status.update(label="Assessment complete", state="complete", expanded=False)
        st.toast(f"Assessment complete — {result.stats.llm_calls} LLM calls used.")
    except Exception:
        status.update(label="Assessment could not be completed", state="error")
        st.error(
            "We could not process this source. Check the file type or URL and try again. "
            "Details are available in the run audit log."
        )


def _gauge(assessment: Assessment) -> go.Figure:
    """Build the large TRL gauge."""
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=assessment.verdict.trl,
            number={"font": {"size": 58, "color": "#1f6f78"}, "suffix": " / 9"},
            title={"text": "Technology Readiness Level", "font": {"size": 20}},
            gauge={
                "axis": {"range": [1, 9], "dtick": 1},
                "bar": {"color": "#1f6f78"},
                "steps": [
                    {"range": [1, 3], "color": "#e6f0ef"},
                    {"range": [3, 6], "color": "#cfe3e1"},
                    {"range": [6, 9], "color": "#b8d7d4"},
                ],
                "threshold": {
                    "line": {"color": "#123e43", "width": 5},
                    "thickness": 0.8,
                    "value": assessment.verdict.trl,
                },
            },
        )
    ).update_layout(height=280, margin={"t": 45, "b": 10, "l": 35, "r": 35})


def _badge(value: str) -> str:
    """Render a consistent polarity or attribution badge."""
    colour = POLARITY_COLOURS.get(value, "#647779")
    return f'<span class="badge" style="background:{colour}">{value}</span>'


def _show_result(assessment: Assessment) -> None:
    """Render the result header and all result tabs."""
    st.header("Assessment result")
    left, middle, right = st.columns([1.2, 1, 1.5])
    with left:
        st.plotly_chart(
            _gauge(assessment),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with middle:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{assessment.verdict.range_low}–{assessment.verdict.range_high}'
            f'</div><div class="metric-label">Confidence range</div></div>',
            unsafe_allow_html=True,
        )
        label = assessment.verdict.votes.get("confidence_label", "Low")
        st.metric("Confidence", f"{assessment.verdict.confidence:.0%}", label)
    with right:
        st.subheader(assessment.tech.name)
        st.write(assessment.tech.summary)
        st.caption(f"{assessment.tech.domain} · {assessment.tech.doc_type}")

    tabs = st.tabs(["Evidence", "Gaps and roadmap", "Report", "Audit trail"])
    _show_evidence(tabs[0], assessment)
    _show_gaps(tabs[1], assessment)
    _show_report(tabs[2], assessment)
    _show_audit(tabs[3], assessment)


def _show_evidence(tab: Any, assessment: Assessment) -> None:
    """Render filtered evidence grouped by TRL level."""
    with tab:
        levels = ["All"] + [str(level.level) for level in assessment.levels]
        level_filter = st.selectbox("Level", levels, key="evidence_level")
        polarity_filter = st.multiselect(
            "Polarity",
            list(POLARITY_COLOURS),
            default=list(POLARITY_COLOURS),
            key="evidence_polarity",
        )
        show_rejected = st.toggle("Show rejected quotes for transparency", key="show_rejected")
        selected = [
            item
            for item in assessment.evidence
            if (level_filter == "All" or item.criterion_id.startswith(f"T{level_filter}."))
            and item.polarity in polarity_filter
            and (show_rejected or item.verified)
        ]
        if not selected:
            st.info("No evidence matches these filters.")
        for item in selected:
            tick = "✓" if item.verified else "✗"
            st.markdown(
                f"**{item.evidence_id} · {item.criterion_id} · page {item.page}** "
                f"{_badge(item.polarity)} {_badge(item.attribution)} "
                f'<span class="{"verified" if item.verified else "unverified"}">{tick}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="quote">“{item.quote}”</div>', unsafe_allow_html=True)
            st.caption(item.why)


def _show_gaps(tab: Any, assessment: Assessment) -> None:
    """Render gap cards and the now/next/later roadmap."""
    with tab:
        st.subheader(f"Next target: TRL {assessment.gaps.target_trl}")
        st.write(assessment.gaps.headline)
        columns = st.columns(3)
        for index, phase in enumerate(("now", "next", "later")):
            with columns[index]:
                st.markdown(f"### {phase.capitalize()}")
                for action in assessment.gaps.roadmap.get(phase, []):
                    st.markdown(f"- {action}")
        st.divider()
        for item in assessment.gaps.items:
            st.markdown(
                f'<div class="gap-card"><strong>{item.criterion_id}</strong> · '
                f'Priority {item.priority}<br>{item.missing}<br>'
                f'<span class="muted">{item.why_it_matters}</span></div>',
                unsafe_allow_html=True,
            )
            for action in item.actions:
                st.markdown(
                    f"**Done when:** {action.done_when}  \n"
                    f"`{action.effort}` · {action.typical_duration} · {action.owner}"
                )


def _show_report(tab: Any, assessment: Assessment) -> None:
    """Render and download the self-contained report artifacts."""
    with tab:
        report_html = st.session_state.get("report_html") or render_html(assessment)
        components.html(report_html, height=800, scrolling=True)
        columns = st.columns(3)
        with columns[0]:
            st.download_button("Download HTML", report_html, "trl-scout-report.html", "text/html")
        with columns[1]:
            pdf_path = Path(settings.data_dir) / "runs" / assessment.run_id / "report.pdf"
            pdf_data = pdf_path.read_bytes() if pdf_path.exists() else None
            st.download_button(
                "Download PDF",
                pdf_data,
                "trl-scout-report.pdf",
                "application/pdf",
                disabled=pdf_data is None,
            )
        with columns[2]:
            st.download_button(
                "Download JSON",
                assessment.model_dump_json(indent=2),
                "assessment.json",
                "application/json",
            )


def _show_audit(tab: Any, assessment: Assessment) -> None:
    """Render flags, statistics, and per-step audit JSON."""
    with tab:
        st.json({"flags": assessment.flags, "stats": assessment.stats.model_dump(mode="json")})
        run_dir = Path(settings.data_dir) / "runs" / assessment.run_id
        if run_dir.exists():
            for path in sorted(run_dir.glob("*.json")):
                with st.expander(path.stem):
                    try:
                        st.json(json.loads(path.read_text(encoding="utf-8")))
                    except (OSError, json.JSONDecodeError):
                        st.warning(f"Could not read {path.name}.")


def main() -> None:
    """Render the Streamlit application."""
    _style()
    st.title("TRL-Scout")
    st.caption("Evidence-grounded technology readiness assessment")
    with st.sidebar:
        st.header("Source")
        demo_mode = st.toggle("Demo mode", value=False)
        if demo_mode:
            st.info("Using the bundled battery example. No LLM calls are made.")
            if "assessment" not in st.session_state or not st.session_state.assessment.is_example:
                try:
                    _set_result(_load_demo())
                except Exception:
                    st.error("The bundled example assessment is invalid.")
        else:
            source, source_kind = _source_controls()
            if st.button("Assess document", type="primary", disabled=source is None):
                st.session_state.pop("assessment", None)
                st.session_state.pop("report_html", None)
                _run_real(source, source_kind or "text")
        if st.session_state.get("assessment") is not None:
            st.success("Assessment loaded")
            st.caption(f"Run ID: {st.session_state.assessment.run_id}")

    assessment = st.session_state.get("assessment")
    if assessment is None:
        st.info("Choose Demo mode or provide a document source in the sidebar.")
        return
    _show_result(assessment)


if __name__ == "__main__":
    main()
