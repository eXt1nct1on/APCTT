"""Pydantic schemas for pipeline contracts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Sentence(BaseModel):
    """A sentence extracted from a document."""

    sid: str = Field(..., description="Sentence identifier.")
    text: str = Field(..., description="Sentence text.")


class Chunk(BaseModel):
    """A searchable document chunk."""

    chunk_id: str = Field(..., description="Chunk identifier.")
    page_start: int = Field(..., ge=1, description="First page containing the chunk.")
    page_end: int = Field(..., ge=1, description="Last page containing the chunk.")
    section: str = Field(..., description="Document section name.")
    text: str = Field(..., description="Chunk text.")
    sentences: list[Sentence] = Field(..., description="Sentences in the chunk.")


class Document(BaseModel):
    """An ingested source document."""

    doc_id: str = Field(..., description="Document identifier.")
    filename: str = Field(..., description="Original filename.")
    doc_type: str = Field(..., description="Document type.")
    n_pages: int = Field(..., ge=1, description="Number of pages.")
    sha256: str = Field(..., description="SHA-256 content hash.")
    meta: dict[str, Any] = Field(default_factory=dict, description="Document metadata.")
    chunks: list[Chunk] = Field(..., description="Searchable document chunks.")


class ClaimedResult(BaseModel):
    """A claimed performance result from the source."""

    metric: str = Field(..., description="Metric name.")
    value: str = Field(..., description="Metric value.")
    conditions: str = Field(..., description="Conditions for the result.")


class TechProfile(BaseModel):
    """Technology identity and high-level description."""

    name: str = Field(..., description="Technology name.")
    domain: str = Field(..., description="Technology domain.")
    doc_type: str = Field(..., description="Source document type.")
    summary: str = Field(..., description="Technology summary.")
    intended_application: str = Field(..., description="Intended application.")
    key_components: list[str] = Field(..., description="Key components.")
    claimed_results: list[ClaimedResult] = Field(..., description="Claimed results.")
    author_stated_maturity: str | None = Field(
        ..., description="Maturity stated by the authors, if any."
    )
    keywords: list[str] = Field(..., description="Search keywords.")


class EvidenceItem(BaseModel):
    """A quote linked to a TRL criterion."""

    evidence_id: str = Field(..., description="Evidence identifier.")
    criterion_id: str = Field(..., description="Linked criterion identifier.")
    chunk_id: str = Field(..., description="Source chunk identifier.")
    page: int = Field(..., ge=1, description="Source page.")
    quote: str = Field(..., description="Verbatim evidence quote.")
    polarity: Literal["achieved", "planned", "hypothetical", "background"] = Field(
        ..., description="Evidence polarity."
    )
    attribution: Literal["own", "cited"] = Field(..., description="Evidence attribution.")
    strength: float = Field(..., ge=0, le=1, description="Evidence strength.")
    why: str = Field(..., description="Reason the quote supports the criterion.")
    verified: bool = Field(..., description="Whether the quote was source-verified.")
    match_score: float = Field(..., ge=0, le=1, description="Text matching score.")
    span_start: int = Field(..., ge=0, description="Start offset in source text.")
    span_end: int = Field(..., ge=0, description="End offset in source text.")


class CriterionResult(BaseModel):
    """Assessment result for one rubric criterion."""

    criterion_id: str = Field(..., description="Criterion identifier.")
    status: Literal["met", "partial", "not_met", "unknown"] = Field(
        ..., description="Criterion status."
    )
    evidence_ids: list[str] = Field(..., description="Supporting evidence identifiers.")
    score: float = Field(..., ge=0, le=1, description="Criterion score.")
    note: str = Field(..., description="Assessment note.")


class LevelScore(BaseModel):
    """Assessment score for a TRL level."""

    level: int = Field(..., ge=1, le=9, description="TRL level.")
    score: float = Field(..., ge=0, le=1, description="Level score.")
    criteria: list[CriterionResult] = Field(..., description="Criterion results.")


class TRLVerdict(BaseModel):
    """Final technology readiness verdict."""

    trl: int = Field(..., ge=1, le=9, description="Assigned TRL.")
    range_low: int = Field(..., ge=1, le=9, description="Lower TRL bound.")
    range_high: int = Field(..., ge=1, le=9, description="Upper TRL bound.")
    confidence: float = Field(..., ge=0, le=1, description="Verdict confidence.")
    votes: dict[str, Any] = Field(..., description="Rule and judge votes.")
    reasoning: str = Field(..., description="Verdict reasoning.")
    decisive_evidence_ids: list[str] = Field(..., description="Decisive evidence identifiers.")
    contradictions: list[str] = Field(..., description="Contradictions in the evidence.")
    flags: list[str] = Field(..., description="Verdict flags.")


class GapAction(BaseModel):
    """An action addressing a readiness gap."""

    step: str = Field(..., description="Action step.")
    done_when: str = Field(..., description="Measurable completion condition.")
    effort: Literal["low", "medium", "high"] = Field(..., description="Estimated effort.")
    typical_duration: str = Field(..., description="Typical duration.")
    owner: str = Field(..., description="Responsible owner.")


class GapItem(BaseModel):
    """A missing or partial criterion and its actions."""

    criterion_id: str = Field(..., description="Criterion identifier.")
    missing: str = Field(..., description="Missing capability or evidence.")
    why_it_matters: str = Field(..., description="Importance of closing the gap.")
    actions: list[GapAction] = Field(..., description="Actions to close the gap.")
    priority: int = Field(..., ge=1, le=3, description="Gap priority.")


class GapAnalysis(BaseModel):
    """Roadmap for reaching the target TRL."""

    target_trl: int = Field(..., ge=1, le=9, description="Target TRL.")
    headline: str = Field(..., description="Primary blocker headline.")
    items: list[GapItem] = Field(..., description="Readiness gaps.")
    risks: list[str] = Field(..., description="Risks.")
    assumptions: list[str] = Field(..., description="Assumptions.")
    roadmap: dict[str, list[str]] = Field(..., description="Now, next, and later roadmap.")


class Summary(BaseModel):
    """Executive summary of an assessment."""

    text: str = Field(..., description="Summary text.")
    key_takeaways: list[str] = Field(..., description="Key takeaways.")
    caveats: list[str] = Field(..., description="Caveats.")


class RunStats(BaseModel):
    """Execution statistics for an assessment run."""

    llm_calls: int = Field(..., ge=0, description="Number of LLM calls.")
    cache_hits: int = Field(..., ge=0, description="Number of cache hits.")
    tokens_in: int = Field(..., ge=0, description="Input token count.")
    tokens_out: int = Field(..., ge=0, description="Output token count.")
    seconds: float = Field(..., ge=0, description="Run duration in seconds.")


class ProgressEvent(BaseModel):
    """Progress update emitted during a run."""

    step: str = Field(..., description="Pipeline step.")
    status: str = Field(..., description="Step status.")
    detail: str = Field(..., description="Progress detail.")
    pct: float = Field(..., ge=0, le=100, description="Completion percentage.")


class Assessment(BaseModel):
    """Complete assessment data contract shared by pipeline and UI."""

    run_id: str = Field(..., description="Assessment run identifier.")
    created_at: datetime = Field(..., description="Assessment creation timestamp.")
    is_example: bool = Field(..., description="Whether this is a hand-written example.")
    source: str = Field(..., description="Source document reference.")
    tech: TechProfile = Field(..., description="Technology profile.")
    verdict: TRLVerdict = Field(..., description="TRL verdict.")
    levels: list[LevelScore] = Field(..., description="Scores for assessed levels.")
    evidence: list[EvidenceItem] = Field(..., description="Evidence items.")
    gaps: GapAnalysis = Field(..., description="Gap analysis.")
    summary: Summary = Field(..., description="Executive summary.")
    author_stated_trl_mentions: list[str] = Field(
        ..., description="TRL mentions made by the authors."
    )
    planned_items: list[str] = Field(..., description="Planned work items.")
    flags: list[str] = Field(..., description="Assessment flags.")
    stats: RunStats = Field(..., description="Run statistics.")
