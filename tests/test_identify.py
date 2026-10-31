"""Tests for technology identification."""

import pytest

from trl_scout.llm.cache import JsonCache
from trl_scout.llm.client import FakeLLM, LLMClient
from trl_scout.pipeline.identify import identify_technology
from trl_scout.schemas import Chunk, Document, Sentence


def _document() -> Document:
    """Build a section-labelled document for R1 selection."""
    chunks = []
    for index, (section, text) in enumerate(
        [
            ("abstract", "A sodium battery stores energy for grid applications."),
            ("introduction", "The architecture uses a sodium-ion cathode and electrolyte."),
            ("results", "The prototype delivered measured capacity in a laboratory."),
            ("conclusion", "The authors conclude that the cell is promising for storage."),
        ],
        1,
    ):
        chunks.append(
            Chunk(
                chunk_id=f"c{index:03d}",
                page_start=index,
                page_end=index,
                section=section,
                text=text,
                sentences=[Sentence(sid=f"s{index:04d}", text=text)],
            )
        )
    return Document(
        doc_id="identify-test",
        filename="battery.txt",
        doc_type="paper",
        n_pages=4,
        sha256="identify-test-sha",
        meta={"title": "Sodium battery"},
        chunks=chunks,
    )


def test_identify_maps_response_and_uses_heuristic_on_low_confidence(tmp_path) -> None:
    """The public profile is mapped and low-confidence type falls back to ingestion."""
    fake = FakeLLM(
        {
            "IdentificationResponse": {
                "doc_type": "other",
                "doc_type_confidence": 0.2,
                "technology_name": "Sodium battery",
                "domain": "energy storage",
                "one_line_summary": "A sodium-ion cell for grid storage.",
                "intended_application": "Grid storage",
                "key_components": ["cathode", "electrolyte"],
                "claimed_results": [
                    {"metric": "capacity", "value": "100 mAh/g", "conditions": "laboratory"}
                ],
                "author_stated_maturity": None,
                "search_keywords": ["sodium battery", "grid storage"],
            }
        }
    )
    profile = identify_technology(
        _document(),
        LLMClient(completion=fake, cache=JsonCache(tmp_path)),
    )
    assert profile.name == "Sodium battery"
    assert profile.doc_type == "research_paper"
    assert fake.calls == 1


@pytest.mark.live
@pytest.mark.skip(reason="live provider test is opt-in")
def test_identify_live() -> None:
    """Run identification against the configured provider when explicitly enabled."""
    from trl_scout.ingest.loaders import load_document
    from trl_scout.llm.client import LLMClient

    profile = identify_technology(
        load_document("A battery converts stored chemical energy into electrical energy."),
        LLMClient(),
    )
    assert profile.name
