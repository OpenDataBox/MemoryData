"""
Tests for MultimodalAtomicUnit and related data structures.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from omni_memory.core.mau import (
    MultimodalAtomicUnit,
    ModalityType,
    MAUMetadata,
    MAULinks,
    QualityMetrics,
)


# ---------------------------------------------------------------------------
# ModalityType enum
# ---------------------------------------------------------------------------

class TestModalityType:
    def test_enum_values(self):
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.VISUAL.value == "visual"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"
        assert ModalityType.MULTIMODAL.value == "multimodal"

    def test_enum_from_value(self):
        assert ModalityType("text") is ModalityType.TEXT
        assert ModalityType("visual") is ModalityType.VISUAL

    def test_enum_is_str(self):
        # ModalityType inherits from str
        assert isinstance(ModalityType.TEXT, str)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ModalityType("invalid_modality")


# ---------------------------------------------------------------------------
# QualityMetrics
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    def test_defaults(self):
        qm = QualityMetrics()
        assert qm.trigger_score == 0.0
        assert qm.confidence == 0.0
        assert qm.entropy_delta == 0.0

    def test_to_dict(self):
        qm = QualityMetrics(trigger_score=0.8, confidence=0.9, entropy_delta=0.1)
        d = qm.to_dict()
        assert d["trigger_score"] == 0.8
        assert d["confidence"] == 0.9
        assert d["entropy_delta"] == 0.1

    def test_from_dict_roundtrip(self):
        original = QualityMetrics(trigger_score=0.5, confidence=0.7, entropy_delta=0.3)
        reconstructed = QualityMetrics.from_dict(original.to_dict())
        assert reconstructed.trigger_score == original.trigger_score
        assert reconstructed.confidence == original.confidence
        assert reconstructed.entropy_delta == original.entropy_delta


# ---------------------------------------------------------------------------
# MAUMetadata
# ---------------------------------------------------------------------------

class TestMAUMetadata:
    def test_defaults(self):
        meta = MAUMetadata()
        assert meta.session_id is None
        assert meta.source is None
        assert meta.tags == []
        assert isinstance(meta.quality, QualityMetrics)

    def test_with_fields(self):
        meta = MAUMetadata(
            session_id="sess1",
            source="user_input",
            tags=["important"],
            persons=["Alice"],
            keywords=["meeting", "project"],
            location="office",
            topic="work",
        )
        assert meta.session_id == "sess1"
        assert meta.persons == ["Alice"]
        assert meta.keywords == ["meeting", "project"]

    def test_to_dict_omits_none(self):
        meta = MAUMetadata(session_id="sess1")
        d = meta.to_dict()
        assert "session_id" in d
        # None fields should be omitted
        assert "source" not in d
        assert "persons" not in d

    def test_from_dict_roundtrip(self):
        meta = MAUMetadata(
            session_id="s1",
            source="mic",
            tags=["tag1"],
            quality=QualityMetrics(trigger_score=0.5),
            speaker_id="spk1",
        )
        d = meta.to_dict()
        restored = MAUMetadata.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.source == "mic"
        assert restored.tags == ["tag1"]
        assert restored.quality.trigger_score == 0.5
        assert restored.speaker_id == "spk1"


# ---------------------------------------------------------------------------
# MAULinks
# ---------------------------------------------------------------------------

class TestMAULinks:
    def test_defaults(self):
        links = MAULinks()
        assert links.event_id is None
        assert links.prev_mau_id is None
        assert links.next_mau_id is None
        assert links.related_mau_ids == []

    def test_to_dict(self):
        links = MAULinks(event_id="evt1", prev_mau_id="mau_a", next_mau_id="mau_b")
        d = links.to_dict()
        assert d["event_id"] == "evt1"
        assert d["prev"] == "mau_a"
        assert d["next"] == "mau_b"

    def test_from_dict_roundtrip(self):
        links = MAULinks(
            event_id="evt1",
            prev_mau_id="p1",
            next_mau_id="n1",
            related_mau_ids=["r1", "r2"],
        )
        d = links.to_dict()
        restored = MAULinks.from_dict(d)
        assert restored.event_id == "evt1"
        assert restored.prev_mau_id == "p1"
        assert restored.next_mau_id == "n1"
        assert restored.related_mau_ids == ["r1", "r2"]


# ---------------------------------------------------------------------------
# MultimodalAtomicUnit
# ---------------------------------------------------------------------------

class TestMAU:
    def test_default_creation(self):
        mau = MultimodalAtomicUnit()
        assert mau.id.startswith("mau_")
        assert mau.modality_type == ModalityType.TEXT
        assert mau.summary == ""
        assert mau.embedding is None
        assert mau.status == "ACTIVE"
        assert mau.storage_tier == "HOT"
        assert isinstance(mau.metadata, MAUMetadata)
        assert isinstance(mau.links, MAULinks)

    def test_creation_with_all_fields(self):
        mau = MultimodalAtomicUnit(
            id="mau_test_001",
            timestamp=1000.0,
            modality_type=ModalityType.VISUAL,
            summary="A cat on a mat",
            embedding=[0.1, 0.2, 0.3],
            raw_pointer="/cold/img_001.jpg",
            details={"caption": "orange tabby cat"},
            status="PINNED",
            storage_tier="COLD",
        )
        assert mau.id == "mau_test_001"
        assert mau.timestamp == 1000.0
        assert mau.modality_type == ModalityType.VISUAL
        assert mau.summary == "A cat on a mat"
        assert mau.embedding == [0.1, 0.2, 0.3]
        assert mau.raw_pointer == "/cold/img_001.jpg"
        assert mau.details == {"caption": "orange tabby cat"}
        assert mau.status == "PINNED"
        assert mau.storage_tier == "COLD"

    def test_unique_ids(self):
        mau1 = MultimodalAtomicUnit()
        mau2 = MultimodalAtomicUnit()
        assert mau1.id != mau2.id

    def test_to_dict(self):
        mau = MultimodalAtomicUnit(
            id="mau_test",
            modality_type=ModalityType.AUDIO,
            summary="hello world",
        )
        d = mau.to_dict()
        assert d["id"] == "mau_test"
        assert d["modality_type"] == "audio"
        assert d["summary"] == "hello world"
        assert "metadata" in d
        assert "links" in d

    def test_from_dict_roundtrip(self):
        original = MultimodalAtomicUnit(
            id="mau_rt",
            modality_type=ModalityType.VIDEO,
            summary="a video clip",
            embedding=[0.1, 0.2],
            status="ARCHIVED",
            storage_tier="COLD",
        )
        d = original.to_dict()
        restored = MultimodalAtomicUnit.from_dict(d)
        assert restored.id == original.id
        assert restored.modality_type == original.modality_type
        assert restored.summary == original.summary
        assert restored.embedding == original.embedding
        assert restored.status == original.status
        assert restored.storage_tier == original.storage_tier

    def test_json_roundtrip(self):
        original = MultimodalAtomicUnit(
            id="mau_json",
            summary="json test",
            modality_type=ModalityType.MULTIMODAL,
        )
        json_str = original.to_json()
        restored = MultimodalAtomicUnit.from_json(json_str)
        assert restored.id == original.id
        assert restored.modality_type == ModalityType.MULTIMODAL

    def test_get_lightweight_dict(self):
        mau = MultimodalAtomicUnit(
            id="mau_lw",
            summary="lightweight",
            details={"heavy": "data"},
        )
        lw = mau.get_lightweight_dict()
        # Lightweight dict should NOT include details
        assert "details" not in lw
        assert lw["id"] == "mau_lw"
        assert lw["summary"] == "lightweight"

    def test_status_transitions(self):
        mau = MultimodalAtomicUnit()
        assert mau.status == "ACTIVE"
        mau.status = "ARCHIVED"
        assert mau.status == "ARCHIVED"
        mau.status = "PINNED"
        assert mau.status == "PINNED"

    def test_storage_tier(self):
        mau = MultimodalAtomicUnit()
        assert mau.storage_tier == "HOT"
        mau.storage_tier = "COLD"
        assert mau.storage_tier == "COLD"

    def test_has_details(self):
        mau = MultimodalAtomicUnit()
        assert mau.has_details() is False
        mau.details = {"key": "value"}
        assert mau.has_details() is True

    def test_clear_details(self):
        mau = MultimodalAtomicUnit(details={"key": "value"})
        assert mau.has_details() is True
        mau.clear_details()
        assert mau.has_details() is False

    def test_set_event(self):
        mau = MultimodalAtomicUnit()
        mau.set_event("evt_123")
        assert mau.links.event_id == "evt_123"

    def test_link_previous_next(self):
        mau = MultimodalAtomicUnit()
        mau.link_previous("mau_prev")
        mau.link_next("mau_next")
        assert mau.links.prev_mau_id == "mau_prev"
        assert mau.links.next_mau_id == "mau_next"

    def test_add_related_no_duplicates(self):
        mau = MultimodalAtomicUnit()
        mau.add_related("mau_r1")
        mau.add_related("mau_r1")
        mau.add_related("mau_r2")
        assert mau.links.related_mau_ids == ["mau_r1", "mau_r2"]

    def test_add_tag_no_duplicates(self):
        mau = MultimodalAtomicUnit()
        mau.add_tag("important")
        mau.add_tag("important")
        mau.add_tag("urgent")
        assert mau.metadata.tags == ["important", "urgent"]
