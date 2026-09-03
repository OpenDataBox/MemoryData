"""
Tests for KnowledgeGraph, Entity, and Relation.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from omni_memory.knowledge.knowledge_graph import KnowledgeGraph, Entity, Relation


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------

class TestEntity:
    def test_default_fields(self):
        ent = Entity(name="Test")
        assert ent.name == "Test"
        assert ent.entity_type == "CONCEPT"
        assert ent.mention_count == 1
        assert ent.aliases == []
        assert ent.attributes == {}

    def test_to_dict_roundtrip(self):
        ent = Entity(name="Alice", entity_type="PERSON", attributes={"role": "engineer"})
        d = ent.to_dict()
        restored = Entity.from_dict(d)
        assert restored.name == "Alice"
        assert restored.entity_type == "PERSON"
        assert restored.attributes["role"] == "engineer"

    def test_merge_combines_sources(self):
        e1 = Entity(name="Alice", source_mau_ids=["m1"])
        e2 = Entity(name="Alice", source_mau_ids=["m2"])
        e1.merge(e2)
        assert "m1" in e1.source_mau_ids
        assert "m2" in e1.source_mau_ids
        assert e1.mention_count == 2

    def test_merge_adds_aliases(self):
        e1 = Entity(name="Bob")
        e2 = Entity(name="Robert", aliases=["Bobby"])
        e1.merge(e2)
        assert "Robert" in e1.aliases
        assert "Bobby" in e1.aliases

    def test_merge_combines_attributes(self):
        e1 = Entity(name="X", attributes={"a": 1})
        e2 = Entity(name="X", attributes={"b": 2})
        e1.merge(e2)
        assert e1.attributes == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Relation dataclass
# ---------------------------------------------------------------------------

class TestRelation:
    def test_default_fields(self):
        rel = Relation(subject_id="s1", predicate="knows", object_id="o1")
        assert rel.subject_id == "s1"
        assert rel.predicate == "knows"
        assert rel.object_id == "o1"
        assert rel.confidence == 1.0

    def test_get_triple(self):
        rel = Relation(subject_id="s1", predicate="works_at", object_id="o1")
        assert rel.get_triple() == ("s1", "works_at", "o1")

    def test_to_dict_roundtrip(self):
        rel = Relation(subject_id="s1", predicate="likes", object_id="o1", confidence=0.9)
        d = rel.to_dict()
        restored = Relation.from_dict(d)
        assert restored.predicate == "likes"
        assert restored.confidence == 0.9


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    @pytest.fixture
    def graph(self, tmp_path):
        return KnowledgeGraph(storage_path=str(tmp_path / "kg"))

    def test_add_and_get_entity(self, graph):
        ent = Entity(name="Alice", entity_type="PERSON")
        eid = graph.add_entity(ent)
        retrieved = graph.get_entity(eid)
        assert retrieved is not None
        assert retrieved.name == "Alice"

    def test_add_duplicate_entity_merges(self, graph):
        e1 = Entity(name="Alice", entity_type="PERSON", source_mau_ids=["m1"])
        e2 = Entity(name="Alice", entity_type="PERSON", source_mau_ids=["m2"])
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)
        assert id1 == id2
        merged = graph.get_entity(id1)
        assert "m1" in merged.source_mau_ids
        assert "m2" in merged.source_mau_ids
        assert merged.mention_count == 2

    def test_get_entity_by_name_case_insensitive(self, graph):
        ent = Entity(name="Alice")
        graph.add_entity(ent)
        assert graph.get_entity_by_name("alice") is not None
        assert graph.get_entity_by_name("ALICE") is not None
        assert graph.get_entity_by_name("nonexistent") is None

    def test_get_entities_by_type(self, graph):
        graph.add_entity(Entity(name="Alice", entity_type="PERSON"))
        graph.add_entity(Entity(name="Bob", entity_type="PERSON"))
        graph.add_entity(Entity(name="Paris", entity_type="LOCATION"))

        persons = graph.get_entities_by_type("PERSON")
        assert len(persons) == 2
        locations = graph.get_entities_by_type("LOCATION")
        assert len(locations) == 1

    def test_get_entities_by_mau(self, graph):
        graph.add_entity(Entity(name="Alice", source_mau_ids=["m1"]))
        graph.add_entity(Entity(name="Bob", source_mau_ids=["m1", "m2"]))
        graph.add_entity(Entity(name="Carol", source_mau_ids=["m2"]))

        m1_entities = graph.get_entities_by_mau("m1")
        names = {e.name for e in m1_entities}
        assert "Alice" in names
        assert "Bob" in names
        assert "Carol" not in names

    def test_add_and_get_relation(self, graph):
        e1 = Entity(name="Alice", entity_type="PERSON")
        e2 = Entity(name="Acme Corp", entity_type="ORGANIZATION")
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)

        rel = Relation(subject_id=id1, predicate="works_at", object_id=id2)
        rid = graph.add_relation(rel)
        retrieved = graph.get_relation(rid)
        assert retrieved is not None
        assert retrieved.predicate == "works_at"

    def test_duplicate_relation_increments_count(self, graph):
        e1 = Entity(name="A")
        e2 = Entity(name="B")
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)

        r1 = Relation(subject_id=id1, predicate="knows", object_id=id2)
        r2 = Relation(subject_id=id1, predicate="knows", object_id=id2)
        rid1 = graph.add_relation(r1)
        rid2 = graph.add_relation(r2)
        assert rid1 == rid2
        rel = graph.get_relation(rid1)
        assert rel.mention_count == 2

    def test_get_relations_for_entity(self, graph):
        e1 = Entity(name="A")
        e2 = Entity(name="B")
        e3 = Entity(name="C")
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)
        id3 = graph.add_entity(e3)

        graph.add_relation(Relation(subject_id=id1, predicate="knows", object_id=id2))
        graph.add_relation(Relation(subject_id=id3, predicate="likes", object_id=id1))

        # Both directions
        rels = graph.get_relations_for_entity(id1, direction="both")
        assert len(rels) == 2

        # Outgoing only
        out = graph.get_relations_for_entity(id1, direction="outgoing")
        assert len(out) == 1
        assert out[0].predicate == "knows"

        # Incoming only
        inc = graph.get_relations_for_entity(id1, direction="incoming")
        assert len(inc) == 1
        assert inc[0].predicate == "likes"

    def test_search_entities_by_name(self, graph):
        graph.add_entity(Entity(name="Albert Einstein", entity_type="PERSON"))
        graph.add_entity(Entity(name="Albert Camus", entity_type="PERSON"))
        graph.add_entity(Entity(name="Marie Curie", entity_type="PERSON"))

        results = graph.search_entities("Albert")
        assert len(results) == 2
        names = {e.name for e in results}
        assert "Albert Einstein" in names
        assert "Albert Camus" in names

    def test_search_entities_by_type_filter(self, graph):
        graph.add_entity(Entity(name="Albert Einstein", entity_type="PERSON"))
        graph.add_entity(Entity(name="Albert Hall", entity_type="LOCATION"))

        results = graph.search_entities("Albert", entity_types=["PERSON"])
        assert len(results) == 1
        assert results[0].entity_type == "PERSON"

    def test_search_entities_by_alias(self, graph):
        ent = Entity(name="Robert", aliases=["Bob", "Bobby"])
        graph.add_entity(ent)
        results = graph.search_entities("Bob")
        assert len(results) == 1
        assert results[0].name == "Robert"

    def test_count(self, graph):
        assert graph.count_entities() == 0
        assert graph.count_relations() == 0
        e1 = Entity(name="A")
        e2 = Entity(name="B")
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)
        assert graph.count_entities() == 2
        graph.add_relation(Relation(subject_id=id1, predicate="rel", object_id=id2))
        assert graph.count_relations() == 1

    def test_get_stats(self, graph):
        graph.add_entity(Entity(name="A", entity_type="PERSON"))
        graph.add_entity(Entity(name="B", entity_type="LOCATION"))
        stats = graph.get_stats()
        assert stats["total_entities"] == 2
        assert "PERSON" in stats["entity_types"]

    def test_clear(self, graph):
        graph.add_entity(Entity(name="X"))
        graph.clear()
        assert graph.count_entities() == 0

    def test_get_neighbors(self, graph):
        e1 = Entity(name="A")
        e2 = Entity(name="B")
        e3 = Entity(name="C")
        id1 = graph.add_entity(e1)
        id2 = graph.add_entity(e2)
        id3 = graph.add_entity(e3)

        graph.add_relation(Relation(subject_id=id1, predicate="knows", object_id=id2))
        graph.add_relation(Relation(subject_id=id2, predicate="knows", object_id=id3))

        neighbors = graph.get_neighbors(id1, max_hops=1)
        assert id2 in neighbors
        assert id3 not in neighbors

        neighbors_2hop = graph.get_neighbors(id1, max_hops=2)
        assert id2 in neighbors_2hop
        assert id3 in neighbors_2hop

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "kg_persist")
        g1 = KnowledgeGraph(storage_path=path)
        e1 = Entity(name="Alice", entity_type="PERSON")
        e2 = Entity(name="Bob", entity_type="PERSON")
        id1 = g1.add_entity(e1)
        id2 = g1.add_entity(e2)
        g1.add_relation(Relation(subject_id=id1, predicate="knows", object_id=id2))
        g1.save()

        # Reload
        g2 = KnowledgeGraph(storage_path=path)
        assert g2.count_entities() == 2
        assert g2.count_relations() == 1
        assert g2.get_entity_by_name("alice") is not None
