import pytest
from unittest.mock import MagicMock, patch, call
from graph import AMLGraph, GraphConfig

SAMPLE_TX = {
    "transaction_id":   "TX-001",
    "timestamp_utc":    "2025-06-08T20:00:00Z",
    "amount_rial":      5_000_000,
    "card_pan":         "6037991234567890",
    "account_number":   "ACC-001",
    "national_code":    "1000000001",
    "mobile_normalized":"09121111111",
    "ip_address":       "185.1.1.1",
    "device_fp_hash":   "fp-abc-hash",
    "ip_confidence":    1.0,
}

NAT_TX = {**SAMPLE_TX, "transaction_id": "TX-002",
          "ip_address": "192.168.1.1", "ip_confidence": 0.3}


@pytest.fixture
def mock_graph():
    with patch("graph.GraphDatabase") as mock_db:
        mock_session = MagicMock()
        mock_db.driver.return_value.session.return_value.__enter__ = lambda s, *a: mock_session
        mock_db.driver.return_value.session.return_value.__exit__ = MagicMock(return_value=False)
        graph = AMLGraph()
        graph.driver = mock_db.driver.return_value
        yield graph, mock_session


class TestMergeNodes:
    def test_merge_nodes_batch_groups_by_label(self, mock_graph):
        graph, session = mock_graph
        nodes = [("Card", "c1"), ("Card", "c2"), ("Account", "a1")]
        graph.merge_nodes_batch(session, nodes)
        calls = [str(c) for c in session.run.call_args_list]
        assert any("Card" in c for c in calls)
        assert any("Account" in c for c in calls)

    def test_merge_nodes_batch_empty(self, mock_graph):
        graph, session = mock_graph
        graph.merge_nodes_batch(session, [])
        session.run.assert_not_called()


class TestIngestBatch:
    def test_ingest_creates_person(self, mock_graph):
        graph, session = mock_graph
        session.execute_write = MagicMock()
        graph.ingest_normalized_batch([SAMPLE_TX])
        person_calls = [c for c in session.run.call_args_list
                        if "Person" in str(c) and "MERGE" in str(c)]
        assert len(person_calls) >= 1

    def test_ingest_nat_ip_low_confidence(self, mock_graph):
        graph, session = mock_graph
        edges_captured = []

        def capture_write(fn, rel_type, from_label, to_label, edges):
            if rel_type == "USES_IP":
                edges_captured.extend(edges)

        session.execute_write.side_effect = capture_write
        graph.ingest_normalized_batch([NAT_TX])
        assert any(e["confidence"] == 0.3 for e in edges_captured)

    def test_ingest_full_confidence_for_public_ip(self, mock_graph):
        graph, session = mock_graph
        edges_captured = []

        def capture_write(fn, rel_type, from_label, to_label, edges):
            if rel_type == "USES_IP":
                edges_captured.extend(edges)

        session.execute_write.side_effect = capture_write
        graph.ingest_normalized_batch([SAMPLE_TX])
        assert all(e["confidence"] == 1.0 for e in edges_captured)

    def test_ingest_all_six_rel_types(self, mock_graph):
        graph, session = mock_graph
        rel_types_seen = []

        def capture_write(fn, rel_type, *args, **kwargs):
            rel_types_seen.append(rel_type)

        session.execute_write.side_effect = capture_write
        graph.ingest_normalized_batch([SAMPLE_TX])
        expected = {"OWNS_CARD", "OWNS_ACCOUNT", "HAS_MOBILE",
                    "USES_IP", "USES_DEVICE", "IDENTIFIED_BY"}
        assert expected == set(rel_types_seen)


class TestGetIdentitySubgraph:
    def test_returns_none_for_missing_person(self, mock_graph):
        graph, session = mock_graph
        session.run.return_value.single.return_value = None
        result = graph.get_identity_subgraph("unknown-person")
        assert result is None

    def test_returns_dict_for_existing_person(self, mock_graph):
        graph, session = mock_graph
        session.run.return_value.single.return_value = {
            "person_id": "p1",
            "entities": [{"type": "Card", "id": "6037...", "confidence": 1.0}]
        }
        result = graph.get_identity_subgraph("p1")
        assert result["person_id"] == "p1"
        assert len(result["entities"]) == 1


class TestGraphConfig:
    def test_nat_confidence_lower_than_full(self):
        assert GraphConfig.NAT_CONFIDENCE < GraphConfig.FULL_CONFIDENCE

    def test_cleanup_days_positive(self):
        assert GraphConfig.CLEANUP_DAYS > 0
