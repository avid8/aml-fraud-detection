import json
import time
import pytest
from kafka import KafkaProducer, KafkaConsumer
from neo4j import GraphDatabase, basic_auth
from ingestion import RawTransaction, normalize, DeadLetterQueue, InputPipeline, BankAPIClient
from graph import AMLGraph
from rules import RuleEngine, AccountType
from risk_engine import RiskEngine, RiskLevel

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "amlpassword"
KAFKA_BROKERS = "localhost:9092"
TOPIC_CLEAN   = "aml.test.clean"
TOPIC_DLQ     = "aml.test.dlq"

VALID_TX = {
    "transaction_id":   "TX-INT-001",
    "timestamp":        "2025-06-08T22:30:00",
    "amount_rial":      5_000_000,
    "card_pan":         "6037991234567890",
    "account_number":   "ACC-INT-001",
    "national_code":    "1000000001",
    "mobile":           "09121111111",
    "ip_address":       "185.55.224.10",
    "device_fingerprint": "fp-integration-test",
    "channel":          "ONLINE",
    "result":           "SUCCESS",
}

HIGH_RISK_TX = {
    "transaction_id":   "TX-INT-002",
    "timestamp":        "2025-06-08T23:30:00",
    "amount_rial":      600_000_000,
    "card_pan":         "6104337777777777",
    "account_number":   "ACC-INT-002",
    "national_code":    "1000000001",
    "mobile":           "09122222222",
    "ip_address":       "185.55.224.10",
    "device_fingerprint": "fp-integration-test",
    "channel":          "ONLINE",
    "result":           "SUCCESS",
}


# ─────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────
@pytest.fixture(scope="session")
def neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASS))
    yield driver
    driver.close()


@pytest.fixture(scope="session")
def kafka_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    yield producer
    producer.close()


@pytest.fixture(autouse=True)
def cleanup_neo4j(neo4j_driver):
    """قبل از هر تست گراف رو پاک می‌کنه"""
    with neo4j_driver.session() as session:
        session.run("MATCH (n) WHERE n.id STARTS WITH 'TX-INT' OR n.id STARTS WITH 'ACC-INT' OR n.id STARTS WITH 'fp-int' DETACH DELETE n")
    yield
    with neo4j_driver.session() as session:
        session.run("MATCH (n) WHERE n.id STARTS WITH 'TX-INT' OR n.id STARTS WITH 'ACC-INT' DETACH DELETE n")


# ─────────────────────────────────────────────
# تست‌ها
# ─────────────────────────────────────────────
class TestNeo4jConnection:
    def test_neo4j_is_up(self, neo4j_driver):
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 AS n").single()
            assert result["n"] == 1

    def test_create_and_read_node(self, neo4j_driver):
        with neo4j_driver.session() as session:
            session.run("MERGE (n:TestNode {id: 'test-1'})")
            result = session.run("MATCH (n:TestNode {id: 'test-1'}) RETURN n.id AS id").single()
            assert result["id"] == "test-1"
            session.run("MATCH (n:TestNode {id: 'test-1'}) DELETE n")


class TestKafkaConnection:
    def test_kafka_is_up(self, kafka_producer):
        future = kafka_producer.send(TOPIC_CLEAN, {"test": "ping"})
        record = future.get(timeout=10)
        assert record.topic == TOPIC_CLEAN

    def test_produce_and_consume(self, kafka_producer):
        kafka_producer.send(TOPIC_CLEAN, {"msg": "hello"})
        kafka_producer.flush()
        consumer = KafkaConsumer(
            TOPIC_CLEAN,
            bootstrap_servers=KAFKA_BROKERS,
            auto_offset_reset="earliest",
            consumer_timeout_ms=5000,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        messages = [m.value for m in consumer]
        consumer.close()
        assert any(m.get("msg") == "hello" for m in messages)


class TestIngestionToNeo4j:
    def test_normalize_and_store(self, neo4j_driver):
        raw = RawTransaction(**VALID_TX)
        norm = normalize(raw)

        graph = AMLGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS)
        graph.ensure_indexes()
        graph.ingest_normalized_batch([vars(norm)])
        graph.close()

        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (c:Card {id: $id}) RETURN c.id AS id",
                id=norm.card_pan
            ).single()
            assert result is not None
            assert result["id"] == norm.card_pan

    def test_person_connected_to_card(self, neo4j_driver):
        raw = RawTransaction(**VALID_TX)
        norm = normalize(raw)

        graph = AMLGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS)
        graph.ingest_normalized_batch([vars(norm)])
        graph.close()

        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (p:Person)-[:OWNS_CARD]->(c:Card {id: $card})
                RETURN p.id AS pid
            """, card=norm.card_pan).single()
            assert result is not None

    def test_nat_ip_low_confidence(self, neo4j_driver):
        tx = {**VALID_TX, "ip_address": "192.168.1.1",
              "transaction_id": "TX-INT-NAT"}
        raw  = RawTransaction(**tx)
        norm = normalize(raw)
        assert norm.ip_confidence == 0.3

        graph = AMLGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS)
        graph.ingest_normalized_batch([vars(norm)])
        graph.close()

        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (p:Person)-[r:USES_IP]->(ip:IP {id: '192.168.1.1'})
                RETURN r.confidence AS conf
            """).single()
            assert result is not None
            assert result["conf"] == 0.3


class TestRuleEngineIntegration:
    def test_high_amount_triggers_rule(self):
        raw     = RawTransaction(**HIGH_RISK_TX)
        norm    = normalize(raw)
        engine  = RuleEngine()
        features = {
            "acc_total_amount_24h":    0,
            "acc_small_tx_count_24h":  0,
            "ip_unique_cards_1h":      1,
            "dev_unique_cards_1h":     1,
            "acc_fail_ratio_1h":       0.0,
            "card_unique_accounts_24h":1,
        }
        result = engine.evaluate(vars(norm), features, AccountType.NORMAL)
        assert len(result.alerts) > 0
        assert result.should_block is True

    def test_night_tx_detected(self):
        raw     = RawTransaction(**HIGH_RISK_TX)
        norm    = normalize(raw)
        assert norm.is_night is True


class TestFullPipeline:
    def test_end_to_end_clean_tx(self, neo4j_driver, kafka_producer):
        """تراکنش سالم: normalize → graph → rule → risk"""
        raw  = RawTransaction(**VALID_TX)
        norm = normalize(raw)

        # ذخیره در گراف
        graph = AMLGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS)
        graph.ingest_normalized_batch([vars(norm)])
        graph.close()

        # rule engine
        engine   = RuleEngine()
        features = {
            "acc_total_amount_24h": 0, "acc_small_tx_count_24h": 0,
            "ip_unique_cards_1h": 1, "dev_unique_cards_1h": 1,
            "acc_fail_ratio_1h": 0.0, "card_unique_accounts_24h": 1,
        }
        rule_result = engine.evaluate(vars(norm), features, AccountType.NORMAL)

        # risk engine
        risk    = RiskEngine()
        decision = risk.decide(rule_result)

        assert decision.risk_level == RiskLevel.LOW
        assert decision.should_block is False
        assert decision.risk_score < 0.3

        # ارسال به Kafka
        kafka_producer.send(TOPIC_CLEAN, decision.to_dict())
        kafka_producer.flush()

    def test_end_to_end_fraud_tx(self, neo4j_driver, kafka_producer):
        """تراکنش پرریسک: باید block بشه"""
        raw  = RawTransaction(**HIGH_RISK_TX)
        norm = normalize(raw)

        graph = AMLGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS)
        graph.ingest_normalized_batch([vars(norm)])
        graph.close()

        engine   = RuleEngine()
        features = {
            "acc_total_amount_24h": 700_000_000,
            "acc_small_tx_count_24h": 0,
            "ip_unique_cards_1h": 4,
            "dev_unique_cards_1h": 4,
            "acc_fail_ratio_1h": 0.0,
            "card_unique_accounts_24h": 1,
        }
        rule_result = engine.evaluate(vars(norm), features, AccountType.NORMAL)
        risk        = RiskEngine()
        decision    = risk.decide(rule_result)

        assert decision.should_block is True
        assert decision.risk_score > 0.3
        assert len(decision.top_alerts) > 0

        kafka_producer.send(TOPIC_CLEAN, decision.to_dict())
        kafka_producer.flush()
