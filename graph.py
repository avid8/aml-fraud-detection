"""
AML Pipeline — لایه گراف هویت
دیتابیس: Neo4j
مسئولیت‌ها:
  1. ساخت و merge نودها (Person, Card, Account, IP, Device, Mobile, NationalCode)
  2. ساخت یال‌ها با confidence و timestamp
  3. cleanup نودهای قدیمی
  4. دریافت subgraph هویت برای هر تراکنش
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from neo4j import GraphDatabase, basic_auth

logger = logging.getLogger(__name__)


class GraphConfig:
    URI      = "bolt://localhost:7687"
    USER     = "neo4j"
    PASSWORD = "password"
    POOL     = 50
    NAT_CONFIDENCE  = 0.3
    FULL_CONFIDENCE = 1.0
    CLEANUP_DAYS    = 30


class AMLGraph:
    def __init__(self, uri=GraphConfig.URI, user=GraphConfig.USER, password=GraphConfig.PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password),
                                           max_connection_pool_size=GraphConfig.POOL)

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ─────────────────────────────────────────────
    # 1. ساخت نودها
    # ─────────────────────────────────────────────
    def merge_nodes_batch(self, session, nodes: list[tuple[str, str]]):
        """
        nodes = [("Card", "6037..."), ("Account", "ACC-001"), ...]
        هر نوع رو جداگانه batch می‌کنیم
        """
        by_label: dict[str, list[str]] = {}
        for label, nid in nodes:
            by_label.setdefault(label, []).append(nid)

        for label, ids in by_label.items():
            session.run(
                f"UNWIND $ids AS id MERGE (:{label} {{id: id}})",
                ids=ids
            )

    # ─────────────────────────────────────────────
    # 2. ساخت یال‌ها
    # ─────────────────────────────────────────────
    def _merge_edges_for_rel(self, tx, rel_type: str, from_label: str,
                              to_label: str, edges: list[dict]):
        """
        برای هر rel_type یه query جداگانه — چون Neo4j label پارامتری نمیپذیره
        """
        query = f"""
        UNWIND $edges AS e
        MATCH (a:{from_label} {{id: e.from_id}})
        MATCH (b:{to_label} {{id: e.to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.timestamp_start  = coalesce(r.timestamp_start, e.timestamp),
            r.timestamp_end    = datetime('9999-12-31T23:59:59Z'),
            r.confidence       = e.confidence,
            r.total_amount     = coalesce(r.total_amount, 0) + e.amount,
            r.tx_count         = coalesce(r.tx_count, 0) + 1,
            r.last_timestamp   = e.timestamp
        """
        tx.run(query, edges=edges)

    def ingest_normalized_batch(self, transactions: list[dict]):
        """
        ورودی: لیست NormalizedTransaction به صورت dict
        خروجی: نودها و یال‌ها توی گراف
        """
        with self.driver.session() as session:
            # جمع‌آوری همه نودها
            nodes: set[tuple[str, str]] = set()
            for tx in transactions:
                nodes.add(("Card",         tx["card_pan"]))
                nodes.add(("Account",      tx["account_number"]))
                nodes.add(("Mobile",       tx["mobile_normalized"]))
                nodes.add(("IP",           tx["ip_address"]))
                nodes.add(("Device",       tx["device_fp_hash"]))
                nodes.add(("NationalCode", tx["national_code"]))

            self.merge_nodes_batch(session, list(nodes))

            # ساخت Person و یال‌ها
            edges_by_rel: dict[str, list[dict]] = {
                "OWNS_CARD":      [],
                "OWNS_ACCOUNT":   [],
                "HAS_MOBILE":     [],
                "USES_IP":        [],
                "USES_DEVICE":    [],
                "IDENTIFIED_BY":  [],
            }

            for tx in transactions:
                pid = tx["transaction_id"] + "_person"
                session.run("MERGE (p:Person {id: $pid})", pid=pid)

                conf_ip = tx.get("ip_confidence", GraphConfig.FULL_CONFIDENCE)
                ts      = tx["timestamp_utc"]
                amt     = tx["amount_rial"]

                edges_by_rel["OWNS_CARD"].append(
                    {"from_id": pid, "to_id": tx["card_pan"],         "timestamp": ts, "confidence": GraphConfig.FULL_CONFIDENCE, "amount": amt})
                edges_by_rel["OWNS_ACCOUNT"].append(
                    {"from_id": pid, "to_id": tx["account_number"],   "timestamp": ts, "confidence": GraphConfig.FULL_CONFIDENCE, "amount": amt})
                edges_by_rel["HAS_MOBILE"].append(
                    {"from_id": pid, "to_id": tx["mobile_normalized"],"timestamp": ts, "confidence": GraphConfig.FULL_CONFIDENCE, "amount": amt})
                edges_by_rel["USES_IP"].append(
                    {"from_id": pid, "to_id": tx["ip_address"],       "timestamp": ts, "confidence": conf_ip,                     "amount": amt})
                edges_by_rel["USES_DEVICE"].append(
                    {"from_id": pid, "to_id": tx["device_fp_hash"],   "timestamp": ts, "confidence": GraphConfig.FULL_CONFIDENCE, "amount": amt})
                edges_by_rel["IDENTIFIED_BY"].append(
                    {"from_id": pid, "to_id": tx["national_code"],    "timestamp": ts, "confidence": GraphConfig.FULL_CONFIDENCE, "amount": amt})

            rel_meta = {
                "OWNS_CARD":     ("Person", "Card"),
                "OWNS_ACCOUNT":  ("Person", "Account"),
                "HAS_MOBILE":    ("Person", "Mobile"),
                "USES_IP":       ("Person", "IP"),
                "USES_DEVICE":   ("Person", "Device"),
                "IDENTIFIED_BY": ("Person", "NationalCode"),
            }

            for rel_type, edges in edges_by_rel.items():
                if edges:
                    from_label, to_label = rel_meta[rel_type]
                    session.execute_write(
                        self._merge_edges_for_rel,
                        rel_type, from_label, to_label, edges
                    )

            logger.info(f"ingested {len(transactions)} transactions into graph")


    # ─────────────────────────────────────────────
    # 3. cleanup نودهای قدیمی
    # ─────────────────────────────────────────────
    def cleanup_old_nodes(self, days: int = GraphConfig.CLEANUP_DAYS):
        """حذف IP و Device که در X روز گذشته استفاده نشدن"""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.driver.session() as session:
            for label, rel in [("IP", "USES_IP"), ("Device", "USES_DEVICE")]:
                result = session.run(f"""
                    MATCH (n:{label})
                    WHERE NOT EXISTS {{
                        MATCH (p:Person)-[r:{rel}]->(n)
                        WHERE r.last_timestamp > datetime($cutoff)
                    }}
                    WITH n, count(n) AS cnt
                    DETACH DELETE n
                    RETURN cnt
                """, cutoff=cutoff)
                record = result.single()
                deleted = record["cnt"] if record else 0
                logger.info(f"cleanup: {deleted} {label} nodes deleted (older than {days} days)")

    # ─────────────────────────────────────────────
    # 4. دریافت subgraph هویت
    # ─────────────────────────────────────────────
    def get_identity_subgraph(self, person_id: str) -> Optional[dict]:
        """
        دریافت همه موجودیت‌های متصل به یه Person تا عمق ۲
        خروجی: {person_id, entities: [{type, id, confidence}]}
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {id: $pid})
                OPTIONAL MATCH (p)-[r]->(e)
                WHERE e:Card OR e:Account OR e:Mobile
                   OR e:IP OR e:Device OR e:NationalCode
                RETURN p.id AS person_id,
                       collect(DISTINCT {
                           type:       labels(e)[0],
                           id:         e.id,
                           confidence: r.confidence,
                           tx_count:   r.tx_count,
                           last_ts:    r.last_timestamp
                       }) AS entities
            """, pid=person_id).single()

            if not result:
                return None
            return dict(result)

    def get_shared_entities(self, person_id: str) -> list[dict]:
        """
        پیدا کردن افرادی که یه موجودیت مشترک با این Person دارن
        سیگنال مهم برای تشخیص شبکه تقلب
        """
        with self.driver.session() as session:
            results = session.run("""
                MATCH (p1:Person {id: $pid})-[r1]->(e)<-[r2]-(p2:Person)
                WHERE p1 <> p2
                  AND r1.confidence >= 0.5
                  AND r2.confidence >= 0.5
                RETURN p2.id          AS related_person,
                       labels(e)[0]   AS entity_type,
                       e.id           AS entity_id,
                       r1.confidence  AS conf1,
                       r2.confidence  AS conf2
                ORDER BY conf1 DESC, conf2 DESC
                LIMIT 50
            """, pid=person_id)
            return [dict(r) for r in results]

    def get_high_risk_ips(self, min_unique_cards: int = 3) -> list[dict]:
        """
        IP هایی که بیش از N کارت منحصربه‌فرد داشتن — سیگنال تقلب
        """
        with self.driver.session() as session:
            results = session.run("""
                MATCH (ip:IP)<-[:USES_IP]-(p:Person)-[:OWNS_CARD]->(c:Card)
                WITH ip, count(DISTINCT c) AS unique_cards,
                     count(DISTINCT p)     AS unique_persons
                WHERE unique_cards >= $min_cards
                RETURN ip.id        AS ip_address,
                       unique_cards  AS unique_cards,
                       unique_persons AS unique_persons
                ORDER BY unique_cards DESC
                LIMIT 100
            """, min_cards=min_unique_cards)
            return [dict(r) for r in results]


    def ensure_indexes(self):
        """
        ساخت index روی id هر نوع نود — یه بار اجرا میشه
        """
        with self.driver.session() as session:
            for label in ["Person", "Card", "Account", "Mobile",
                          "IP", "Device", "NationalCode"]:
                session.run(f"""
                    CREATE INDEX {label.lower()}_id_idx IF NOT EXISTS
                    FOR (n:{label}) ON (n.id)
                """)
        logger.info("indexes ensured")
