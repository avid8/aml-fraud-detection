"""
تست‌های Rule-Based Engine
"""

import pytest
from rules import (
    RuleEngine, AccountType, AlertSeverity, RuleCode,
    LIMITS, BEHAVIOR_THRESHOLDS
)

engine = RuleEngine()

BASE_TX = {
    "transaction_id": "TX-001",
    "timestamp_utc":  "2025-06-08T20:00:00Z",
    "amount_rial":    1_000_000,
    "card_pan":       "6037991234567890",
    "account_number": "ACC-001",
    "national_code":  "1000000001",
    "mobile_normalized": "09121111111",
    "ip_address":     "185.1.1.1",
    "device_fp_hash": "fp-abc",
    "is_night":       False,
    "ip_confidence":  1.0,
}

BASE_FEATURES = {
    "acc_total_amount_24h":    0,
    "acc_small_tx_count_24h":  0,
    "ip_unique_cards_1h":      1,
    "dev_unique_cards_1h":     1,
    "acc_fail_ratio_1h":       0.0,
    "card_unique_accounts_24h":1,
}


class TestSingleTxLimit:
    def test_normal_below_limit_no_alert(self):
        tx = {**BASE_TX, "amount_rial": 400_000_000}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.SINGLE_TX_LIMIT for a in r.alerts)

    def test_normal_above_limit_alert(self):
        tx = {**BASE_TX, "amount_rial": 600_000_000}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.NORMAL)
        assert any(a.rule_code == RuleCode.SINGLE_TX_LIMIT for a in r.alerts)

    def test_normal_above_limit_blocks(self):
        tx = {**BASE_TX, "amount_rial": 600_000_000}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.NORMAL)
        assert r.should_block is True

    def test_commercial_higher_limit(self):
        tx = {**BASE_TX, "amount_rial": 600_000_000}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.COMMERCIAL)
        assert not any(a.rule_code == RuleCode.SINGLE_TX_LIMIT for a in r.alerts)

    def test_vip_highest_limit(self):
        tx = {**BASE_TX, "amount_rial": 5_000_000_000}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.VIP)
        assert not any(a.rule_code == RuleCode.SINGLE_TX_LIMIT for a in r.alerts)


class TestDailyLimit:
    def test_daily_below_limit_no_alert(self):
        features = {**BASE_FEATURES, "acc_total_amount_24h": 1_000_000_000}
        r = engine.evaluate(BASE_TX, features, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.DAILY_LIMIT for a in r.alerts)

    def test_daily_above_limit_alert(self):
        features = {**BASE_FEATURES, "acc_total_amount_24h": 3_000_000_000}
        r = engine.evaluate(BASE_TX, features, AccountType.NORMAL)
        assert any(a.rule_code == RuleCode.DAILY_LIMIT for a in r.alerts)
        assert r.should_block is True


class TestStructuring:
    def test_structuring_detected(self):
        tx = {**BASE_TX, "amount_rial": 480_000_000}
        features = {**BASE_FEATURES, "acc_small_tx_count_24h": 4}
        r = engine.evaluate(tx, features, AccountType.NORMAL)
        assert any(a.rule_code == RuleCode.STRUCTURING for a in r.alerts)

    def test_structuring_not_triggered_below_count(self):
        tx = {**BASE_TX, "amount_rial": 480_000_000}
        features = {**BASE_FEATURES, "acc_small_tx_count_24h": 2}
        r = engine.evaluate(tx, features, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.STRUCTURING for a in r.alerts)

    def test_structuring_not_triggered_above_threshold(self):
        tx = {**BASE_TX, "amount_rial": 600_000_000}
        features = {**BASE_FEATURES, "acc_small_tx_count_24h": 5}
        r = engine.evaluate(tx, features, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.STRUCTURING for a in r.alerts)


class TestMultiCardIP:
    def test_multi_card_ip_medium(self):
        features = {**BASE_FEATURES, "ip_unique_cards_1h": 3}
        r = engine.evaluate(BASE_TX, features, AccountType.NORMAL)
        alert = next((a for a in r.alerts if a.rule_code == RuleCode.MULTI_CARD_IP), None)
        assert alert is not None
        assert alert.severity == AlertSeverity.MEDIUM

    def test_multi_card_ip_high(self):
        features = {**BASE_FEATURES, "ip_unique_cards_1h": 6}
        r = engine.evaluate(BASE_TX, features, AccountType.NORMAL)
        alert = next((a for a in r.alerts if a.rule_code == RuleCode.MULTI_CARD_IP), None)
        assert alert.severity == AlertSeverity.HIGH

    def test_single_card_ip_no_alert(self):
        features = {**BASE_FEATURES, "ip_unique_cards_1h": 1}
        r = engine.evaluate(BASE_TX, features, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.MULTI_CARD_IP for a in r.alerts)


class TestNightHighAmount:
    def test_night_high_amount_alert(self):
        tx = {**BASE_TX, "amount_rial": 1_200_000_000, "is_night": True}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.NORMAL)
        assert any(a.rule_code == RuleCode.NIGHT_HIGH_AMOUNT for a in r.alerts)

    def test_night_low_amount_no_alert(self):
        tx = {**BASE_TX, "amount_rial": 100_000_000, "is_night": True}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.NIGHT_HIGH_AMOUNT for a in r.alerts)

    def test_day_high_amount_no_night_alert(self):
        tx = {**BASE_TX, "amount_rial": 1_200_000_000, "is_night": False}
        r = engine.evaluate(tx, BASE_FEATURES, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.NIGHT_HIGH_AMOUNT for a in r.alerts)


class TestFailRatio:
    def test_high_fail_ratio_alert(self):
        features = {**BASE_FEATURES, "acc_fail_ratio_1h": 0.5}
        r = engine.evaluate(BASE_TX, features, AccountType.NORMAL)
        assert any(a.rule_code == RuleCode.HIGH_FAIL_RATIO for a in r.alerts)

    def test_low_fail_ratio_no_alert(self):
        features = {**BASE_FEATURES, "acc_fail_ratio_1h": 0.1}
        r = engine.evaluate(BASE_TX, features, AccountType.NORMAL)
        assert not any(a.rule_code == RuleCode.HIGH_FAIL_RATIO for a in r.alerts)


class TestRiskScore:
    def test_clean_tx_zero_score(self):
        r = engine.evaluate(BASE_TX, BASE_FEATURES, AccountType.NORMAL)
        assert r.risk_score == 0.0

    def test_multiple_alerts_accumulate_score(self):
        tx = {**BASE_TX, "amount_rial": 600_000_000, "is_night": True}
        features = {**BASE_FEATURES, "ip_unique_cards_1h": 4, "acc_fail_ratio_1h": 0.5}
        r = engine.evaluate(tx, features, AccountType.NORMAL)
        assert r.risk_score > 0.5

    def test_score_capped_at_one(self):
        tx = {**BASE_TX, "amount_rial": 600_000_000, "is_night": True}
        features = {
            **BASE_FEATURES,
            "acc_total_amount_24h":    3_000_000_000,
            "acc_small_tx_count_24h":  5,
            "ip_unique_cards_1h":      8,
            "dev_unique_cards_1h":     6,
            "acc_fail_ratio_1h":       0.8,
            "card_unique_accounts_24h":4,
        }
        r = engine.evaluate(tx, features, AccountType.NORMAL)
        assert r.risk_score <= 1.0
