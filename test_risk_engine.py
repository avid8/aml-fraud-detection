"""
تست‌های Risk Engine
"""

import pytest
from unittest.mock import MagicMock
from rules import RuleResult, AccountType, Alert, AlertSeverity, RuleCode
from risk_engine import RiskEngine, RiskLevel, FinalDecision
from ml_models import MLScorer


def make_rule_result(risk_score=0.0, should_block=False, alerts=None):
    r = RuleResult(
        transaction_id="TX-001",
        account_id="ACC-001",
        account_type=AccountType.NORMAL,
        risk_score=risk_score,
        should_block=should_block,
        alerts=alerts or [],
    )
    return r


def make_alert(severity=AlertSeverity.HIGH, code=RuleCode.SINGLE_TX_LIMIT):
    return Alert(
        rule_code=code,
        severity=severity,
        message="test alert",
        transaction_id="TX-001",
        account_id="ACC-001",
        card_pan=None,
        amount=None,
    )


class TestRiskLevel:
    def test_low_score(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=0.1)
        d = engine.decide(r)
        assert d.risk_level == RiskLevel.LOW

    def test_medium_score(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=0.5)
        d = engine.decide(r)
        assert d.risk_level == RiskLevel.MEDIUM

    def test_high_score(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=1.0)
        d = engine.decide(r)
        assert d.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_critical_score(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=1.0)
        d = engine.decide(r)
        assert d.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


class TestBlocking:
    def test_rule_block_overrides_low_score(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=0.1, should_block=True)
        d = engine.decide(r)
        assert d.should_block is True

    def test_high_score_blocks(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=1.0, should_block=True)
        d = engine.decide(r)
        assert d.should_block is True

    def test_low_score_no_block(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=0.1)
        d = engine.decide(r)
        assert d.should_block is False


class TestScoreCombination:
    def test_rule_weight(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=1.0)
        d = engine.decide(r)
        assert d.rule_score == 1.0
        assert d.ml_score == 0.0
        assert abs(d.risk_score - RiskEngine.RULE_WEIGHT) < 0.01

    def test_ml_score_combined(self):
        mock_scorer = MagicMock(spec=MLScorer)
        mock_scorer.score.return_value = {
            "final_score": 1.0,
            "lstm_score": 0.9,
            "ae_score": 0.8,
            "behavioral_score": 0.7,
            "profile_name": "مشکوک",
        }
        engine = RiskEngine(ml_scorer=mock_scorer)
        r = make_rule_result(risk_score=1.0)
        d = engine.decide(r)
        assert d.risk_score == 1.0
        assert d.profile_name == "مشکوک"

    def test_score_capped_at_one(self):
        mock_scorer = MagicMock(spec=MLScorer)
        mock_scorer.score.return_value = {"final_score": 1.0,
            "lstm_score": 1.0, "ae_score": 1.0,
            "behavioral_score": 1.0, "profile_name": None}
        engine = RiskEngine(ml_scorer=mock_scorer)
        r = make_rule_result(risk_score=1.0)
        d = engine.decide(r)
        assert d.risk_score <= 1.0


class TestAlertSorting:
    def test_high_alerts_first(self):
        alerts = [
            make_alert(AlertSeverity.LOW,    RuleCode.HIGH_FAIL_RATIO),
            make_alert(AlertSeverity.HIGH,   RuleCode.SINGLE_TX_LIMIT),
            make_alert(AlertSeverity.MEDIUM, RuleCode.NIGHT_HIGH_AMOUNT),
        ]
        engine = RiskEngine()
        r = make_rule_result(risk_score=0.5, alerts=alerts)
        d = engine.decide(r)
        assert d.top_alerts[0]["severity"] == "high"
        assert d.top_alerts[1]["severity"] == "medium"
        assert d.top_alerts[2]["severity"] == "low"

    def test_max_five_alerts(self):
        alerts = [make_alert() for _ in range(10)]
        engine = RiskEngine()
        r = make_rule_result(risk_score=0.5, alerts=alerts)
        d = engine.decide(r)
        assert len(d.top_alerts) <= 5


class TestFinalDecision:
    def test_to_dict_has_all_keys(self):
        engine = RiskEngine()
        r = make_rule_result(risk_score=0.4)
        d = engine.decide(r)
        keys = d.to_dict().keys()
        for key in ["transaction_id", "risk_score", "risk_level",
                    "should_block", "rule_score", "ml_score",
                    "alert_count", "decided_at"]:
            assert key in keys

    def test_decided_at_is_utc(self):
        engine = RiskEngine()
        r = make_rule_result()
        d = engine.decide(r)
        assert d.decided_at.endswith("Z")
