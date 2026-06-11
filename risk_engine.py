"""
AML Pipeline — لایه Risk Engine (خروجی نهایی)
ترکیب Rule Engine + ML Models → Risk Score نهایی + هشدار اولویت‌بندی‌شده
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from rules import RuleResult, AlertSeverity, RuleCode
from ml_models import MLScorer

logger = logging.getLogger(__name__)


class RiskLevel:
    LOW      = "low"       # 0.0 - 0.3
    MEDIUM   = "medium"    # 0.3 - 0.6
    HIGH     = "high"      # 0.6 - 0.8
    CRITICAL = "critical"  # 0.8 - 1.0


@dataclass
class FinalDecision:
    transaction_id:  str
    account_id:      str
    risk_score:      float
    risk_level:      str
    should_block:    bool
    rule_score:      float
    ml_score:        float
    lstm_score:      Optional[float]
    ae_score:        Optional[float]
    behavioral_score:Optional[float]
    profile_name:    Optional[str]
    alert_count:     int
    top_alerts:      list
    decided_at:      str = field(default_factory=lambda:
                         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> dict:
        return {
            "transaction_id":   self.transaction_id,
            "account_id":       self.account_id,
            "risk_score":       self.risk_score,
            "risk_level":       self.risk_level,
            "should_block":     self.should_block,
            "rule_score":       self.rule_score,
            "ml_score":         self.ml_score,
            "lstm_score":       self.lstm_score,
            "ae_score":         self.ae_score,
            "behavioral_score": self.behavioral_score,
            "profile_name":     self.profile_name,
            "alert_count":      self.alert_count,
            "top_alerts":       self.top_alerts,
            "decided_at":       self.decided_at,
        }


class RiskEngine:
    """
    ترکیب Rule Engine + ML Scorer → تصمیم نهایی
    وزن‌ها: rules 60% + ml 40%
    """
    RULE_WEIGHT = 0.6
    ML_WEIGHT   = 0.4

    BLOCK_THRESHOLD    = 0.7
    CRITICAL_THRESHOLD = 0.8
    HIGH_THRESHOLD     = 0.6
    MEDIUM_THRESHOLD   = 0.3

    def __init__(self, ml_scorer: Optional[MLScorer] = None):
        self.ml_scorer = ml_scorer

    def decide(self,
               rule_result: RuleResult,
               sequence=None,
               behavioral=None) -> FinalDecision:

        # ── score از rules ──
        rule_score = rule_result.risk_score

        # ── score از ML ──
        ml_details = {"final_score": 0.0, "lstm_score": None,
                      "ae_score": None, "behavioral_score": None,
                      "profile_name": None}
        if self.ml_scorer:
            ml_details = self.ml_scorer.score(
                sequence=sequence, behavioral=behavioral
            )
        ml_score = ml_details.get("final_score", 0.0) or 0.0

        # ── ترکیب نهایی ──
        final_score = round(
            rule_score * self.RULE_WEIGHT + ml_score * self.ML_WEIGHT, 4
        )
        final_score = min(1.0, final_score)

        # ── سطح ریسک ──
        if final_score >= self.CRITICAL_THRESHOLD:
            risk_level = RiskLevel.CRITICAL
        elif final_score >= self.HIGH_THRESHOLD:
            risk_level = RiskLevel.HIGH
        elif final_score >= self.MEDIUM_THRESHOLD:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # ── مسدود کردن ──
        should_block = (
            rule_result.should_block or
            final_score >= self.BLOCK_THRESHOLD
        )

        # ── مرتب‌سازی alert ها بر اساس severity ──
        severity_order = {
            AlertSeverity.HIGH:   0,
            AlertSeverity.MEDIUM: 1,
            AlertSeverity.LOW:    2,
        }
        sorted_alerts = sorted(
            rule_result.alerts,
            key=lambda a: severity_order.get(a.severity, 9)
        )
        top_alerts = [
            {
                "code":     a.rule_code.value,
                "severity": a.severity.value,
                "message":  a.message,
            }
            for a in sorted_alerts[:5]
        ]

        decision = FinalDecision(
            transaction_id   = rule_result.transaction_id,
            account_id       = rule_result.account_id,
            risk_score       = final_score,
            risk_level       = risk_level,
            should_block     = should_block,
            rule_score       = round(rule_score, 4),
            ml_score         = round(ml_score, 4),
            lstm_score       = ml_details.get("lstm_score"),
            ae_score         = ml_details.get("ae_score"),
            behavioral_score = ml_details.get("behavioral_score"),
            profile_name     = ml_details.get("profile_name"),
            alert_count      = len(rule_result.alerts),
            top_alerts       = top_alerts,
        )

        logger.info(
            f"[RiskEngine] tx={decision.transaction_id} "
            f"score={decision.risk_score} "
            f"level={decision.risk_level} "
            f"block={decision.should_block}"
        )
        return decision
