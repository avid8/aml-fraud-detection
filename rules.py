"""
AML Pipeline — لایه Rule-Based Engine
سقف معاملاتی بر اساس نوع حساب + تشخیص الگوهای مشکوک
خروجی: Alert + مسدود کردن + Risk Score
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. انواع و ثابت‌ها
# ─────────────────────────────────────────────
class AccountType(Enum):
    NORMAL     = "normal"
    COMMERCIAL = "commercial"
    VIP        = "vip"


class AlertSeverity(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class RuleCode(Enum):
    SINGLE_TX_LIMIT        = "R001"  # تراکنش تکی بالای سقف
    DAILY_LIMIT            = "R002"  # مجموع روزانه بالای سقف
    STRUCTURING            = "R003"  # تراکنش‌های زیر سقف با تعداد بالا
    MULTI_CARD_IP          = "R004"  # چند کارت از یه IP
    MULTI_CARD_DEVICE      = "R005"  # چند کارت از یه دستگاه
    NIGHT_HIGH_AMOUNT      = "R006"  # مبلغ بالا در ساعت شب
    HIGH_FAIL_RATIO        = "R007"  # نسبت تراکنش ناموفق بالا
    MULTI_ACCOUNT_CARD     = "R008"  # یه کارت با چند حساب


# سقف‌های معاملاتی (ریال)
LIMITS = {
    AccountType.NORMAL: {
        "single_tx":      500_000_000,      # ۵۰۰ میلیون
        "daily_total":  2_000_000_000,      # ۲ میلیارد
        "structuring_threshold": 490_000_000,
        "structuring_count": 3,
    },
    AccountType.COMMERCIAL: {
        "single_tx":    5_000_000_000,      # ۵ میلیارد
        "daily_total": 20_000_000_000,      # ۲۰ میلیارد
        "structuring_threshold": 4_900_000_000,
        "structuring_count": 5,
    },
    AccountType.VIP: {
        "single_tx":   50_000_000_000,      # ۵۰ میلیارد
        "daily_total": 100_000_000_000,     # ۱۰۰ میلیارد
        "structuring_threshold": 49_000_000_000,
        "structuring_count": 10,
    },
}

# آستانه‌های رفتاری
BEHAVIOR_THRESHOLDS = {
    "multi_card_ip_count":      3,    # بیش از ۳ کارت از یه IP
    "multi_card_device_count":  3,    # بیش از ۳ کارت از یه دستگاه
    "night_high_amount_ratio":  0.5,  # بیش از ۵۰٪ سقف روزانه در شب
    "fail_ratio_threshold":     0.4,  # بیش از ۴۰٪ تراکنش ناموفق
    "multi_account_card_count": 3,    # یه کارت با بیش از ۳ حساب
}


# ─────────────────────────────────────────────
# 2. مدل‌های خروجی
# ─────────────────────────────────────────────
@dataclass
class Alert:
    rule_code:      RuleCode
    severity:       AlertSeverity
    message:        str
    transaction_id: str
    account_id:     str
    card_pan:       Optional[str]
    amount:         Optional[int]
    extra:          dict = field(default_factory=dict)
    created_at:     str  = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


@dataclass
class RuleResult:
    transaction_id: str
    account_id:     str
    account_type:   AccountType
    risk_score:     float           # 0.0 تا 1.0
    should_block:   bool
    alerts:         list[Alert] = field(default_factory=list)

    def add_alert(self, alert: Alert):
        self.alerts.append(alert)
        self.risk_score = min(1.0, self.risk_score + _alert_score(alert.severity))
        if alert.severity == AlertSeverity.HIGH:
            self.should_block = True


def _alert_score(severity: AlertSeverity) -> float:
    return {AlertSeverity.LOW: 0.1, AlertSeverity.MEDIUM: 0.25, AlertSeverity.HIGH: 0.4}[severity]


# ─────────────────────────────────────────────
# 3. Rule Engine
# ─────────────────────────────────────────────
class RuleEngine:
    def evaluate(self, tx: dict, features: dict, account_type: AccountType) -> RuleResult:
        """
        tx: NormalizedTransaction به صورت dict
        features: ویژگی‌های محاسبه‌شده از لایه ۲ (Spark)
        account_type: نوع حساب
        """
        result = RuleResult(
            transaction_id=tx["transaction_id"],
            account_id=tx["account_number"],
            account_type=account_type,
            risk_score=0.0,
            should_block=False,
        )

        limits = LIMITS[account_type]

        self._check_single_tx_limit(tx, limits, result)
        self._check_daily_limit(tx, features, limits, result)
        self._check_structuring(tx, features, limits, result)
        self._check_multi_card_ip(tx, features, result)
        self._check_multi_card_device(tx, features, result)
        self._check_night_high_amount(tx, limits, result)
        self._check_fail_ratio(features, result)
        self._check_multi_account_card(features, result)

        logger.info(
            f"[RuleEngine] tx={tx['transaction_id']} "
            f"score={result.risk_score:.2f} "
            f"block={result.should_block} "
            f"alerts={len(result.alerts)}"
        )
        return result

    # ── R001: تراکنش تکی بالای سقف ──
    def _check_single_tx_limit(self, tx, limits, result):
        if tx["amount_rial"] > limits["single_tx"]:
            result.add_alert(Alert(
                rule_code=RuleCode.SINGLE_TX_LIMIT,
                severity=AlertSeverity.HIGH,
                message=f"مبلغ تراکنش {tx['amount_rial']:,} ریال از سقف {limits['single_tx']:,} ریال بیشتره",
                transaction_id=tx["transaction_id"],
                account_id=tx["account_number"],
                card_pan=tx.get("card_pan"),
                amount=tx["amount_rial"],
                extra={"limit": limits["single_tx"]},
            ))

    # ── R002: مجموع روزانه بالای سقف ──
    def _check_daily_limit(self, tx, features, limits, result):
        daily_total = features.get("acc_total_amount_24h", 0)
        if daily_total > limits["daily_total"]:
            result.add_alert(Alert(
                rule_code=RuleCode.DAILY_LIMIT,
                severity=AlertSeverity.HIGH,
                message=f"مجموع روزانه {daily_total:,} ریال از سقف {limits['daily_total']:,} ریال بیشتره",
                transaction_id=tx["transaction_id"],
                account_id=tx["account_number"],
                card_pan=tx.get("card_pan"),
                amount=daily_total,
                extra={"limit": limits["daily_total"], "window": "24h"},
            ))

    # ── R003: structuring (smurfing) ──
    def _check_structuring(self, tx, features, limits, result):
        small_count = features.get("acc_small_tx_count_24h", 0)
        threshold   = limits["structuring_threshold"]
        min_count   = limits["structuring_count"]

        if tx["amount_rial"] < threshold and small_count >= min_count:
            result.add_alert(Alert(
                rule_code=RuleCode.STRUCTURING,
                severity=AlertSeverity.HIGH,
                message=f"الگوی structuring: {small_count} تراکنش زیر سقف در ۲۴ ساعت",
                transaction_id=tx["transaction_id"],
                account_id=tx["account_number"],
                card_pan=tx.get("card_pan"),
                amount=tx["amount_rial"],
                extra={"small_tx_count": small_count, "threshold": threshold},
            ))

    # ── R004: چند کارت از یه IP ──
    def _check_multi_card_ip(self, tx, features, result):
        unique_cards = features.get("ip_unique_cards_1h", 0)
        limit        = BEHAVIOR_THRESHOLDS["multi_card_ip_count"]
        if unique_cards >= limit:
            severity = AlertSeverity.HIGH if unique_cards >= limit * 2 else AlertSeverity.MEDIUM
            result.add_alert(Alert(
                rule_code=RuleCode.MULTI_CARD_IP,
                severity=severity,
                message=f"IP {tx.get('ip_address')} در ۱ ساعت با {unique_cards} کارت مختلف استفاده شده",
                transaction_id=tx["transaction_id"],
                account_id=tx["account_number"],
                card_pan=tx.get("card_pan"),
                amount=tx["amount_rial"],
                extra={"ip": tx.get("ip_address"), "unique_cards": unique_cards},
            ))

    # ── R005: چند کارت از یه دستگاه ──
    def _check_multi_card_device(self, tx, features, result):
        unique_cards = features.get("dev_unique_cards_1h", 0)
        limit        = BEHAVIOR_THRESHOLDS["multi_card_device_count"]
        if unique_cards >= limit:
            severity = AlertSeverity.HIGH if unique_cards >= limit * 2 else AlertSeverity.MEDIUM
            result.add_alert(Alert(
                rule_code=RuleCode.MULTI_CARD_DEVICE,
                severity=severity,
                message=f"دستگاه {tx.get('device_fp_hash','')[:8]}... با {unique_cards} کارت مختلف",
                transaction_id=tx["transaction_id"],
                account_id=tx["account_number"],
                card_pan=tx.get("card_pan"),
                amount=tx["amount_rial"],
                extra={"device": tx.get("device_fp_hash"), "unique_cards": unique_cards},
            ))

    # ── R006: مبلغ بالا در شب ──
    def _check_night_high_amount(self, tx, limits, result):
        if not tx.get("is_night"):
            return
        ratio = tx["amount_rial"] / limits["daily_total"]
        if ratio >= BEHAVIOR_THRESHOLDS["night_high_amount_ratio"]:
            result.add_alert(Alert(
                rule_code=RuleCode.NIGHT_HIGH_AMOUNT,
                severity=AlertSeverity.MEDIUM,
                message=f"تراکنش {ratio*100:.0f}٪ سقف روزانه در ساعت شب",
                transaction_id=tx["transaction_id"],
                account_id=tx["account_number"],
                card_pan=tx.get("card_pan"),
                amount=tx["amount_rial"],
                extra={"night_ratio": ratio},
            ))

    # ── R007: نسبت تراکنش ناموفق ──
    def _check_fail_ratio(self, features, result):
        fail_ratio = features.get("acc_fail_ratio_1h", 0.0)
        if fail_ratio >= BEHAVIOR_THRESHOLDS["fail_ratio_threshold"]:
            result.add_alert(Alert(
                rule_code=RuleCode.HIGH_FAIL_RATIO,
                severity=AlertSeverity.MEDIUM,
                message=f"نسبت تراکنش ناموفق {fail_ratio*100:.0f}٪ در ۱ ساعت",
                transaction_id=result.transaction_id,
                account_id=result.account_id,
                card_pan=None,
                amount=None,
                extra={"fail_ratio": fail_ratio},
            ))

    # ── R008: یه کارت با چند حساب ──
    def _check_multi_account_card(self, features, result):
        unique_accounts = features.get("card_unique_accounts_24h", 0)
        limit           = BEHAVIOR_THRESHOLDS["multi_account_card_count"]
        if unique_accounts >= limit:
            result.add_alert(Alert(
                rule_code=RuleCode.MULTI_ACCOUNT_CARD,
                severity=AlertSeverity.MEDIUM,
                message=f"یه کارت با {unique_accounts} حساب مختلف در ۲۴ ساعت",
                transaction_id=result.transaction_id,
                account_id=result.account_id,
                card_pan=None,
                amount=None,
                extra={"unique_accounts": unique_accounts},
            ))
