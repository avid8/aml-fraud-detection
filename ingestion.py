import hashlib
import ipaddress
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import json
import os

import requests
import phonenumbers
from pydantic import BaseModel, field_validator, model_validator, ValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


class Config:
    BANK_API_BASE_URL = os.getenv("BANK_API_URL", "https://api.bank.internal/v1")
    BANK_API_KEY = os.getenv("BANK_API_KEY", "")
    BANK_API_TIMEOUT = int(os.getenv("BANK_API_TIMEOUT", "10"))
    POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "5"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
    KAFKA_TOPIC_CLEAN = "aml.transactions.clean"
    KAFKA_TOPIC_DLQ = "aml.transactions.dlq"
    LOCAL_TZ = "Asia/Tehran"
    CARRIER_NAT_RANGES = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10"]


def _is_valid_iranian_national_code(code: str) -> bool:
    if len(set(code)) == 1:
        return False
    check = int(code[9])
    s = sum(int(code[i]) * (10 - i) for i in range(9)) % 11
    return (s < 2 and check == s) or (s >= 2 and check == 11 - s)


class RawTransaction(BaseModel):
    transaction_id:     str
    timestamp:          str
    amount_rial:        int
    card_pan:           str
    account_number:     str
    national_code:      str
    mobile:             str
    ip_address:         str
    device_fingerprint: str
    user_agent:         Optional[str] = None
    channel:            Optional[str] = None
    result:             Optional[str] = None

    @field_validator("card_pan")
    @classmethod
    def validate_card(cls, v: str) -> str:
        digits = re.sub(r"\D", "", v)
        if len(digits) != 16:
            raise ValueError(f"card_pan باید ۱۶ رقم باشه، دریافت شد: {len(digits)}")
        return digits

    @field_validator("national_code")
    @classmethod
    def validate_national_code(cls, v: str) -> str:
        digits = re.sub(r"\D", "", v)
        if len(digits) != 10:
            raise ValueError("کد ملی باید ۱۰ رقم باشه")
        if not _is_valid_iranian_national_code(digits):
            raise ValueError(f"کد ملی {digits} معتبر نیست (checksum)")
        return digits

    @field_validator("amount_rial")
    @classmethod
    def validate_amount(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("مبلغ باید مثبت باشه")
        if v > 500_000_000_000:
            raise ValueError(f"مبلغ غیرمعقول: {v}")
        return v

    @field_validator("ip_address")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        try:
            ipaddress.ip_address(v.strip())
            return v.strip()
        except ValueError:
            raise ValueError(f"IP نامعتبر: {v}")

    @model_validator(mode="after")
    def check_required_fields_non_empty(self) -> "RawTransaction":
        required = ["transaction_id", "card_pan", "account_number",
                    "national_code", "mobile", "ip_address", "device_fingerprint"]
        for f in required:
            val = getattr(self, f, None)
            if not val or str(val).strip() == "":
                raise ValueError(f"فیلد اجباری خالیه: {f}")
        return self


@dataclass
class NormalizedTransaction:
    transaction_id:       str
    timestamp_utc:        str
    timestamp_local:      str
    amount_rial:          int
    card_pan:             str
    card_pan_masked:      str
    account_number:       str
    national_code:        str
    mobile_e164:          str
    mobile_normalized:    str
    ip_address:           str
    is_carrier_nat:       bool
    ip_confidence:        float
    device_fingerprint:   str
    device_fp_hash:       str
    user_agent:           Optional[str]
    channel:              Optional[str]
    result:               Optional[str]
    ingested_at:          str
    is_night:             bool


@dataclass
class DLQRecord:
    raw_payload:  dict
    error:        str
    failed_at:    str
    source:       str = "bank_api"


def normalize_mobile(mobile: str) -> tuple:
    raw = re.sub(r"\D", "", mobile)
    if raw.startswith("98") and len(raw) == 12:
        raw = "0" + raw[2:]
    elif raw.startswith("9") and len(raw) == 10:
        raw = "0" + raw
    try:
        parsed = phonenumbers.parse(raw, "IR")
        if phonenumbers.is_valid_number(parsed):
            e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            normalized = "0" + str(parsed.national_number)
            return e164, normalized
    except phonenumbers.phonenumberutil.NumberParseException:
        pass
    return f"+98{raw.lstrip('0')}", raw


def parse_timestamp(ts_str: str, local_tz: str = "Asia/Tehran") -> tuple:
    from zoneinfo import ZoneInfo
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    dt = None
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str.strip(), fmt)
            break
        except ValueError:
            continue
    if dt is None:
        try:
            dt = datetime.fromisoformat(ts_str.strip().replace("Z", "+00:00"))
        except Exception:
            raise ValueError(f"نمی‌تونم timestamp رو parse کنم: {ts_str}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(local_tz))
    utc_dt = dt.astimezone(timezone.utc)
    local_dt = dt.astimezone(ZoneInfo(local_tz))
    is_night = utc_dt.hour >= 20 or utc_dt.hour < 2
    return (
        utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        local_dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        is_night,
    )


def is_carrier_nat(ip: str, nat_ranges: list) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        for cidr in nat_ranges:
            if addr in ipaddress.ip_network(cidr, strict=False):
                return True
    except ValueError:
        pass
    return False


def mask_card(pan: str) -> str:
    return pan[:6] + "x" * (len(pan) - 10) + pan[-4:]


def hash_device(fp: str) -> str:
    return hashlib.sha256(fp.encode("utf-8")).hexdigest()


def normalize(raw: RawTransaction) -> NormalizedTransaction:
    utc_ts, local_ts, is_night = parse_timestamp(raw.timestamp)
    e164, mobile_norm = normalize_mobile(raw.mobile)
    nat = is_carrier_nat(raw.ip_address, Config.CARRIER_NAT_RANGES)
    return NormalizedTransaction(
        transaction_id=raw.transaction_id,
        timestamp_utc=utc_ts,
        timestamp_local=local_ts,
        amount_rial=raw.amount_rial,
        card_pan=raw.card_pan,
        card_pan_masked=mask_card(raw.card_pan),
        account_number=raw.account_number,
        national_code=raw.national_code,
        mobile_e164=e164,
        mobile_normalized=mobile_norm,
        ip_address=raw.ip_address,
        is_carrier_nat=nat,
        ip_confidence=0.3 if nat else 1.0,
        device_fingerprint=raw.device_fingerprint,
        device_fp_hash=hash_device(raw.device_fingerprint),
        user_agent=raw.user_agent,
        channel=raw.channel,
        result=raw.result,
        ingested_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        is_night=is_night,
    )


class DeadLetterQueue:
    def __init__(self, kafka_producer=None, topic: str = Config.KAFKA_TOPIC_DLQ):
        self._producer = kafka_producer
        self._topic = topic
        self._local_buffer: list = []

    def send(self, payload: dict, error: str) -> None:
        record = DLQRecord(
            raw_payload=payload,
            error=error,
            failed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self._local_buffer.append(record)
        logger.warning(f"[DLQ] tx_id={payload.get('transaction_id','?')} | {error}")

    def flush(self) -> list:
        out = self._local_buffer.copy()
        self._local_buffer.clear()
        return out


class BankAPIClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 10):
        self._base = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key, "Accept": "application/json"}
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self._headers)

    def fetch_batch(self, since: Optional[str] = None, batch_size: int = 500, retries: int = 3) -> list:
        params: dict = {"limit": batch_size}
        if since:
            params["since"] = since
        last_error = None
        for attempt in range(retries):
            try:
                resp = self._session.get(f"{self._base}/transactions", params=params, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ("transactions", "data", "records", "items"):
                        if key in data:
                            return data[key]
                return [data]
            except requests.HTTPError as e:
                status = e.response.status_code if e.response else 0
                if status in (401, 403):
                    raise
                time.sleep(2 ** attempt)
                last_error = e
            except Exception as e:
                time.sleep(2 ** attempt)
                last_error = e
        raise ConnectionError(f"بعد از {retries} تلاش داده نگرفتیم: {last_error}")


class InputPipeline:
    def __init__(self, client: BankAPIClient, dlq: DeadLetterQueue, kafka_producer=None):
        self._client = client
        self._dlq = dlq
        self._producer = kafka_producer
        self._last_seen_timestamp: Optional[str] = None
        self._stats = {"ok": 0, "dlq": 0, "batches": 0}

    def process_batch(self, raw_records: list) -> list:
        normalized = []
        for record in raw_records:
            try:
                raw_tx = RawTransaction(**record)
                norm_tx = normalize(raw_tx)
                normalized.append(norm_tx)
                self._stats["ok"] += 1
            except ValidationError as e:
                errors = "; ".join(f"{err['loc']}: {err['msg']}" for err in e.errors())
                self._dlq.send(record, f"ValidationError: {errors}")
                self._stats["dlq"] += 1
            except Exception as e:
                self._dlq.send(record, f"UnexpectedError: {e}")
                self._stats["dlq"] += 1
        return normalized

    def run_once(self) -> dict:
        raw = self._client.fetch_batch(since=self._last_seen_timestamp, batch_size=Config.BATCH_SIZE)
        if not raw:
            return self._stats
        normalized = self.process_batch(raw)
        if normalized:
            self._last_seen_timestamp = max(tx.timestamp_utc for tx in normalized)
        self._stats["batches"] += 1
        return self._stats


def build_pipeline(kafka_producer=None) -> InputPipeline:
    client = BankAPIClient(base_url=Config.BANK_API_BASE_URL, api_key=Config.BANK_API_KEY)
    dlq = DeadLetterQueue(kafka_producer=kafka_producer)
    return InputPipeline(client=client, dlq=dlq, kafka_producer=kafka_producer)
