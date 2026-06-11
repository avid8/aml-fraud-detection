import pytest
from unittest.mock import MagicMock
from ingestion import (
    RawTransaction, normalize, normalize_mobile,
    parse_timestamp, is_carrier_nat, mask_card,
    DeadLetterQueue, InputPipeline, BankAPIClient, Config,
)
from pydantic import ValidationError

VALID_RECORD = {
    "transaction_id": "TX-001",
    "timestamp": "2025-06-08T22:30:00",
    "amount_rial": 5_000_000,
    "card_pan": "6037991234567890",
    "account_number": "0123456789",
    "national_code": "1000000001",
    "mobile": "09123456789",
    "ip_address": "185.55.224.10",
    "device_fingerprint": "fp-abc-123",
    "channel": "ONLINE",
    "result": "SUCCESS",
}

class TestValidation:
    def test_valid_record_passes(self):
        tx = RawTransaction(**VALID_RECORD)
        assert tx.transaction_id == "TX-001"

    def test_short_card_rejected(self):
        with pytest.raises(ValidationError):
            RawTransaction(**{**VALID_RECORD, "card_pan": "12345"})

    def test_negative_amount_rejected(self):
        with pytest.raises(ValidationError):
            RawTransaction(**{**VALID_RECORD, "amount_rial": -100})

    def test_invalid_ip_rejected(self):
        with pytest.raises(ValidationError):
            RawTransaction(**{**VALID_RECORD, "ip_address": "999.999.0.1"})

class TestMobileNormalization:
    def test_09_format(self):
        e164, norm = normalize_mobile("09123456789")
        assert e164 == "+989123456789"
        assert norm == "09123456789"

    def test_98_prefix(self):
        e164, _ = normalize_mobile("989123456789")
        assert e164 == "+989123456789"

    def test_9_prefix_10_digits(self):
        _, norm = normalize_mobile("9123456789")
        assert norm == "09123456789"

class TestTimestampParsing:
    def test_iso_no_tz(self):
        utc, _, _ = parse_timestamp("2025-06-08T22:30:00")
        assert utc.endswith("Z")

    def test_night_detection(self):
        _, _, is_night = parse_timestamp("2025-06-08T23:30:00")
        assert is_night is True

    def test_day_detection(self):
        _, _, is_night = parse_timestamp("2025-06-08T10:00:00")
        assert is_night is False

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_timestamp("not-a-date")

class TestCarrierNAT:
    def test_private_ip_is_nat(self):
        assert is_carrier_nat("192.168.1.1", Config.CARRIER_NAT_RANGES) is True
        assert is_carrier_nat("10.0.0.5", Config.CARRIER_NAT_RANGES) is True

    def test_public_ip_not_nat(self):
        assert is_carrier_nat("185.55.224.10", Config.CARRIER_NAT_RANGES) is False

class TestNormalize:
    def test_full_normalization(self):
        raw = RawTransaction(**VALID_RECORD)
        norm = normalize(raw)
        assert norm.mobile_e164.startswith("+98")
        assert norm.ip_confidence == 1.0
        assert norm.timestamp_utc.endswith("Z")

    def test_nat_ip_gets_low_confidence(self):
        raw = RawTransaction(**{**VALID_RECORD, "ip_address": "192.168.1.1"})
        norm = normalize(raw)
        assert norm.is_carrier_nat is True
        assert norm.ip_confidence == 0.3

class TestDeadLetterQueue:
    def test_record_goes_to_dlq(self):
        dlq = DeadLetterQueue()
        dlq.send({"transaction_id": "TX-BAD"}, "test error")
        records = dlq.flush()
        assert len(records) == 1
        assert dlq.flush() == []

class TestInputPipeline:
    def _make_pipeline(self):
        client = MagicMock(spec=BankAPIClient)
        dlq = DeadLetterQueue()
        return InputPipeline(client=client, dlq=dlq), client, dlq

    def test_valid_batch(self):
        pipeline, client, _ = self._make_pipeline()
        client.fetch_batch.return_value = [VALID_RECORD]
        stats = pipeline.run_once()
        assert stats["ok"] == 1
        assert stats["dlq"] == 0

    def test_invalid_batch_goes_to_dlq(self):
        pipeline, client, dlq = self._make_pipeline()
        client.fetch_batch.return_value = [{**VALID_RECORD, "amount_rial": -1}]
        stats = pipeline.run_once()
        assert stats["dlq"] == 1

    def test_empty_batch(self):
        pipeline, client, _ = self._make_pipeline()
        client.fetch_batch.return_value = []
        stats = pipeline.run_once()
        assert stats["ok"] == 0
