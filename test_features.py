import pytest
from datetime import datetime, timezone
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from features import (
    create_spark_session,
    card_features,
    account_features,
    ip_features,
    device_features,
    NORMALIZED_SCHEMA,
)


@pytest.fixture(scope="session")
def spark():
    return (SparkSession.builder
        .appName("AML_Test")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate())


SAMPLE_ROWS = [
    ("TX-001", "2025-06-08T20:00:00Z", 5_000_000,  "6037991234567890", "ACC-001", "1000000001", "+989121111111", "185.1.1.1", False, 1.0, "fp-aaa", "ONLINE",  "SUCCESS", False),
    ("TX-002", "2025-06-08T20:05:00Z", 4_500_000,  "6037991234567890", "ACC-001", "1000000001", "+989121111111", "185.1.1.1", False, 1.0, "fp-aaa", "ONLINE",  "SUCCESS", False),
    ("TX-003", "2025-06-08T20:10:00Z", 3_000_000,  "6037991234567890", "ACC-002", "1000000001", "+989121111111", "185.1.1.2", False, 1.0, "fp-bbb", "ONLINE",  "FAILED",  False),
    ("TX-004", "2025-06-08T21:00:00Z", 200_000_000,"6104337777777777", "ACC-003", "2000000002", "+989122222222", "10.0.0.1",  True,  0.3, "fp-ccc", "ATM",     "SUCCESS", True),
    ("TX-005", "2025-06-08T21:30:00Z", 180_000_000,"6104337777777777", "ACC-003", "2000000002", "+989122222222", "10.0.0.1",  True,  0.3, "fp-ccc", "ATM",     "SUCCESS", True),
    ("TX-006", "2025-06-08T22:00:00Z", 40_000_000, "6221061234567890", "ACC-004", "3000000003", "+989123333333", "185.1.1.1", False, 1.0, "fp-aaa", "ONLINE",  "SUCCESS", True),
]

COLS = [
    "transaction_id","timestamp_utc","amount_rial","card_pan","account_number",
    "national_code","mobile_e164","ip_address","is_carrier_nat","ip_confidence",
    "device_fp_hash","channel","result","is_night"
]


def make_df(spark):
    df = spark.createDataFrame(SAMPLE_ROWS, COLS)
    return df.withColumn("event_time", expr("CAST(timestamp_utc AS TIMESTAMP)"))


class TestCardFeatures:
    def test_card_tx_count(self, spark):
        df = make_df(spark)
        result = card_features(df, "1 hour").toPandas()
        card = result[result["card_pan"] == "6037991234567890"]
        assert len(card) > 0
        assert card["card_tx_count"].iloc[0] == 3

    def test_card_unique_accounts(self, spark):
        df = make_df(spark)
        result = card_features(df, "1 hour").toPandas()
        card = result[result["card_pan"] == "6037991234567890"]
        assert card["card_unique_accounts"].iloc[0] == 2

    def test_card_fail_ratio(self, spark):
        df = make_df(spark)
        result = card_features(df, "1 hour").toPandas()
        card = result[result["card_pan"] == "6037991234567890"]
        ratio = card["card_fail_ratio"].iloc[0]
        assert abs(ratio - (1/3)) < 0.01

    def test_card_night_ratio(self, spark):
        df = make_df(spark)
        result = card_features(df, "1 hour").toPandas()
        card = result[result["card_pan"] == "6104337777777777"]
        assert card["card_night_ratio"].iloc[0] == 1.0


class TestAccountFeatures:
    def test_acc_tx_count(self, spark):
        df = make_df(spark)
        result = account_features(df, "1 hour").toPandas()
        acc = result[result["account_number"] == "ACC-001"]
        assert acc["acc_tx_count"].iloc[0] == 2

    def test_structuring_detection(self, spark):
        """تراکنش‌های زیر ۵۰ میلیون — سیگنال smurfing"""
        df = make_df(spark)
        result = account_features(df, "24 hours").toPandas()
        acc = result[result["account_number"] == "ACC-001"]
        assert acc["acc_small_tx_count"].iloc[0] == 2


class TestIPFeatures:
    def test_ip_unique_cards(self, spark):
        """یه IP با چند کارت — سیگنال تقلب"""
        df = make_df(spark)
        result = ip_features(df, "24 hours").toPandas()
        ip = result[result["ip_address"] == "185.1.1.1"]
        assert ip["ip_unique_cards"].iloc[0] >= 2

    def test_nat_ip_detected(self, spark):
        df = make_df(spark)
        result = ip_features(df, "24 hours").toPandas()
        nat_ip = result[result["ip_address"] == "10.0.0.1"]
        assert nat_ip["ip_nat_ratio"].iloc[0] == 1.0


class TestDeviceFeatures:
    def test_device_unique_cards(self, spark):
        """یه دستگاه با چند کارت"""
        df = make_df(spark)
        result = device_features(df, "24 hours").toPandas()
        dev = result[result["device_fp_hash"] == "fp-aaa"]
        assert dev["dev_unique_cards"].iloc[0] >= 2

    def test_device_tx_count(self, spark):
        df = make_df(spark)
        result = device_features(df, "24 hours").toPandas()
        dev = result[result["device_fp_hash"] == "fp-ccc"]
        assert dev["dev_tx_count"].iloc[0] == 2
