"""
AML Pipeline — لایه Feature Extraction
ورودی: تراکنش‌های نرمال‌شده از Kafka
خروجی: feature vectors به Kafka
پنجره‌های زمانی: 1h, 6h, 24h, 7d
"""

import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_json, struct, window,
    count, sum as spark_sum, avg, stddev,
    approx_count_distinct, max as spark_max,
    min as spark_min, when, hour, dayofweek,
    lit, expr
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    LongType, BooleanType, DoubleType
)

logger = logging.getLogger(__name__)

KAFKA_BROKERS   = os.getenv("KAFKA_BROKERS", "localhost:9092")
INPUT_TOPIC     = os.getenv("KAFKA_INPUT_TOPIC", "aml.transactions.clean")
OUTPUT_TOPIC    = os.getenv("KAFKA_OUTPUT_TOPIC", "aml.features")
CHECKPOINT_BASE = os.getenv("CHECKPOINT_PATH", "/tmp/aml_checkpoints")

NORMALIZED_SCHEMA = StructType([
    StructField("transaction_id",    StringType(),  False),
    StructField("timestamp_utc",     StringType(),  False),
    StructField("amount_rial",       LongType(),    False),
    StructField("card_pan",          StringType(),  True),
    StructField("account_number",    StringType(),  True),
    StructField("national_code",     StringType(),  True),
    StructField("mobile_e164",       StringType(),  True),
    StructField("ip_address",        StringType(),  True),
    StructField("is_carrier_nat",    BooleanType(), True),
    StructField("ip_confidence",     DoubleType(),  True),
    StructField("device_fp_hash",    StringType(),  True),
    StructField("channel",           StringType(),  True),
    StructField("result",            StringType(),  True),
    StructField("is_night",          BooleanType(), True),
])

WINDOWS = ["1 hour", "6 hours", "24 hours", "7 days"]


def create_spark_session():
    return (SparkSession.builder
        .appName("AML_FeatureExtraction")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.streaming.schemaInference", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate())


def read_from_kafka(spark):
    return (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKERS)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
        .select(from_json(col("value").cast("string"), NORMALIZED_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("event_time", expr("CAST(timestamp_utc AS TIMESTAMP)")))


def card_features(df, win):
    """ویژگی‌های کارت: تعداد تراکنش، مجموع مبلغ، تعداد حساب منحصربه‌فرد"""
    return (df
        .withWatermark("event_time", "1 hour")
        .groupBy(col("card_pan"), window("event_time", win))
        .agg(
            count("*").alias("card_tx_count"),
            spark_sum("amount_rial").alias("card_total_amount"),
            avg("amount_rial").alias("card_avg_amount"),
            stddev("amount_rial").alias("card_std_amount"),
            spark_max("amount_rial").alias("card_max_amount"),
            spark_min("amount_rial").alias("card_min_amount"),
            approx_count_distinct("account_number").alias("card_unique_accounts"),
            approx_count_distinct("ip_address").alias("card_unique_ips"),
            approx_count_distinct("device_fp_hash").alias("card_unique_devices"),
            avg(col("is_night").cast("int")).alias("card_night_ratio"),
            (count(when(col("result") == "FAILED", 1)) / count("*")).alias("card_fail_ratio"),
        )
        .withColumn("window_duration", lit(win))
        .withColumn("entity_type", lit("card"))
        .withColumn("entity_id", col("card_pan")))


def account_features(df, win):
    """ویژگی‌های حساب: حجم تراکنش، تعداد کارت، تشخیص structuring"""
    return (df
        .withWatermark("event_time", "1 hour")
        .groupBy(col("account_number"), window("event_time", win))
        .agg(
            count("*").alias("acc_tx_count"),
            spark_sum("amount_rial").alias("acc_total_amount"),
            avg("amount_rial").alias("acc_avg_amount"),
            stddev("amount_rial").alias("acc_std_amount"),
            approx_count_distinct("card_pan").alias("acc_unique_cards"),
            approx_count_distinct("mobile_e164").alias("acc_unique_mobiles"),
            approx_count_distinct("ip_address").alias("acc_unique_ips"),
            avg(col("is_night").cast("int")).alias("acc_night_ratio"),
            # structuring detection: تعداد تراکنش‌های زیر ۵۰ میلیون
            count(when(col("amount_rial") < 50_000_000, 1)).alias("acc_small_tx_count"),
            (count(when(col("result") == "FAILED", 1)) / count("*")).alias("acc_fail_ratio"),
        )
        .withColumn("window_duration", lit(win))
        .withColumn("entity_type", lit("account"))
        .withColumn("entity_id", col("account_number")))


def ip_features(df, win):
    """ویژگی‌های IP: تعداد کارت منحصربه‌فرد — سیگنال اصلی تقلب"""
    return (df
        .withWatermark("event_time", "1 hour")
        .groupBy(col("ip_address"), window("event_time", win))
        .agg(
            count("*").alias("ip_tx_count"),
            approx_count_distinct("card_pan").alias("ip_unique_cards"),
            approx_count_distinct("national_code").alias("ip_unique_persons"),
            approx_count_distinct("device_fp_hash").alias("ip_unique_devices"),
            avg("amount_rial").alias("ip_avg_amount"),
            avg(col("is_night").cast("int")).alias("ip_night_ratio"),
            avg(col("is_carrier_nat").cast("int")).alias("ip_nat_ratio"),
        )
        .withColumn("window_duration", lit(win))
        .withColumn("entity_type", lit("ip"))
        .withColumn("entity_id", col("ip_address")))


def device_features(df, win):
    """ویژگی‌های دستگاه: چند کارت از یه دستگاه؟"""
    return (df
        .withWatermark("event_time", "1 hour")
        .groupBy(col("device_fp_hash"), window("event_time", win))
        .agg(
            count("*").alias("dev_tx_count"),
            approx_count_distinct("card_pan").alias("dev_unique_cards"),
            approx_count_distinct("national_code").alias("dev_unique_persons"),
            approx_count_distinct("account_number").alias("dev_unique_accounts"),
            avg("amount_rial").alias("dev_avg_amount"),
            avg(col("is_night").cast("int")).alias("dev_night_ratio"),
        )
        .withColumn("window_duration", lit(win))
        .withColumn("entity_type", lit("device"))
        .withColumn("entity_id", col("device_fp_hash")))


def to_kafka_stream(df, entity_type, win_label):
    """تبدیل DataFrame به فرمت Kafka JSON"""
    win_clean = win_label.replace(" ", "_")
    return df.select(
        to_json(struct(
            col("entity_type"),
            col("entity_id"),
            col("window_duration"),
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            *[c for c in df.columns if c not in
              ("entity_type", "entity_id", "window_duration", "window")]
        )).alias("value")
    )


def write_to_kafka(df, entity_type, win_label):
    win_clean = win_label.replace(" ", "_")
    checkpoint = f"{CHECKPOINT_BASE}/{entity_type}_{win_clean}"
    return (df.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKERS)
        .option("topic", OUTPUT_TOPIC)
        .option("checkpointLocation", checkpoint)
        .outputMode("update")
        .trigger(processingTime="30 seconds")
        .start())


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    raw = read_from_kafka(spark)
    writers = []

    for win in WINDOWS:
        for fn, name in [
            (card_features,    "card"),
            (account_features, "account"),
            (ip_features,      "ip"),
            (device_features,  "device"),
        ]:
            features_df = fn(raw, win)
            kafka_df    = to_kafka_stream(features_df, name, win)
            writer      = write_to_kafka(kafka_df, name, win)
            writers.append(writer)

    logger.info(f"{len(writers)} stream نوشتن شروع شد")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
