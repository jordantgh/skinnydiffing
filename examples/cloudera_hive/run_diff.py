"""
Example runner for skinnydiffing using a custom Hive/S3 downloader.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

import boto3
import polars as pl
from pyspark.sql import SparkSession

from skinnydiffing import run_config

logger = logging.getLogger(__name__)


def load_hive(
    table: str,
    cache_dir: str | None = None,
    *,
    spark: SparkSession,
    ssl_cert: str | None = None,
) -> pl.LazyFrame:
    """
    Uses Spark to find a Hive table's S3 path, downloads the raw Parquet files
    locally, and returns a native Polars LazyFrame.
    """
    if cache_dir is None:
        cache_dir = f"/tmp/hive_{table}"

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    rows = spark.sql(f"DESCRIBE FORMATTED {table}").collect()
    location = next((r[1].strip() for r in rows if r[0] and "Location" in r[0]), None)
    if not location:
        raise RuntimeError(f"Could not resolve S3 location for '{table}'")

    parsed = urlparse(str(location).replace("s3a://", "s3://"))
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    client = boto3.client("s3")
    try:
        import raz_client

        if ssl_cert:
            raz_client.configure_ranger_raz(client, ssl_file=ssl_cert)
    except ImportError:
        pass

    logger.info("Downloading %s -> %s", location, cache_path)
    n_files = 0
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".parquet"):
                continue
            n_files += 1
            relative = (
                obj["Key"][len(prefix) :].lstrip("/")
                if obj["Key"].startswith(prefix)
                else Path(obj["Key"]).name
            )
            dest = cache_path / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(dest))

    if not n_files:
        raise FileNotFoundError(f"No parquet files found at {location}")

    return pl.scan_parquet(str(cache_path / "**/*.parquet"), hive_partitioning=True)


def main():
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        level=logging.INFO,
    )

    logging.info("Initialising SparkSession...")
    spark = (
        SparkSession.builder.appName("TableDiff_Hive_S3")
        .enableHiveSupport()
        .getOrCreate()
    )

    current_dir = Path(__file__).parent
    config_path = current_dir / "config.yml"

    logging.info("Loading config from %s", config_path)

    loaders = {
        "hive": lambda conf: load_hive(
            **conf, spark=spark, ssl_cert="/etc/pki/tls/certs/ca-bundle.crt"
        )
    }

    results = run_config(
        config_path,
        loaders=loaders,
    )

    logging.info("Completed %d comparison(s).", len(results))
    spark.stop()


if __name__ == "__main__":
    main()
