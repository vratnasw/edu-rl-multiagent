"""Cloudflare R2 client. Reads 5 env vars; gracefully reports
no-credential mode so dry-runs and unit tests work without secrets."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_ENV = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                  "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_ENDPOINT_URL")


class R2NotConfiguredError(RuntimeError):
    """Raised when an operation requires R2 credentials and none are set."""


def env_status() -> dict:
    return {k: ("set" if os.environ.get(k) else "missing") for k in REQUIRED_ENV}


def _missing_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.environ.get(k)]


def _client():
    miss = _missing_env()
    if miss:
        raise R2NotConfiguredError(
            f"R2 credentials missing: {miss}. Copy .env.example to .env and fill in.")
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


def bucket_name() -> str:
    if not os.environ.get("R2_BUCKET_NAME"):
        raise R2NotConfiguredError("R2_BUCKET_NAME env var missing")
    return os.environ["R2_BUCKET_NAME"]


# --------------------------------------------------------------------------- #
# Core operations
# --------------------------------------------------------------------------- #

def upload(local_path: str | Path, key: str) -> dict:
    """Upload a single file. Returns {key, size_bytes, duration_s}."""
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(local_path)
    c = _client()
    size = p.stat().st_size
    t0 = time.time()
    c.upload_file(str(p), bucket_name(), key)
    dur = time.time() - t0
    log.info("R2 uploaded %s (%.1f KB) -> %s in %.2fs",
              p.name, size / 1024.0, key, dur)
    return {"key": key, "size_bytes": int(size), "duration_s": round(dur, 3)}


def download(key: str, dest: Optional[Path] = None) -> pd.DataFrame:
    """Download key and return a DataFrame.

    On Windows, s3transfer's mkstemp+rename cycle hits PermissionError
    [WinError 32] when the source temp handle isn't released before the
    rename. We use `get_object` + `BytesIO` to bypass the rename path
    entirely; if the user provides an explicit `dest`, that path is used
    via the slower (but caller-controlled) download_file route."""
    import io
    c = _client()
    if dest is not None:
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
        c.download_file(bucket_name(), key, str(dest))
        return _read_any(dest)
    # In-memory path: avoids the temp-file rename race entirely.
    obj = c.get_object(Bucket=bucket_name(), Key=key)
    body = obj["Body"].read()
    suffix = Path(key).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(body))
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(body))
    if suffix == ".tsv":
        return pd.read_csv(io.BytesIO(body), sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(io.BytesIO(body))
    if suffix == ".json":
        return pd.read_json(io.BytesIO(body))
    raise ValueError(f"unrecognized file type: {suffix}")


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".json":
        return pd.read_json(path)
    raise ValueError(f"unrecognized file type: {suf}")


def exists(key: str) -> Optional[dict]:
    """Return {size, last_modified} or None."""
    try:
        c = _client()
    except R2NotConfiguredError:
        return None
    try:
        h = c.head_object(Bucket=bucket_name(), Key=key)
    except Exception:  # noqa: BLE001
        return None
    return {
        "size": int(h.get("ContentLength", 0)),
        "last_modified": h.get("LastModified", datetime.now(timezone.utc)).isoformat(),
        "etag": h.get("ETag", "").strip('"'),
    }


def list_prefix(prefix: str) -> list[dict]:
    """List all keys under prefix as {key, size, last_modified}."""
    c = _client()
    paginator = c.get_paginator("list_objects_v2")
    out: list[dict] = []
    for page in paginator.paginate(Bucket=bucket_name(), Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            out.append({
                "key": obj["Key"],
                "size": int(obj.get("Size", 0)),
                "last_modified": obj.get("LastModified",
                                              datetime.now(timezone.utc)).isoformat(),
            })
    return out


def make_public(key: str) -> bool:
    """Set ACL to public-read. R2 supports `public-read` via the put_object_acl
    interface; if not, the bucket must be configured with a public-read policy
    upstream — this function then becomes a no-op success."""
    c = _client()
    try:
        c.put_object_acl(Bucket=bucket_name(), Key=key, ACL="public-read")
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("make_public(%s) failed (%s) — bucket must be configured "
                      "with a public-read policy at the bucket level instead", key, e)
        return False


def public_url(key: str) -> str:
    """Construct the public R2 URL for a key. Requires the bucket to have
    public access enabled or a custom domain attached. Format:
    https://pub-<account>.r2.dev/<key>"""
    acct = os.environ.get("R2_ACCOUNT_ID", "")
    return f"https://pub-{acct}.r2.dev/{key}"


# --------------------------------------------------------------------------- #
# Smoke check
# --------------------------------------------------------------------------- #

def smoke_check() -> dict:
    """Quick credential + bucket reachability test."""
    miss = _missing_env()
    if miss:
        return {"ok": False, "missing_env": miss,
                  "advice": "Copy .env.example → .env and fill in"}
    try:
        c = _client()
        c.head_bucket(Bucket=bucket_name())
        return {"ok": True, "bucket": bucket_name(),
                  "endpoint": os.environ["R2_ENDPOINT_URL"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
