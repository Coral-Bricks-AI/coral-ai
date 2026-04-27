"""Per-run gzipped JSONL uploader to Coral Bricks' managed S3 bucket.

One writer per sync run. Gzip buffer per stream in memory; flushed to
`{bucket}/{prefix}/{stream}/part-0000.jsonl.gz` on close().

Credentials are passed in by the caller. The CLI passes STS session
creds whose IAM policy is pinned to exactly `{prefix}/*` for 1 hour.
The ECS supervisor passes None and relies on the task IAM role.

Record shape written: the inner Airbyte Protocol RECORD payload,
verbatim — `{stream, data, emitted_at}` — so downstream consumers parse
the same protocol the source emitted, no Coral envelope layer.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass
from typing import Any

import boto3


@dataclass
class StsCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: str | None = None


class ScopedS3Writer:
    """Buffer raw Airbyte records per stream, gzip, upload on close.

    Resource use is O(records × record-size) until close() — fine for
    typical connector sizes (tens of MB per stream). Rotating to multiple
    parts per stream is future work.

    `sts=None` falls back to the ambient AWS session (e.g. ECS task IAM
    role); pass `StsCredentials` to use temporary session credentials.
    """

    def __init__(
        self,
        *,
        bucket: str,
        key_prefix: str,
        sts: StsCredentials | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.bucket = bucket
        self.key_prefix = key_prefix.rstrip("/")
        self.region = region
        self._sts = sts
        self._buffers: dict[str, io.BytesIO] = {}
        self._gzippers: dict[str, gzip.GzipFile] = {}
        self._record_counts: dict[str, int] = {}
        self._bytes_per_stream: dict[str, int] = {}
        self._closed = False

    def _gzipper_for(self, stream: str) -> gzip.GzipFile:
        existing = self._gzippers.get(stream)
        if existing is not None:
            return existing
        buf = io.BytesIO()
        # mtime=0 → byte-identical gzip output for identical inputs,
        # which makes re-uploads deterministic + less noisy in tests.
        gz = gzip.GzipFile(fileobj=buf, mode="wb", mtime=0)
        self._buffers[stream] = buf
        self._gzippers[stream] = gz
        self._record_counts[stream] = 0
        self._bytes_per_stream[stream] = 0
        return gz

    def write_record(
        self,
        stream: str,
        data: dict[str, Any],
        emitted_at: int | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("ScopedS3Writer.write_record called after close()")
        payload: dict[str, Any] = {"stream": stream, "data": data}
        if emitted_at is not None:
            payload["emitted_at"] = emitted_at
        line = (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        self._gzipper_for(stream).write(line)
        self._record_counts[stream] += 1
        self._bytes_per_stream[stream] += len(line)

    def close(self) -> dict[str, Any]:
        """Flush all streams to S3. Returns an upload summary.

        Shape: `{records, bytes, streams: {stream: {records, bytes, key}}}`.
        Totals go to the backend `/complete` endpoint; the per-stream
        breakdown is shown to the user.
        """
        if self._closed:
            raise RuntimeError("ScopedS3Writer.close() called twice")
        self._closed = True

        if self._sts is not None:
            client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=self._sts.access_key_id,
                aws_secret_access_key=self._sts.secret_access_key,
                aws_session_token=self._sts.session_token,
            )
        else:
            client = boto3.client("s3", region_name=self.region)

        summary: dict[str, Any] = {"records": 0, "bytes": 0, "streams": {}}
        for stream, gz in self._gzippers.items():
            gz.close()
            buf = self._buffers[stream]
            compressed = buf.getvalue()
            key = f"{self.key_prefix}/{stream}/part-0000.jsonl.gz"
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=compressed,
                ContentType="application/gzip",
                ContentEncoding="gzip",
            )
            records = self._record_counts[stream]
            uncompressed_bytes = self._bytes_per_stream[stream]
            summary["records"] += records
            summary["bytes"] += uncompressed_bytes
            summary["streams"][stream] = {
                "records": records,
                "bytes": uncompressed_bytes,
                "key": key,
                "gzippedBytes": len(compressed),
            }
        return summary

    @property
    def stream_count(self) -> int:
        return len(self._gzippers)
