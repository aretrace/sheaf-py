import base64
import datetime as _dt
import os
import uuid


def pseudlid(from_timestamp: float | int | _dt.datetime | None = None) -> str:
    CROCKFORD = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ"

    if from_timestamp is None:
        raw = uuid.uuid7().bytes
    else:
        if isinstance(from_timestamp, _dt.datetime):
            ts_sec = from_timestamp.timestamp()
        else:
            ts_sec = float(from_timestamp)

        ts_ms = int(ts_sec * 1000)

        b = bytearray(os.urandom(16))
        b[0] = (ts_ms >> 40) & 0xFF
        b[1] = (ts_ms >> 32) & 0xFF
        b[2] = (ts_ms >> 24) & 0xFF
        b[3] = (ts_ms >> 16) & 0xFF
        b[4] = (ts_ms >> 8) & 0xFF
        b[5] = ts_ms & 0xFF
        b[6] = (b[6] & 0x0F) | 0x70
        b[8] = (b[8] & 0x3F) | 0x80
        raw = bytes(b)

    return (
        base64.b32encode(raw)
        .rstrip(b"=")
        .translate(bytes.maketrans(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", CROCKFORD))
        .decode()
    )
