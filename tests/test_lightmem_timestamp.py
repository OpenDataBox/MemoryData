from datetime import datetime
from types import SimpleNamespace

from methods.lightmem.lightmem_adapter import LightMemAdapter


def test_parse_timestamp_accepts_official_locomo_session_format():
    assert LightMemAdapter._parse_timestamp("1:14 pm on 25 May, 2023") == datetime(
        2023,
        5,
        25,
        13,
        14,
    )


def test_close_releases_local_qdrant_client():
    closed = []
    adapter = LightMemAdapter.__new__(LightMemAdapter)
    adapter.lightmem = SimpleNamespace(
        embedding_retriever=SimpleNamespace(
            client=SimpleNamespace(close=lambda: closed.append(True))
        )
    )

    adapter.close()

    assert closed == [True]
