"""联网冒烟:默认跳过,只有 ASHARE_LIVE=1 才跑(CI 与日常测试不依赖网络)。"""
from __future__ import annotations

import os

import pytest

from ashare_forge.data.baostock_source import BaostockSource

pytestmark = pytest.mark.skipif(os.environ.get("ASHARE_LIVE") != "1",
                                reason="需 ASHARE_LIVE=1 才跑联网冒烟")


def test_live_daily_and_index():
    with BaostockSource() as src:
        d = src.fetch_daily("sz.000001", "2026-09-01", "2026-09-19")
        assert len(d) > 0
        for c in ("date", "close", "turn", "tradestatus", "isST", "peTTM"):
            assert c in d.columns
        idx = src.fetch_index_daily("sh.000905", "2026-09-01", "2026-09-19")
        assert len(idx) > 0 and "close" in idx.columns
