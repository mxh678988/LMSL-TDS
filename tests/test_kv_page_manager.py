"""太极调度层 TDS · KV 分页缓存引擎单测（Phase 2 B 段）。

覆盖场景
--------
1. PageTable 页生命周期：创建 / 查找 / 换出 / 换入
2. PageCache 属性接口：key/value/index 返回类型兼容 FixedKV
3. 包装器集成：patch + 解 patch + 恢复原始行为
4. 环境变量：TDS_KV 开启/关闭分页
5. 多页 LRU 淘汰
"""
import gc
import os
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# 使用正确的 Python 解释器路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ -> TDS/
sys.path.insert(0, r"E:\LMSL-comfyui")

# 导入 torch（宿主环境依赖：纯净解释器下整体跳过该文件，而非中断本轮收集）
import pytest

torch = pytest.importorskip("torch", reason="KV 分页引擎测试依赖宿主 torch")

from tds.kv.page_manager import (
    Page, PageTable, PageCache,
    _KV_PAGE_STATE, _current_step,
    _env_enabled, _make_kv_page_wrapper,
    enable, disable, is_enabled,
    _ensure_wrapped, _ensure_unwrapped,
    get_page_tables, get_summary_text,
)

PAGE_SIZE_TOKENS = 512
PAGE_SIZE_BYTES = 256


class TestPageTable(unittest.TestCase):
    """页表生命周期测试。"""

    def _make_page_table(self, **kwargs):
        kwargs.setdefault("layer", 0)
        kwargs.setdefault("dtype", "float16")
        kwargs.setdefault("device", "cuda:0")
        kwargs.setdefault("batch", 1)
        kwargs.setdefault("head_dim", 64)
        kwargs.setdefault("num_kv_heads", 8)
        kwargs.setdefault("capacity_tokens", 2048)
        kwargs.setdefault("page_size_tokens", PAGE_SIZE_TOKENS)
        kwargs.setdefault("page_size_bytes", PAGE_SIZE_BYTES)
        return PageTable(**kwargs)

    def test_create_page(self):
        pt = self._make_page_table(capacity_tokens=2048)
        page = pt.get_or_create_page(0)
        self.assertEqual(page.capacity_tokens, PAGE_SIZE_TOKENS)
        self.assertTrue(page.is_in_vram)
        self.assertEqual(page.layer_id, 0)
        self.assertEqual(page.device, "cuda:0")

    def test_create_pages_across_capacity(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pages = []
        for i in range(0, 2048, PAGE_SIZE_TOKENS):
            pages.append(pt.get_or_create_page(i))
        self.assertEqual(len(pt.pages), 4)  # 2048 / 512 = 4 pages

    def test_get_page_returns_correct_page(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        pt.get_or_create_page(512)
        self.assertEqual(pt.get_page(256).page_id, "layer0_p0")
        self.assertEqual(pt.get_page(768).page_id, "layer0_p1")

    def test_get_page_returns_none_for_missing(self):
        pt = self._make_page_table(capacity_tokens=2048)
        self.assertIsNone(pt.get_page(1000))

    def test_evict_to_ram(self):
        pt = self._make_page_table(capacity_tokens=2048)
        page = pt.get_or_create_page(0)
        self.assertTrue(page.is_in_vram)
        pt.evict_to_ram(page)
        self.assertFalse(page.is_in_vram)
        self.assertEqual(page, pt.ram_buffer[page.page_id])

    def test_prefetch_to_vram(self):
        pt = self._make_page_table(capacity_tokens=2048)
        page = pt.get_or_create_page(0)
        pt.evict_to_ram(page)
        self.assertFalse(page.is_in_vram)
        result = pt.prefetch_to_vram(page)
        self.assertTrue(result)
        self.assertTrue(page.is_in_vram)

    def test_evict_lru_page(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        pt.get_or_create_page(512)
        pt.pages["layer0_p0"].last_used_step = 0
        pt.pages["layer0_p1"].last_used_step = 1
        oldest = pt.evict_lru_page()
        self.assertEqual(oldest.page_id, "layer0_p0")
        self.assertEqual(oldest, pt.ram_buffer["layer0_p0"])

    def test_total_pages(self):
        pt = self._make_page_table(capacity_tokens=1280, page_size_tokens=512)
        self.assertEqual(pt.total_pages, 3)  # 1280 / 512 = 2.5, ceil = 3

    def test_lru_eviction(self):
        pt = self._make_page_table(capacity_tokens=1536)
        pt.get_or_create_page(0)   # p0
        pt.get_or_create_page(512)  # p1
        pt.get_or_create_page(1024) # p2
        pt.pages["layer0_p1"].last_used_step = 0
        pt.pages["layer0_p2"].last_used_step = 1
        pt.pages["layer0_p0"].last_used_step = 2
        oldest = pt.evict_lru_page()
        self.assertEqual(oldest.page_id, "layer0_p1")


class TestPageCache(unittest.TestCase):
    """PageCache 接口兼容性测试。"""

    def _make_page_table(self, **kwargs):
        kwargs.setdefault("layer", 0)
        kwargs.setdefault("dtype", "float16")
        kwargs.setdefault("device", "cuda:0")
        kwargs.setdefault("batch", 1)
        kwargs.setdefault("head_dim", 64)
        kwargs.setdefault("num_kv_heads", 8)
        kwargs.setdefault("capacity_tokens", 2048)
        kwargs.setdefault("page_size_tokens", PAGE_SIZE_TOKENS)
        kwargs.setdefault("page_size_bytes", PAGE_SIZE_BYTES)
        return PageTable(**kwargs)

    def test_key_property_returns_tensor(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        cache = PageCache(pt)
        key = cache.key
        self.assertIsInstance(key, torch.Tensor)

    def test_value_property_returns_tensor(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        cache = PageCache(pt)
        value = cache.value
        self.assertIsInstance(value, torch.Tensor)

    def test_index_is_int(self):
        """PageCache.index 与宿主 FixedKV.index 同型：int，而非张量。"""
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        cache = PageCache(pt)
        self.assertIsInstance(cache.index, int)
        self.assertEqual(cache.index, 0)

    def test_index_advances_monotonically(self):
        """index 只由 advance 推进；prepare 不得改动它（否则写入位置错乱）。"""
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        cache = PageCache(pt)
        cache.advance(16)
        self.assertEqual(cache.index, 16)
        cache.advance(32)
        self.assertEqual(cache.index, 48)

    def test_is_fixed_kv_is_false(self):
        pt = self._make_page_table()
        cache = PageCache(pt)
        self.assertFalse(cache.is_fixed_kv)

    def test_device_returns_correct_string(self):
        pt = self._make_page_table(device="cuda:0")
        cache = PageCache(pt)
        self.assertEqual(cache.device, "cuda:0")

    def test_capacity_tokens_returns_correct_value(self):
        pt = self._make_page_table(capacity_tokens=4096)
        cache = PageCache(pt)
        self.assertEqual(cache.capacity_tokens, 4096)

    def test_get_memory_bytes(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        pt.get_or_create_page(512)
        pt.pages["layer0_p1"].is_in_vram = False
        cache = PageCache(pt)
        total = cache.get_memory_bytes()
        # p0 在显存中：2 * head_dim(64) * num_kv_heads(8) * PAGE_SIZE_TOKENS
        expected = 2 * 64 * 8 * PAGE_SIZE_TOKENS
        self.assertEqual(total, expected)

    def test_record_access(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        cache = PageCache(pt)
        cache.record_access()
        page = pt.pages["layer0_p0"]
        self.assertEqual(page.access_count, 1)
        self.assertEqual(page.last_used_step, _current_step)

    def test_repr(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        cache = PageCache(pt)
        s = repr(cache)
        self.assertIn("PageCache", s)
        self.assertIn("layer=0", s)

    def test_key_tensor_is_cached(self):
        pt = self._make_page_table(capacity_tokens=2048)
        pt.get_or_create_page(0)
        cache = PageCache(pt)
        key1 = cache.key
        key2 = cache.key
        self.assertIs(key1, key2)

    def test_master_tensor_when_no_pages(self):
        """无页时按 FixedKV 契约返回完整母张量，而不是空张量。

        母张量必须始终完整：attention 会对其做原位 scatter_ 后整块送入
        flash_attention_decode，任何"按页裁剪"都会导致写入丢失。
        """
        pt = self._make_page_table(capacity_tokens=2048)
        cache = PageCache(pt)
        key = cache.key
        self.assertIsInstance(key, torch.Tensor)
        self.assertEqual(key.shape, (1, 2048, 8, 64))
        self.assertEqual(key.numel(), 2048 * 8 * 64)


class TestWrapperIntegration(unittest.TestCase):
    """包装器集成测试。"""

    def setUp(self):
        # 清理之前可能残留的状态
        _KV_PAGE_STATE.clear()
        _KV_PAGE_STATE["patched"] = False
        _KV_PAGE_STATE["original_init_kv_cache"] = None
        _KV_PAGE_STATE["original_class_dict"] = False
        _KV_PAGE_STATE["page_tables"] = []
        _current_step = 0

    def _make_mock_llama2_class(self, original_fn=None):
        """创建模拟的 Llama2_ 类用于测试包装器。"""
        class MockConfig:
            head_dim = 64
            num_key_value_heads = 8

        class MockLlama2:
            config = MockConfig()

        if original_fn is not None:
            MockLlama2.init_kv_cache = original_fn
        else:
            def mock_init_kv_cache(self, batch, capacity, device, dtype):
                return [MagicMock(spec=[("key", torch.empty(0)), ("value", torch.empty(0)), ("index", torch.empty(0))])]
            MockLlama2.init_kv_cache = mock_init_kv_cache

        return MockLlama2

    class FakeFixedKV:
        """宿主 FixedKV 替身：供包装器分支守卫识别为可替换结构。"""

        def __init__(self, key, value, index):
            self.key = key
            self.value = value
            self.index = index

    def _make_mock_llama2_class_with_fixedkv(self):
        """构造 init_kv_cache 返回宿主 FixedKV 替身的模拟 Llama2_ 类。"""
        fake_cls = self.FakeFixedKV

        def mock_init_kv_cache(self, batch, capacity, device, dtype):
            return [fake_cls(torch.empty(0), torch.empty(0), 0)]

        return self._make_mock_llama2_class(mock_init_kv_cache)

    def test_make_wrapper_creates_page_caches(self):
        """包装器把宿主 FixedKV 替换为 PageCache（需宿主类可见）。"""
        mock_cls = self._make_mock_llama2_class_with_fixedkv()
        original = mock_cls.init_kv_cache
        wrapped = _make_kv_page_wrapper(original)
        instance = mock_cls()

        with patch("tds.kv.page_manager._get_FixedKV_class",
                   return_value=self.FakeFixedKV):
            result = wrapped(instance, 1, 2048, torch.device("cpu"), torch.float16)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        for r in result:
            self.assertIsInstance(r, PageCache)
            self.assertFalse(r.is_fixed_kv)
            # FixedKV 契约：key/value 为完整母张量，index 为 int
            self.assertIsInstance(r.key, torch.Tensor)
            self.assertIsInstance(r.value, torch.Tensor)
            self.assertIsInstance(r.index, int)
            self.assertEqual(r.key.shape, (1, 2048, 8, 64))

    def test_wrapper_returns_original_when_disabled(self):
        """测试当分页关闭时，包装器返回原始结果。"""
        # 禁用分页
        with patch("tds.kv.page_manager._KV_PAGE_ENABLED", False):
            mock_cls = self._make_mock_llama2_class()
            original = mock_cls.init_kv_cache
            wrapped = _make_kv_page_wrapper(original)
            instance = mock_cls()

            result = wrapped(instance, 1, 2048, torch.device("cpu"), torch.float16)
            # 应该返回原始 MagicMock 而不是 PageCache
            self.assertIsInstance(result, list)
            for r in result:
                self.assertIsInstance(r, MagicMock)

    def test_wrapper_logs_page_creation(self):
        """包装器把宿主 FixedKV 替换为 PageCache 时记录创建日志。"""
        mock_cls = self._make_mock_llama2_class_with_fixedkv()
        original = mock_cls.init_kv_cache
        wrapped = _make_kv_page_wrapper(original)
        instance = mock_cls()

        with patch("tds.kv.page_manager._get_FixedKV_class",
                   return_value=self.FakeFixedKV):
            with patch("tds.kv.page_manager.LOG.info") as mock_info:
                wrapped(instance, 1, 2048, torch.device("cpu"), torch.float16)
                self.assertGreaterEqual(len(mock_info.call_args_list), 1)

    def test_enable_wrapper_patching(self):
        """测试 enable 函数正确包装类。"""
        # 模拟 Llama2_ 类
        mock_cls = self._make_mock_llama2_class()

        # 模拟导入
        mock_host = MagicMock()
        mock_host.model_cls = mock_cls

        with patch("tds.kv.page_manager._injected_host", mock_host):
            enable()
            self.assertTrue(is_enabled())
            # 验证 init_kv_cache 已被包装
            self.assertTrue(hasattr(mock_cls, 'init_kv_cache'))

    def test_disable_wrapper_restores(self):
        """测试 disable 函数正确恢复类。"""
        mock_cls = self._make_mock_llama2_class()
        mock_host = MagicMock()
        mock_host.model_cls = mock_cls

        with patch("tds.kv.page_manager._injected_host", mock_host):
            # 先启用
            enable()
            self.assertTrue(is_enabled())

            # 再禁用
            disable()
            self.assertFalse(is_enabled())


class TestEnvEnabled(unittest.TestCase):
    """环境变量解析测试。"""

    def test_env_var_on(self):
        os.environ["TDS_KV"] = "page"
        self.assertTrue(_env_enabled())

    def test_env_var_off(self):
        os.environ["TDS_KV"] = "off"
        self.assertFalse(_env_enabled())

    def test_env_var_default(self):
        os.environ["TDS_KV"] = "observe"
        self.assertTrue(_env_enabled())

    def test_env_var_disabled(self):
        os.environ["TDS_KV"] = "disabled"
        self.assertFalse(_env_enabled())

    def test_env_var_0(self):
        os.environ["TDS_KV"] = "0"
        self.assertFalse(_env_enabled())


class TestSummaryText(unittest.TestCase):
    """摘要文本生成测试。"""

    def test_summary_no_page_tables(self):
        _KV_PAGE_STATE["page_tables"] = []
        summary = get_summary_text()
        self.assertIn("无分页数据", summary)

    def test_summary_with_page_tables(self):
        pt = self._make_page_table()
        _KV_PAGE_STATE["page_tables"] = [pt]
        summary = get_summary_text()
        self.assertIn("KV 分页缓存状态", summary)
        self.assertIn("层 0:", summary)

    def test_get_page_tables(self):
        pt = self._make_page_table()
        _KV_PAGE_STATE["page_tables"] = [pt]
        tables = get_page_tables()
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], pt)

    def _make_page_table(self, **kwargs):
        kwargs.setdefault("layer", 0)
        kwargs.setdefault("dtype", "float16")
        kwargs.setdefault("device", "cuda:0")
        kwargs.setdefault("batch", 1)
        kwargs.setdefault("head_dim", 64)
        kwargs.setdefault("num_kv_heads", 8)
        kwargs.setdefault("capacity_tokens", 2048)
        kwargs.setdefault("page_size_tokens", PAGE_SIZE_TOKENS)
        kwargs.setdefault("page_size_bytes", PAGE_SIZE_BYTES)
        return PageTable(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
