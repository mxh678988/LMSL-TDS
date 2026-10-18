"""tds.registry 离线单元测试（不触网）。

覆盖：硬件画像一致性、显存估算与档位判定、目录契约与外部注入、
幂等跳过逻辑、端点解析优先级。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tds.registry import catalog as C  # noqa: E402
from tds.registry import estimate as E  # noqa: E402
from tds.registry import planner as P  # noqa: E402
from tds.registry.downloader import _parse_total, download_file  # noqa: E402
from tds.registry.errors import CatalogError, IntegrityError  # noqa: E402
from tds.registry.hardware import probe_hardware  # noqa: E402
from tds.registry.remote import DEFAULT_ENDPOINT, resolve_endpoint  # noqa: E402


class TestHardware(unittest.TestCase):
    def test_profile_consistency(self):
        profile = probe_hardware(refresh=True)
        self.assertEqual(profile.has_gpu, bool(profile.devices))
        self.assertGreaterEqual(profile.cpu_count, 1)
        self.assertTrue(profile.summary())
        if profile.has_gpu:
            primary = profile.primary_device
            self.assertEqual(profile.total_vram_mb, primary.total_mb)
            self.assertEqual(profile.free_vram_mb, primary.free_mb)
            self.assertGreater(profile.total_vram_mb, 0)

    def test_primary_picks_largest(self):
        from tds.registry.hardware import DeviceProfile, HardwareProfile

        profile = HardwareProfile(
            devices=[
                DeviceProfile(index=0, name="small", total_mb=8192, free_mb=100),
                DeviceProfile(index=1, name="big", total_mb=24576, free_mb=200),
            ]
        )
        self.assertEqual(profile.primary_device.name, "big")
        self.assertEqual(profile.total_vram_mb, 24576)


class TestEstimate(unittest.TestCase):
    def test_weights_from_size(self):
        self.assertAlmostEqual(E.weights_mb_from_size(1024 * 1024), 1.0, places=6)
        self.assertAlmostEqual(E.weights_mb_from_size(0), 0.0, places=6)

    def test_weights_from_params(self):
        # 7B × 16bit = 14e9 B ≈ 13351.6 MB
        got = E.weights_mb_from_params(7.0, 16)
        self.assertAlmostEqual(got, 7e9 * 2 / (1024 ** 2), places=2)
        self.assertEqual(E.weights_mb_from_params(0.0, 16), 0.0)
        self.assertEqual(E.weights_mb_from_params(7.0, 0), 0.0)

    def test_kv_requires_all_dims(self):
        self.assertEqual(E.estimate_kv_mb(0, 8, 128, 4096), 0.0)
        self.assertEqual(E.estimate_kv_mb(32, 0, 128, 4096), 0.0)
        self.assertEqual(E.estimate_kv_mb(32, 8, 0, 4096), 0.0)
        self.assertEqual(E.estimate_kv_mb(32, 8, 128, 0), 0.0)
        self.assertGreater(E.estimate_kv_mb(32, 8, 128, 4096), 0.0)

    def _spec(self, **kw):
        base = dict(key="t", repo="r", files=("a.safetensors",))
        base.update(kw)
        return C.ModelSpec(**base)

    def test_footprint_prefers_file_size(self):
        spec = self._spec(sizes={"a.safetensors": 1024 * 1024})
        fp = E.estimate_footprint(spec)
        self.assertEqual(fp.basis, "file_size")
        self.assertAlmostEqual(fp.weights_mb, 1.0, places=4)
        self.assertAlmostEqual(fp.workspace_mb, 0.15, places=4)
        self.assertTrue(fp.complete)

    def test_footprint_unknown_without_any_data(self):
        fp = E.estimate_footprint(self._spec())
        self.assertEqual(fp.basis, "unknown")
        self.assertEqual(fp.total_mb, 0.0)
        verdict = E.judge(fp, 10000)
        self.assertEqual(verdict.level, "unknown")
        self.assertFalse(verdict.fits)

    def test_judge_levels(self):
        ok = E.judge(E.Footprint(total_mb=100.0, basis="file_size", complete=True), 1000.0)
        deep = E.judge(E.Footprint(total_mb=950.0, basis="file_size", complete=True), 1000.0)
        over = E.judge(E.Footprint(total_mb=1200.0, basis="file_size", complete=True), 1000.0)
        self.assertEqual((ok.level, ok.fits), ("ok", True))
        self.assertEqual((deep.level, deep.fits), ("tight", True))
        self.assertEqual((over.level, over.fits), ("oom", False))
        self.assertLess(ok.headroom_mb, 1000.0)

    def test_judge_zero_budget(self):
        verdict = E.judge(E.Footprint(total_mb=10.0, basis="file_size"), 0.0)
        self.assertEqual(verdict.level, "unknown")

    def test_kv_counted_for_text_encoder(self):
        spec = self._spec(
            domain="text_encoder",
            sizes={"a.safetensors": 1024 * 1024},
            layers=24, kv_heads=8, head_dim=64,
        )
        fp = E.estimate_footprint(spec, ctx_tokens=512)
        self.assertGreater(fp.kv_mb, 0.0)
        self.assertGreater(fp.total_mb, fp.weights_mb)

    def test_partial_size_marks_incomplete(self):
        spec = self._spec(files=("a", "b"), sizes={"a": 1024})
        fp = E.estimate_footprint(spec)
        self.assertEqual(fp.basis, "file_size_partial")
        self.assertFalse(fp.complete)


class TestCatalog(unittest.TestCase):
    def test_builtin_loaded_and_unverified(self):
        rows = C.list_entries()
        self.assertTrue(rows)
        self.assertTrue(all(not s.verified for s in rows))
        self.assertEqual(C.list_entries(domain="vae"), [s for s in rows if s.domain == "vae"])

    def test_get_missing_raises(self):
        with self.assertRaises(CatalogError):
            C.get("no.such.entry")

    def test_invalid_domain(self):
        with self.assertRaises(CatalogError):
            C.ModelSpec(key="k", repo="r", files=("f",), domain="bogus")

    def test_empty_key_or_files(self):
        with self.assertRaises(CatalogError):
            C.ModelSpec(key="", repo="r", files=("f",))
        with self.assertRaises(CatalogError):
            C.ModelSpec(key="k", repo="r", files=())

    def test_from_dict_rejects_unknown_field(self):
        with self.assertRaises(CatalogError):
            C.ModelSpec.from_dict({"key": "k", "repo": "r", "files": ["f"], "typo": 1})

    def test_register_duplicate_requires_override(self):
        spec = C.ModelSpec(key="tmp.dup", repo="r", files=("f",))
        C.register(spec)
        with self.assertRaises(CatalogError):
            C.register(spec)
        replaced = C.register(spec, override=True)
        self.assertEqual(replaced.key, "tmp.dup")

    def test_with_sizes_is_pure(self):
        spec = C.ModelSpec(key="k", repo="r", files=("f",))
        new = spec.with_sizes({"f": 123})
        self.assertEqual(new.sizes["f"], 123)
        self.assertEqual(spec.sizes, {})
        self.assertTrue(new.verified)
        self.assertEqual(new.total_size_bytes, 123)

    def test_external_catalog_injection(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cat.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"models": [{"key": "mine.vae", "repo": "me/vae", "files": ["ae.safetensors"],
                                 "domain": "vae"}]},
                    fh,
                )
            merged = C.load_catalog(path=path)
            self.assertIn("mine.vae", merged)
            self.assertIn("flux.vae", merged)

    def test_external_catalog_missing_file(self):
        with self.assertRaises(CatalogError):
            C.load_catalog(path=os.path.join(tempfile.gettempdir(), "definitely_absent.json"))


class TestPlanner(unittest.TestCase):
    class _Cfg:
        vram_budget_mb = 12000
        reserve_mb = 2000

    def test_budget_uses_config_when_no_live(self):
        self.assertAlmostEqual(P.budget_mb(config=self._Cfg(), use_live=False), 10000.0, places=3)

    def test_budget_live_never_exceeds_config(self):
        got = P.budget_mb(config=self._Cfg(), use_live=True)
        self.assertLessEqual(got, 10000.0)
        self.assertGreaterEqual(got, 0.0)

    def test_plan_returns_unknown_without_sizes(self):
        result = P.plan(C.get("flux.vae"), config=self._Cfg())
        self.assertEqual(result["verdict"]["level"], "unknown")
        self.assertEqual(result["weight_basis"], "unknown")

    def test_rank_sorted_and_filtered(self):
        rows = P.rank(domain="text_encoder", config=self._Cfg())
        self.assertTrue(rows)
        self.assertTrue(all(r["domain"] == "text_encoder" for r in rows))
        totals = [r["footprint"].get("total_mb", 0.0) for r in rows]
        self.assertEqual(totals, sorted(totals))


class TestDownloaderOffline(unittest.TestCase):
    def test_parse_total(self):
        self.assertEqual(_parse_total("bytes 100-999/1000"), 1000)
        self.assertEqual(_parse_total("bytes 0-1/*"), 0)
        self.assertEqual(_parse_total("garbage"), 0)

    def test_existing_file_skipped_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = "clip_l.safetensors"
            target = os.path.join(tmp, name)
            with open(target, "wb") as fh:
                fh.write(b"x" * 16)
            events = []
            got = download_file(
                "no/such-repo", name, tmp, expected_size=16,
                progress=lambda n, d, t: events.append((n, d, t)),
            )
            self.assertEqual(got, target)
            self.assertEqual(events, [(name, 16, 16)])
            self.assertFalse(os.path.exists(target + ".part"))


class TestRemoteOffline(unittest.TestCase):
    def test_endpoint_precedence(self):
        old = os.environ.pop("TDS_HF_ENDPOINT", None)
        try:
            self.assertEqual(resolve_endpoint(), DEFAULT_ENDPOINT)
            self.assertEqual(resolve_endpoint("https://example.org/"), "https://example.org")
            os.environ["TDS_HF_ENDPOINT"] = "https://mirror.internal/"
            self.assertEqual(resolve_endpoint("https://example.org"), "https://mirror.internal")
        finally:
            os.environ.pop("TDS_HF_ENDPOINT", None)
            if old is not None:
                os.environ["TDS_HF_ENDPOINT"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
