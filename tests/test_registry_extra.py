"""tds.registry 增量能力单测：元数据归一化（体积/摘要）、摘要校验、probe 回填与缓存。

全部用例离线运行（不发起任何网络请求）。
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tds.registry import catalog as C
from tds.registry import downloader as D
from tds.registry import remote as R
from tds.registry.errors import CatalogError


class ParseSiblingsTest(unittest.TestCase):
    def test_parses_size_lfs_and_blob_id(self):
        payload = {
            "siblings": [
                {"rfilename": "a.safetensors", "size": 100, "blobId": "x",
                 "lfs": {"sha256": "AB", "size": 100}},
                {"rfilename": "b.txt", "size": 3},
                {"rfilename": "", "size": 9},
            ]
        }
        out = R._parse_siblings(payload)
        self.assertEqual(set(out), {"a.safetensors", "b.txt"})
        self.assertEqual(
            out["a.safetensors"], {"size": 100, "sha256": "AB", "blob_id": "x"}
        )
        self.assertEqual(out["b.txt"]["sha256"], "")
        self.assertEqual(out["b.txt"]["blob_id"], "")

    def test_size_only_in_lfs(self):
        out = R._parse_siblings(
            {"siblings": [{"rfilename": "c", "lfs": {"size": 7, "sha256": "D"}}]}
        )
        self.assertEqual(out["c"]["size"], 7)
        self.assertEqual(out["c"]["sha256"], "D")

    def test_tolerates_garbage_size(self):
        out = R._parse_siblings({"siblings": [{"rfilename": "d", "size": "N/A"}]})
        self.assertEqual(out["d"]["size"], 0)

    def test_empty_payload(self):
        self.assertEqual(R._parse_siblings({}), {})
        self.assertEqual(R._parse_siblings({"siblings": None}), {})


class ModelApiUrlTest(unittest.TestCase):
    def test_blobs_flag_default_on(self):
        self.assertIn("blobs=true", R.model_api_url("owner/repo"))

    def test_blobs_flag_can_be_disabled(self):
        self.assertNotIn("blobs", R.model_api_url("owner/repo", with_blobs=False))


class VerifyFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tds_reg_")
        self.path = os.path.join(self.tmp, "x.bin")
        with open(self.path, "wb") as fh:
            fh.write(b"hello")
        self.digest = hashlib.sha256(b"hello").hexdigest()

    def test_matching_digest(self):
        self.assertTrue(D.verify_file(self.path, self.digest))

    def test_digest_case_insensitive(self):
        self.assertTrue(D.verify_file(self.path, self.digest.upper()))

    def test_mismatch_and_blank(self):
        self.assertFalse(D.verify_file(self.path, "0" * 64))
        self.assertFalse(D.verify_file(self.path, ""))

    def test_missing_file(self):
        self.assertFalse(D.verify_file(os.path.join(self.tmp, "nope"), self.digest))


class WithSizesTest(unittest.TestCase):
    def test_checksums_are_merged_not_replaced(self):
        spec = C.ModelSpec(key="k", repo="r", files=("f",),
                           sizes={"f": 1}, checksums={"f": "aa"})
        out = spec.with_sizes({"f": 2}, checksums={"f": "bb", "g": ""})
        self.assertEqual(out.sizes, {"f": 2})
        self.assertEqual(out.checksums, {"f": "bb"})
        self.assertEqual(spec.checksums, {"f": "aa"})  # 原实例不变

    def test_checksums_optional(self):
        spec = C.ModelSpec(key="k", repo="r", files=("f",))
        out = spec.with_sizes({"f": 9}, verified=False)
        self.assertEqual(out.checksums, {})
        self.assertFalse(out.verified)


class ProbeEntryTest(unittest.TestCase):
    def setUp(self):
        C.clear_remote_cache()
        self.calls = []
        self._origin = C.list_remote_files
        self.fake = {"clip_l.safetensors": {"size": 123, "sha256": "abc", "blob_id": "z"}}

        def fake_list(repo, endpoint=None):
            self.calls.append(repo)
            return self.fake

        C.list_remote_files = fake_list

    def tearDown(self):
        C.list_remote_files = self._origin
        C.clear_remote_cache()

    def test_backfills_size_and_checksum(self):
        spec = C.get("flux.text_encoder.clip_l")
        out = C.probe_entry(spec)
        self.assertTrue(out.verified)
        self.assertEqual(out.sizes["clip_l.safetensors"], 123)
        self.assertEqual(out.checksums["clip_l.safetensors"], "abc")
        self.assertFalse(spec.verified)  # 原条目未被就地修改

    def test_remote_listing_cached_per_repo(self):
        C.probe_entry(C.get("flux.text_encoder.clip_l"))
        C.probe_entry(C.get("flux.text_encoder.clip_l"))
        self.assertEqual(len(self.calls), 1)

    def test_cache_clear_forces_refetch(self):
        C.probe_entry(C.get("flux.text_encoder.clip_l"))
        C.clear_remote_cache()
        C.probe_entry(C.get("flux.text_encoder.clip_l"))
        self.assertEqual(len(self.calls), 2)

    def test_missing_declared_file_raises(self):
        spec = C.ModelSpec(key="k", repo="r", files=("absent.safetensors",))
        C.clear_remote_cache()
        with self.assertRaises(CatalogError):
            C.probe_entry(spec)


if __name__ == "__main__":
    unittest.main()
