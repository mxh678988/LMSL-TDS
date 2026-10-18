"""tds.security 签名校验模块测试。

覆盖：验签通过、篡改拒绝、公钥指纹防替换、现有发布 wheel 实测校验。
"""
import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from tds.security import (
    PUBLIC_KEY_FINGERPRINT,
    load_public_key,
    parse_sig_file,
    sha256_bytes,
    sha256_file,
    verify_signature,
    verify_wheel,
)

# 真实发布资产（相对本仓库根目录）
REPO_ROOT = Path(__file__).resolve().parents[1]
WHEEL = REPO_ROOT / "release" / "tds_core-0.2.0-py3-none-any.whl"
SIG = REPO_ROOT / "release" / "tds_core-0.2.0-py3-none-any.whl.sig"
PRIVATE_KEY = REPO_ROOT / "release" / ".sign" / "tds_release_ed25519_private.pem"


class TestVerifySignature(unittest.TestCase):
    def test_roundtrip_ok(self):
        """官方私钥签名的数据应验签通过。"""
        sk = Ed25519PrivateKey.generate()
        data = b"tds security test payload"
        sig = sk.sign(data)
        pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        # 用内置公钥逻辑走一遍（指纹校验会拒绝非官方公钥，故此处验证签名算法逻辑）
        key = Ed25519PublicKey.from_public_bytes(pub)
        key.verify(sig, data)  # 不抛异常即通过
        self.assertTrue(True)

    def test_tamper_rejected(self):
        """篡改数据后验签必须失败。"""
        sk = Ed25519PrivateKey.generate()
        data = b"tds security test payload"
        sig = sk.sign(data)
        tampered = b"tds security test payloax"
        from cryptography.exceptions import InvalidSignature

        key = Ed25519PublicKey.from_public_bytes(
            sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        )
        with self.assertRaises(InvalidSignature):
            key.verify(sig, tampered)

    def test_fingerprint_binding(self):
        """非官方公钥必须被指纹校验拒绝。"""
        sk = Ed25519PrivateKey.generate()
        pem = sk.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        fp = sha256_bytes(pem)
        self.assertNotEqual(fp, PUBLIC_KEY_FINGERPRINT)
        with self.assertRaises(ValueError):
            load_public_key(pem)

    def test_parse_sig_file(self):
        """解析现有 .sig 文本格式。"""
        info = parse_sig_file(
            "tds-core 0.2.0 wheel 自有签名\n算法: Ed25519\n"
            "SHA256: FFEDE73800E50432027CB081B422BF2F825BA995A8E9928B2053972371649F00\n"
            "SIGBASE64: 3WCclQVwSQCYwIDdEqC9FGM01HOCYuwJSENfkVt1sx49v7HFBfkifnTHpmq7YJzqaPARvSWQzPkJtepHMVQgCA==\n"
        )
        self.assertEqual(
            info["sha256"],
            "FFEDE73800E50432027CB081B422BF2F825BA995A8E9928B2053972371649F00",
        )
        self.assertTrue(info["signature_b64"])


@unittest.skipUnless(WHEEL.exists() and SIG.exists(), "release 资产不存在，跳过实包校验")
class TestRealWheel(unittest.TestCase):
    def test_verify_existing_wheel(self):
        """现有发布 wheel 应通过三重校验。"""
        result = verify_wheel(WHEEL, SIG)
        self.assertTrue(result["sha256_ok"], "SHA256 应一致")
        self.assertTrue(result["signature_ok"], "Ed25519 验签应通过")
        self.assertTrue(result["ok"])

    def test_verify_tampered_wheel(self):
        """篡改 wheel 后校验必须失败。"""
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake.whl"
            fake.write_bytes(WHEEL.read_bytes() + b"X")
            result = verify_wheel(fake, SIG)
            self.assertFalse(result["sha256_ok"])
            self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
