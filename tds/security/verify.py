"""TDS 发布包自有签名校验模块（Ed25519 + SHA256）。

授权形态：随包分发公钥、离线校验。签名即授权——校验通过表示发布包确由
BY LAOMENG 网络工作室签名发布且未被篡改，可用于可售 SDK 与启动器更新包
的完整性验真。

防替换设计：模块内置默认公钥，并绑定公钥指纹（SHA256 of PEM）。使用内置
公钥时自动校验指纹，公钥被整体替换即拒绝；也可通过 --pubkey 显式传入
外部公钥（建议同时核验指纹）。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import re
import sys
from pathlib import Path

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    _CRYPTO_OK = True
except ImportError:  # pragma: no cover - 依赖缺失时给出明确错误
    _CRYPTO_OK = False

# 内置默认公钥（随 SDK 分发，来源 release/tds_core_public.pem）
DEFAULT_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAZWRrnnwkaupDIh7T7nswsaouP2+q7mWLgBKJEGAwL+Q=
-----END PUBLIC KEY-----
"""
# 公钥指纹（PEM 的 SHA256），防公钥整体替换
PUBLIC_KEY_FINGERPRINT = "C33CA76AEBA8B861C9CBB419472C213D504E2DF92B761824908AB34C56FFAC0A"


def sha256_bytes(data: bytes) -> str:
    """返回 bytes 的 SHA256（大写十六进制）。"""
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    """返回文件 SHA256（大写十六进制），流式读取适合大文件。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_public_key(pem: bytes | None = None) -> Ed25519PublicKey:
    """加载 Ed25519 公钥并强制校验指纹。指纹不符直接拒绝。"""
    if not _CRYPTO_OK:
        raise RuntimeError(
            "缺少 cryptography 依赖，无法验签。请安装: pip install cryptography"
        )
    pem = pem or DEFAULT_PUBLIC_KEY_PEM
    fp = sha256_bytes(pem)
    if fp != PUBLIC_KEY_FINGERPRINT:
        raise ValueError(
            f"公钥指纹校验失败：期望 {PUBLIC_KEY_FINGERPRINT}，实际 {fp}。"
            "拒绝使用未知公钥（可能被替换或非官方公钥）。"
        )
    return load_pem_public_key(pem)


def verify_signature(data: bytes, signature: bytes, pubkey_pem: bytes | None = None) -> bool:
    """Ed25519 验签：data 的签名是否由官方私钥生成。"""
    key = load_public_key(pubkey_pem)
    try:
        key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def parse_sig_file(sig_text: str) -> dict:
    """解析现有 .sig 文本格式，返回 {sha256, signature_b64}。"""
    m_sha = re.search(r"SHA256:\s*([0-9A-Fa-f]{64})", sig_text)
    m_sig = re.search(r"SIGBASE64:\s*([A-Za-z0-9+/=]+)", sig_text)
    if not m_sha or not m_sig:
        raise ValueError("sig 文件格式不正确：缺少 SHA256 或 SIGBASE64 字段")
    return {"sha256": m_sha.group(1).upper(), "signature_b64": m_sig.group(1)}


def verify_wheel(
    wheel_path: str | Path,
    sig_path: str | Path,
    pubkey_pem: bytes | None = None,
) -> dict:
    """对 wheel 执行三重校验：SHA256 完整性 + Ed25519 验签 + 公钥指纹。

    返回 dict：ok = sha256_ok and signature_ok。
    """
    wheel = Path(wheel_path)
    sig = Path(sig_path)
    if not wheel.exists():
        raise FileNotFoundError(f"wheel 不存在: {wheel}")
    if not sig.exists():
        raise FileNotFoundError(f"签名文件不存在: {sig}")

    result = {
        "wheel": str(wheel),
        "sig": str(sig),
        "ok": False,
        "sha256_ok": False,
        "signature_ok": False,
    }
    # 1) SHA256 完整性
    result["sha256"] = sha256_file(wheel)
    sig_info = parse_sig_file(sig.read_text(encoding="utf-8", errors="replace"))
    result["expected_sha256"] = sig_info["sha256"]
    result["sha256_ok"] = result["sha256"] == sig_info["sha256"]

    # 2) Ed25519 验签（签名对象 = wheel 的 SHA256 摘要字节，与签名流程一致）
    signature = base64.b64decode(sig_info["signature_b64"])
    digest = hashlib.sha256(wheel.read_bytes()).digest()
    result["signature_ok"] = verify_signature(digest, signature, pubkey_pem)

    result["ok"] = result["sha256_ok"] and result["signature_ok"]
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tds.security.verify",
        description="TDS 发布包自有签名校验（Ed25519 + SHA256，签名即授权）",
    )
    ap.add_argument("wheel", help="待校验的 wheel 文件路径")
    ap.add_argument("sig", help="签名文件路径（.sig 文本格式）")
    ap.add_argument(
        "--pubkey",
        help="外部公钥 PEM 路径（默认使用内置公钥并校验指纹）",
    )
    args = ap.parse_args(argv)

    pubkey_pem = None
    if args.pubkey:
        pubkey_pem = Path(args.pubkey).read_bytes()

    try:
        r = verify_wheel(args.wheel, args.sig, pubkey_pem)
    except Exception as e:  # noqa: BLE001 - CLI 层统一兜底
        print(f"[FAIL] {e}")
        return 1

    print(f"wheel         : {r['wheel']}")
    print(f"SHA256        : {r['sha256']}")
    print(f"期望 SHA256   : {r['expected_sha256']}")
    print(f"SHA256 一致   : {'OK' if r['sha256_ok'] else 'FAIL'}")
    print(f"Ed25519 验签  : {'OK' if r['signature_ok'] else 'FAIL'}")
    print(f"结果          : {'PASS 签名即授权' if r['ok'] else 'FAIL 拒绝使用'}")
    return 0 if r["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
