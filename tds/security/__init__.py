"""TDS 安全与授权模块：发布包自有数字签名校验。

授权形态：随包分发公钥 + 离线校验。Ed25519 签名通过即表示发布包确由
BY LAOMENG 网络工作室签名发布且未被篡改（签名即授权）。

典型用法：
    from tds.security import verify_wheel
    result = verify_wheel("tds_core-0.3.0-py3-none-any.whl", "....whl.sig")
    assert result["ok"]

CLI：
    python -m tds.security.verify <wheel> <sig> [--pubkey <pem>]
"""
from .verify import (
    DEFAULT_PUBLIC_KEY_PEM,
    PUBLIC_KEY_FINGERPRINT,
    load_public_key,
    parse_sig_file,
    sha256_bytes,
    sha256_file,
    verify_signature,
    verify_wheel,
)

__all__ = [
    "DEFAULT_PUBLIC_KEY_PEM",
    "PUBLIC_KEY_FINGERPRINT",
    "load_public_key",
    "parse_sig_file",
    "sha256_bytes",
    "sha256_file",
    "verify_signature",
    "verify_wheel",
]
