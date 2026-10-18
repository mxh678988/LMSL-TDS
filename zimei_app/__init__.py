# -*- coding: utf-8 -*-
"""太易 · 紫梅调度控制台（TAIYI ZiMei Console）—— 独立软件包。

太极紫梅系列从调度组件升级为独立软件：金黑赛博玄学 GUI +
零依赖后端（http.server），内核真实接入：
    tds.kv.taiji_zimei   紫梅八卦调度规则（命宫/五行/生克/动态淘汰序）
    tds.evolution        总体进化 + IP 进化双线引擎（档案只读展示）
    tds.testing          测试技术接口分支（压力注入/A-B/回归/探针）

启动方式：
    python -m zimei_app.server --open
"""
__version__ = "1.0.0"
__app_name__ = "太易 · 紫梅调度控制台"
__app_name_en__ = "TAIYI ZiMei Console"
__slogan__ = "命宫为尊 · 卦象定序 · 生克制衡 · 自进化不辍"

__all__ = ["__version__", "__app_name__", "__app_name_en__", "__slogan__"]
