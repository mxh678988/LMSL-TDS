# -*- coding: utf-8 -*-
"""紫梅太极八卦调度规则 —— 中华 KV 分页调度的"命理·卦象"层（TBP 二期 IP）。

在太极（两仪×四象×八卦）基础上，融入：
    紫微斗数：命宫优先 —— 当前活跃域（命宫）永不整层卸载；
              主星保活   —— 域内高频页（主星）驻留、低频页（辅星）可杀。
    梅花易数：八卦生克 —— 跨域淘汰序不再静态预设，而是按"命宫五行 与
              候选域五行"的生克关系动态生成卦象淘汰序（动态卦象）。

五行生克（梅花易数·后天八卦配五行）：
    乾 daima  金 | 坤 jiansuo 土 | 震 tuxiang 木 | 巽 gongju 木
    坎 shipin 水 | 离 duihua  火 | 艮 qian    土 | 兑 qita   金

命宫五行对候选域五行的五种关系（决定淘汰序，值越大越优先淘汰）：
    我克者（财）→ 可驾驭，最先舍        ：优先=4
    我生者（食伤）→ 泄我之气，次之      ：优先=3
    同我者（比劫）→ 平辈同类，可舍      ：优先=2
    生我者（印）→ 养我之源，保留        ：优先=1
    克我者（官）→ 制我之神，最后才动    ：优先=0
    命宫（本宫）→ 永不淘汰              ：绝对保护

本模块纯函数/数据驱动，不 import torch；决策与搬运解耦，由
page_manager 在运行时消费本模块产出的"紫梅淘汰序"与"命宫保护位"。
"""
from __future__ import annotations

from typing import Dict, List, Optional

# ---- 梅花易数：八卦五行（域 → 五行） ----
BAGUA_WUXING: Dict[str, str] = {
    "daima":   "金",   # 乾卦
    "jiansuo": "土",   # 坤卦
    "tuxiang": "木",   # 震卦
    "gongju":  "木",   # 巽卦
    "shipin":  "水",   # 坎卦
    "duihua":  "火",   # 离卦
    "qian":    "土",   # 艮卦
    "qita":    "金",   # 兑卦
}

# 五行相生（母生子）：key=母行 -> value=子行
WUXING_SHENG: Dict[str, str] = {
    "土": "金",   # 土生金
    "金": "水",   # 金生水
    "水": "木",   # 水生木
    "木": "火",   # 木生火
    "火": "土",   # 火生土
}

# 五行相克（克者克被克者）：key=克者 -> value=被克者
WUXING_KE: Dict[str, str] = {
    "金": "木",   # 金克木
    "木": "土",   # 木克土
    "土": "水",   # 土克水
    "水": "火",   # 水克火
    "火": "金",   # 火克金
}


def wuxing_of(domain: str) -> Optional[str]:
    """返回域对应的五行属性；未知域返回 None。"""
    return BAGUA_WUXING.get(domain)


def relation_to_ming(domain: str, ming_domain: str) -> str:
    """候选域相对命宫的五行关系（财/食伤/比劫/印/官）。

    - domain == ming_domain -> "ming"（命宫本宫）
    - 命宫五行克候选域 -> "cai"（我克者，财）
    - 命宫五行生候选域 -> "shishang"（我生者，食伤）
    - 候选域五行生命宫 -> "yin"（生我者，印）
    - 候选域五行克命宫 -> "guan"（克我者，官）
    - 同五行           -> "bijie"（同我者，比劫）
    """
    if domain == ming_domain:
        return "ming"
    m = wuxing_of(ming_domain)
    d = wuxing_of(domain)
    if m is None or d is None:
        return "bijie"  # 未知域：按比劫处理，淘汰序居中
    if WUXING_KE.get(m) == d:
        return "cai"      # 我克者
    if WUXING_SHENG.get(m) == d:
        return "shishang" # 我生者
    if WUXING_SHENG.get(d) == m:
        return "yin"      # 生我者
    if WUXING_KE.get(d) == m:
        return "guan"     # 克我者
    return "bijie"


# 关系 -> 动态淘汰优先级（越大越先淘汰；命宫不参与排序，绝对保护）
RELATION_PRIORITY: Dict[str, int] = {
    "cai":      4,   # 我克者（财）：最先舍
    "shishang": 3,   # 我生者（食伤）：泄气次之
    "bijie":    2,   # 同我者（比劫）：平辈可舍
    "yin":      1,   # 生我者（印）：养我之源，保留
    "guan":     0,   # 克我者（官）：制我之神，最后才动
    "ming":     99,  # 命宫：绝对保护（永不淘汰）
}


def zimei_evict_priority(domain: str, ming_domain: str) -> int:
    """候选域的紫梅动态淘汰优先级（越大越先淘汰）。

    与静态 BAGUA_EVICT_ORDER 不同：此序随命宫（当前活跃域）变化而
    动态变化，即梅花易数"动态卦象"——每个活跃域都是一卦，生克随之。
    """
    rel = relation_to_ming(domain, ming_domain)
    return RELATION_PRIORITY.get(rel, 2)


def zimei_evict_order(ming_domain: str, all_domains: Optional[List[str]] = None) -> List[str]:
    """生成紫梅动态域淘汰序（按优先级降序，命宫本身不进入淘汰序）。

    例：命宫=duihua(火) 时：克火者=水(shipin) 最后动，火生土(jiansuo,
    qian) 先舍 → 序为 [daima, qita, jiansuo, qian, ...]（无命宫）。
    与静态八卦序完全不同；命宫由调用方（page_manager 命宫保护位）单独保护。
    """
    domains = all_domains or list(BAGUA_WUXING.keys())
    return sorted(
        [d for d in domains if d != ming_domain],
        key=lambda d: zimei_evict_priority(d, ming_domain),
        reverse=True,
    )


def is_ming_protected(domain: str, ming_domain: str) -> bool:
    """命宫保护：当前活跃域永不被整层卸载 / 永不淘汰。"""
    return domain == ming_domain


def classify_star(access_count: int, domain_median: Optional[float]) -> str:
    """主星/辅星分类：访问次数 >= 域中位数 → 主星（驻留），否则辅星（可杀）。

    紫微斗数主星掌吉凶、辅星佐助力：主星页高频使用必须驻留，
    辅星页低频可淘汰。中位数为 None（域内暂无统计）时保守视为主星。
    """
    if domain_median is None or access_count >= domain_median:
        return "zhu"   # 主星
    return "fu"        # 辅星


if __name__ == "__main__":
    # 快速自检：命宫=duihua(火) 的动态卦象淘汰序
    print("紫梅规则自检（命宫=duihua 火）")
    for d in ["duihua", "daima", "jiansuo", "tuxiang", "gongju", "shipin", "qian", "qita"]:
        rel = relation_to_ming(d, "duihua")
        print(f"  {d:8s} 五行={wuxing_of(d):2s} 关系={rel:8s} 淘汰优先={zimei_evict_priority(d, 'duihua')}")
    print("动态淘汰序:", zimei_evict_order("duihua"))
