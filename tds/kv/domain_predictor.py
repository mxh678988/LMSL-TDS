# -*- coding: utf-8 -*-
"""访问序列学习预判模块（Phase 3）。

基于滑动窗口 N-gram 马尔可夫链，记录各域历史访问序列，
预测下一目标域，置信度 ≥60% 时触发预取。

设计约束：纯函数，不 import torch，与 PageTable/PageCache 解耦。
"""
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


class DomainPredictor:
    """访问序列预判器：基于滑动窗口 N-gram 马尔可夫链。

    使用 2-gram + 3-gram 混合预测：
      - 2-gram 置信度 ≥ 60% 且与 3-gram 一致 → 确认预判
      - 仅 2-gram ≥60% 但 3-gram 不一致 → 降级为 2-gram 预判
      - 均 <60% → 返回 None，不做预判

    窗口：最近 N 步访问序列（默认 N=20）
    """

    def __init__(self, window: int = 20, confidence_threshold: float = 0.60):
        self.window = window
        self.confidence_threshold = confidence_threshold
        self._history: List[str] = []
        self._ngram_counts: Dict[str, Dict[str, int]] = defaultdict(Counter)  # prev_domain -> {next_domain: count}
        self._trigram_counts: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(Counter)

    def observe(self, domain: str) -> None:
        """记录一次域访问。更新 2-gram/3-gram 转移计数。

        同域连续访问合并为一次（每轮生成内多次 prepare 不会污染序列）。
        """
        if self._history and domain == self._history[-1]:
            return
        if self._history:
            prev = self._history[-1]
            self._ngram_counts[prev][domain] += 1
        if len(self._history) >= 2:
            prev2 = self._history[-2]
            self._trigram_counts[(prev2, self._history[-1])][domain] += 1
        self._history.append(domain)
        if len(self._history) > self.window:
            self._history.pop(0)

    def predict(self, last_domain: str) -> Tuple[Optional[str], float]:
        """预测下一个域。返回 (predicted_domain, confidence)。

        混合策略：2-gram + 3-gram 置信度交叉验证，≥60% 才确认预判。
        last_domain: 最后一次已观测到的域
        """
        # 2-gram 预测
        if last_domain not in self._ngram_counts:
            return None, 0.0
        counts2 = self._ngram_counts[last_domain]
        total2 = sum(counts2.values())
        if total2 == 0:
            return None, 0.0
        best2, conf2_raw = counts2.most_common(1)[0]
        conf2 = conf2_raw / total2

        # 3-gram 预测
        best3, conf3 = None, 0.0
        if len(self._history) >= 3:
            last_two = (self._history[-2], self._history[-1])
            if last_two in self._trigram_counts:
                counts3 = self._trigram_counts[last_two]
                total3 = sum(counts3.values())
                if total3 > 0:
                    best3_raw, conf3_raw = counts3.most_common(1)[0]
                    best3 = best3_raw
                    conf3 = conf3_raw / total3

        # 决策逻辑
        if conf2 >= self.confidence_threshold:
            if best3 is not None and conf3 >= self.confidence_threshold and best2 == best3:
                return best2, max(conf2, conf3)
            if best3 is None:
                return best2, conf2
            if conf3 >= self.confidence_threshold:
                return best3, conf3
        return None, 0.0

    def reset(self) -> None:
        """清空预判器。"""
        self._history.clear()
        self._ngram_counts.clear()
        self._trigram_counts.clear()


# 全局单例
_predictor = DomainPredictor()


def get_predictor() -> DomainPredictor:
    return _predictor
