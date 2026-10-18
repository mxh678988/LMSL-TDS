"""八卦语义分类器。

将输入任务映射到八域（BAGUA_DOMAINS），产出 DomainProfile，
作为"预判预取"的输入信号。

设计原则：
- 首期：关键词/规则启发式，零依赖、低开销、可解释。
- 预留：embed_fn 接口，后续可接入轻量嵌入模型做语义相似度分类。
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import BAGUA_DOMAINS, BAGUA_PRIORITY

# 生成类动作词：写作/生成文本类意图。仅当无图像/视频/代码内容信号时才归对话域；
# 与图像内容词（图/图片/海报/封面等）同时出现时，由图像域接管（生成图片=图像任务）。
GENERATIVE_ACTIONS = {
    "生成", "创作", "设计", "写个", "写一段", "写一个",
    "我想设计", "我想创作", "我想写", "我想生成",
    "我想写个", "我想写一段", "我想写一个",
}

# 各域关键词（首期启发式；后续由嵌入模型替代/增强）
_KEYWORDS = {
    "qian":    ["system", "你是", "角色", "规则", "你必须"],
    "duihua":  ["你好", "继续", "刚才", "上面", "接着",
                "总结", "翻译", "解释", "分析", "回答", "说明", "讨论",
                "回复", "告诉我", "你怎么看", "请帮我", "帮我",
                "怎么", "如何", "为什么", "是什么", "哪个", "推荐", "建议",
                "评价", "对比", "比较", "你觉得", "你怎么", "请分析",
                "请总结", "请解释", "请回答", "请说明", "请帮我",
                "我想知道", "我不确定", "我想", "我想了解", "我想问",
                "我想讨论", "我想分析", "我想总结", "我想翻译",
                "我想评价", "我想对比", "我想比较", "我想推荐",
                "我想建议",
                "文章", "报告", "总结", "翻译", "解释", "分析",
                "回答", "说明", "讨论", "回复", "告诉我"],
    "jiansuo": ["检索", "查一下", "文档", "知识库", "rag", "资料",
                "搜索", "查找", "查询", "检索", "查一下", "找一下"],
    "daima":   ["def ", "class ", "import ", "function", "print(",
                "return ", "if ", "for ", "while ", "try ", "except",
                "finally ", "with ", "yield ", "async ", "await",
                "lambda ", "map(", "filter(", "list(", "dict(", "set(",
                "tuple(", "str(", "int(", "float(", "bool(", "len(",
                "range(", "zip(", "enumerate(", "sorted(", "reversed(",
                "any(", "all(", "open(", "read(", "write(", "close(",
                "os.", "sys.", "json.", "csv.", "re.", "math.",
                "random.", "datetime.", "git ", "commit", "push", "pull",
                "clone", "branch", "merge", "debug", "breakpoint",
                "traceback", "exception", "error", "warning", "log",
                "logging", "test", "testing", "unit test", "mock",
                "patch", "fixture", "assert", "pytest", "unittest",
                "代码", "脚本", "程序", "函数", "调试", "报错",
                "python", "bug", "修复"],
    "tuxiang": ["图", "图片", "图像", "生图", "绘画", "comfyui", "diffusion",
                "sd", "设计图", "海报", "插画", "插图", "头像", "封面",
                "截图", "照片", "画面", "视觉效果", "视觉", "颜色",
                "色彩", "色调", "构图", "透视", "光影", "渲染",
                "风格", "风格化", "艺术", "美术", "绘画", "素描",
                "油画", "水彩", "版画", "手绘", "矢量", "像素",
                "分辨率", "高清", "4K", "图片格式", "图片转换",
                "图片编辑", "图片裁剪", "图片压缩", "图片质量",
                "图片优化", "图片修复", "图片增强", "图片处理",
                "图片分析", "图片识别", "图片分类"],
    "shipin":  ["视频", "动画", "帧", "wan", "ltx", "运镜"],
    "gongju":  ["调用", "工具", "api", "参数", "接口", "curl",
                "wget", "pip", "conda", "npm", "yarn", "docker",
                "gitlab", "github", "gitee", "bitbucket"],
    "qita":    [],
}


@dataclass
class DomainProfile:
    """一次任务的域画像。"""
    scores: dict = field(default_factory=dict)
    primary: str = "qita"

    def top(self, k: int = 3):
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def to_dict(self) -> dict:
        return {"primary": self.primary, "scores": dict(self.scores)}


class BaguaClassifier:
    """八域语义分类器。"""

    def __init__(self, embed_fn: Optional[Callable[[str], list]] = None):
        # embed_fn 预留：输入文本 -> 向量；后续可做语义相似度分类
        self.embed_fn = embed_fn

    def classify(self, text: str, hints: Optional[dict] = None) -> DomainProfile:
        text_l = (text or "").lower()
        scores = {}
        for dom, kws in _KEYWORDS.items():
            if not kws:
                continue
            s = sum(1 for kw in kws if kw.lower() in text_l)
            if s:
                scores[dom] = float(s)
        # 生成类动作词：无图像/视频/代码内容信号时归对话域（写作/生成文本类），
        # 有图像等信号时不增强对话域，避免"生成图片/写个封面"被误判为对话。
        gen_hits = sum(1 for kw in GENERATIVE_ACTIONS if kw.lower() in text_l)
        if gen_hits and not (scores.get("tuxiang") or scores.get("shipin") or scores.get("daima")):
            scores["duihua"] = scores.get("duihua", 0.0) + float(gen_hits)
        if hints:
            for dom, w in hints.items():
                scores[dom] = scores.get(dom, 0.0) + float(w)
        if not scores:
            scores = {"qita": 1.0}
        primary = self._pick_primary(scores)
        return DomainProfile(scores=scores, primary=primary)

    @staticmethod
    def _pick_primary(scores: dict) -> str:
        """分数优先；同分时按 BAGUA_PRIORITY 打破平局（重资源域优先保护）。"""
        def rank(dom: str) -> int:
            try:
                return -BAGUA_PRIORITY.index(dom)   # 越靠前，rank 越大
            except ValueError:
                return -len(BAGUA_PRIORITY)
        return max(scores.items(), key=lambda kv: (kv[1], rank(kv[0])))[0]
