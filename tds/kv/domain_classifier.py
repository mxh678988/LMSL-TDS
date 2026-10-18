# -*- coding: utf-8 -*-
"""域分类器 v2：动作词+内容词组合检测，避免对话关键词单独出现就误判。

当前问题：仅凭"继续"/"分析"等单个对话词就会把对话 prompt 判入 qita，
因为缺少内容词匹配。v2 通过"动作词与内容词至少一个存在"来提准。

用法
----
    from tds.kv.domain_classifier import DomainClassifier
    classifier = DomainClassifier()
    domain = classifier.classify("帮我总结一下这个报告的内容")  # → duihua
    domain = classifier.classify("生成一张风景图")                   # → tuxiang
"""

import re
from typing import Dict, List, Optional

# ---- 关键词组 ----

DIALOGUE_ACTIONS = {
    "继续", "总结", "翻译", "解释", "分析", "回答", "说明", "讨论",
    "回复", "告诉我", "你怎么看", "请帮我", "帮我", "写个", "写一段",
    "生成", "创作", "设计", "写一个", "怎么", "如何", "为什么",
    "是什么", "哪个", "推荐", "建议", "评价", "对比", "比较",
    "你觉得", "你怎么", "请分析", "请总结", "请解释", "请回答",
    "请说明", "请帮我", "我想知道", "我不确定", "我想", "我想了解",
    "我想问", "我想讨论", "我想分析", "我想总结", "我想翻译",
    "我想评价", "我想对比", "我想比较", "我想推荐", "我想建议",
    "我想设计", "我想创作", "我想写", "我想生成", "我想写个",
    "我想写一段", "我想写一个",
}

DIALOGUE_CONTENTS = {
    "文章", "报告", "总结", "翻译", "解释", "分析", "回答", "说明",
    "讨论", "回复", "告诉我",
    "代码", "代码块", "函数", "类", "脚本", "程序", "def ",
    "class ", "import ", "function", "print(", "return ", "if ",
    "for ", "while ", "try ", "except ", "finally ", "with ",
    "yield ", "async ", "await ", "lambda ", "map(", "filter(",
    "list(", "dict(", "set(", "tuple(", "str(", "int(", "float(",
    "bool(", "len(", "range(", "zip(", "enumerate(", "sorted(",
    "reversed(", "any(", "all(", "open(", "read(", "write(",
    "close(", "os.", "sys.", "json.", "csv.", "re.", "math.",
    "random.", "datetime.", "import", "importlib", "from",
    "module", "package", "pip install", "conda install", "npm install",
    "yarn install", "git ", "commit", "push", "pull", "clone",
    "branch", "merge", "debug", "debugging", "breakpoint", "traceback",
    "exception", "error", "warning", "log", "logging", "test",
    "testing", "unit test", "integration test", "mock", "patch",
    "fixture", "assert", "pytest", "unittest",
}

# 图像内容词：出现即视为图像生成/处理任务，优先级高于对话域，
# 避免"生成一张图/设计海报/写个封面"被动作词误判为对话。
IMAGE_CONTENTS = {
    "图片", "图", "截图", "头像", "封面", "生成图", "设计图",
    "海报", "插画", "插图", "照片", "图像", "画面", "视觉效果",
    "视觉", "颜色", "色彩", "色调", "构图", "透视", "光影", "渲染",
    "风格", "风格化", "艺术", "美术", "绘画", "素描", "油画", "水彩",
    "版画", "手绘", "矢量", "像素", "分辨率", "高清", "4K",
    "图片格式", "图片转换", "图片编辑", "图片裁剪", "图片压缩",
    "图片质量", "图片优化", "图片修复", "图片增强", "图片处理",
    "图片分析", "图片识别", "图片分类",
}

SEARCH_KEYWORDS = {
    "搜索", "查找", "查询", "检索", "查一下", "找一下", "搜索一下",
    "查询一下", "查找一下",
}

TOOL_KEYWORDS = {
    "json", "xml", "yaml", "csv", "xlsx", "pdf", "docx", "pptx",
    "html", "css", "javascript", "typescript", "sql", "api",
    "url", "http", "request", "response", "curl", "wget",
    "pip", "conda", "npm", "yarn", "docker", "docker-compose",
    "git", "gitlab", "github", "gitee", "bitbucket",
}


class DomainClassifier:
    """v2 域分类器：动作词+内容词组合检测。

    匹配规则（优先级从高到低）：
    1. 图像：图片/图/截图/头像/封面/生成图等 → tuxiang（图像内容词优先于对话域，
       防止"生成/设计/写个"等动作词把图像任务误判为对话）
    2. 对话：动作词或内容词命中任意一个 → duihua
    3. 代码：def/class/import/function 等 → daima
    4. 检索：搜索/查找/查询/检索 → jiansuo
    5. 工具：json/xml/yaml 等 → gongju
    6. 其他 → qita
    """

    def __init__(self, custom_keywords: Optional[Dict[str, List[str]]] = None):
        self.custom_keywords = custom_keywords or {}
        self._compiled: Dict[str, List[re.Pattern]] = {}

    def _compile(self):
        self._compiled.clear()
        for domain, keywords in {
            "dialogue": list(self.custom_keywords.get("dialogue", []) + list(DIALOGUE_ACTIONS | DIALOGUE_CONTENTS)),
            "image": list(self.custom_keywords.get("image", []) + list(IMAGE_CONTENTS)),
            "code": list(self.custom_keywords.get("code", []) + list(TOOL_KEYWORDS)),
            "search": list(self.custom_keywords.get("search", []) + list(SEARCH_KEYWORDS)),
            "tool": list(self.custom_keywords.get("tool", []) + list(TOOL_KEYWORDS)),
        }.items():
            patterns = []
            for kw in keywords:
                patterns.append(re.compile(re.escape(kw), re.IGNORECASE | re.UNICODE))
            self._compiled[domain] = patterns

    def classify(self, prompt: str) -> str:
        """域分类，返回 duihua/tuxiang/daima/jiansuo/gongju/qita。"""
        if not self._compiled:
            self._compile()

        # 优先级：image > dialogue > code > search > tool
        # 图像内容词优先：出现图/图片/海报/封面等即判 tuxiang，
        # 防止"生成/设计/写个"等动作词把图像任务误判为对话。
        priority = ["image", "dialogue", "code", "search", "tool"]
        for domain in priority:
            if domain not in self._compiled:
                continue
            for pat in self._compiled[domain]:
                if pat.search(prompt):
                    # 对话域：动作词与内容词至少一个存在即可
                    if domain == "dialogue":
                        # 只要命中任意一个关键词，就算对话
                        return "duihua"
                    # 其他域
                    return self._map_domain(domain)
        return "qita"

    def _map_domain(self, domain: str) -> str:
        mapping = {
            "dialogue": "duihua", "image": "tuxiang", "code": "daima",
            "search": "jiansuo", "tool": "gongju",
        }
        return mapping.get(domain, "qita")

    def reset(self):
        self._compiled.clear()
