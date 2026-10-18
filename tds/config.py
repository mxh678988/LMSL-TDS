"""全局配置与八卦八域定义。"""
from dataclasses import dataclass

# 八卦八域：语义分类索引的八个"域"
BAGUA_DOMAINS = {
    "qian":    "乾坤·系统指令与角色规则（永久驻留）",
    "duihua":  "对话·多轮上下文",
    "jiansuo": "检索·RAG 文档片段",
    "daima":   "代码·代码与结构化任务",
    "tuxiang": "图像·图像生成与编辑上下文",
    "shipin":  "视频·视频生成上下文",
    "gongju":  "工具·工具描述与调用协议",
    "qita":    "其他·冷数据",
}

YANG = "yang"  # 阳：热（近期将使用，倾向驻留最快层）
YIN = "yin"    # 阴：冷（低频/历史，倾向下沉）

# 八卦加权动态阈值：各域 keep_steps 基准值
# 对话上下文长、代码逻辑连续 → 大 keep_steps；检索短、工具短暂 → 小 keep_steps
BAGUA_KEEP_STEPS = {
    "qian":    1,    # 乾坤：系统指令，永久保护（通过 pinned 而非 keep_steps）
    "duihua":  3,    # 对话：上下文连贯，多轮复用
    "jiansuo": 2,    # 检索：片段复用率中等
    "daima":   2,    # 代码：逻辑连续，块间引用
    "tuxiang": 2,    # 图像：单次为主，偶有迭代
    "shipin":  1,    # 视频：单次生成为主
    "gongju":  1,    # 工具：短暂调用
    "qita":    1,    # 其他：默认保守
}

# 八卦域优先级（分类分数并列时的 tie-break 顺序，前 = 优先）
# 重资源域（视频/图像）优先保护，其次代码/对话；乾坤最高
BAGUA_PRIORITY = ["qian", "shipin", "tuxiang", "daima", "duihua", "jiansuo", "gongju", "qita"]

# KV Cache 分页大小配置
TDS_PAGE_SIZE_TOKENS = 128  # 每页 token 数量
TDS_PAGE_SIZE_BYTES = TDS_PAGE_SIZE_TOKENS * 2 * 64  # 每页字节数（128 tokens * 2 dims * 64 bytes per element）


@dataclass
class TDSConfig:
    """调度层配置。默认保守，避免 OOM。"""
    vram_budget_mb: int = 10788      # 显存预算上限（MB）· Phase0 实测校准 = 12288 - 1500
    reserve_mb: int = 1500           # 预留显存（防 OOM）· 覆盖桌面/驱动常驻 ~860MB + 抖动垫
    chunk_tokens: int = 256          # 分层块粒度（行业默认，兼顾 I/O 与复用）
    ssd_tier_root: str = "D:\\"      # 分层 SSD 落盘区（C 盘剩余紧张，禁用）
    keep_steps: int = 1              # 默认"下次使用距离"阈值（八卦加权后将覆盖）
    prefetch_enabled: bool = False   # 预取开关（默认关闭，保守起步）
    pressure_guard: bool = True      # 显存压力保护
    pressure_ratio: float = 0.92     # 压力触发阈值（>= 即进入四象·高压力）
    frozen_ratio: float = 0.98       # 冻结阈值（>= 即进入四象·冻结）
    trace_maxlen: int = 200          # 决策轨迹环形缓冲容量（条），长跑不增长
    log_level: str = "INFO"
