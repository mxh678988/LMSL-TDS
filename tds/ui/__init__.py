"""tds.ui —— TDS 对话交互测试台（知识分支，不计入可售 IP）。

定位
----
在真实宿主之外提供一块**可直接上手**的对话面板，用于全方位测试 TDS 调度层
所对接的模型能力：文本对话、多模态图片理解、流式输出、时延与吞吐观测。

边界
----
1. 本子包**不是**可售 IP 的一部分：它只做"观测与交互"，不参与调度决策；
2. 零第三方依赖（标准库 http.server + urllib），不调用任何外部程序；
3. 前端为随包分发的单文件 HTML，无 CDN、无构建步骤。

快速使用
--------
    python -m tds.ui.cli serve --open          # 启动面板并打开浏览器
    python -m tds.ui.cli probe                 # 只做后端自检
"""
from .backend import BackendClient, BackendError, ChatRequest, Chunk, ModelInfo
from .server import DEFAULT_BACKEND, UiServer, serve
from .session import Attachment, ChatSession, Message, SessionStore, make_attachment

__all__ = [
    "BackendClient",
    "BackendError",
    "ChatRequest",
    "Chunk",
    "ModelInfo",
    "UiServer",
    "serve",
    "DEFAULT_BACKEND",
    "Attachment",
    "ChatSession",
    "Message",
    "SessionStore",
    "make_attachment",
]
