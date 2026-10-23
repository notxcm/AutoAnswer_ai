# ================== DeepSeek API配置 ==================

import os

# 从 https://platform.deepseek.com/ 获取API密钥
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY", "sk-your-api-key-here"
)

# DeepSeek API地址
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 模型选择：deepseek-flash 原生支持图片输入（视觉 + 文本）
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

# 是否启用联网搜索（仅用于文本回退；视觉请求不发送该参数）
DEEPSEEK_ENABLE_SEARCH = True

# ================== DeepSeek 视觉识别配置 ==================

# 优先将截图直接交给视觉模型读题；失败后回退 RapidOCR + 文本模型
USE_DEEPSEEK_VISION = os.environ.get(
    "USE_DEEPSEEK_VISION", "1"
).lower() not in ("0", "false", "no")

VISION_MODEL = os.environ.get("VISION_MODEL", "deepseek-flash")
VISION_DETAIL = os.environ.get("VISION_DETAIL", "high")  # low/high/original/auto
VISION_MAX_TOKENS = 4096        # 需覆盖模型推理 tokens，否则正文可能被截断
VISION_TIMEOUT = 40
VISION_MAX_RETRIES = 2
VISION_MAX_IMAGE_SIDE = 1600   # 长边超过则等比缩放，控制请求体积
VISION_FAIL_BACKOFF = 5.0      # 视觉连续失败后的冷却秒数

# ================== 文本回退配置 ==================

TEXT_MAX_TOKENS = 4096         # 原 120 可能不足以输出完整答案；推理 tokens 也计入该预算
TEXT_TIMEOUT = 40

# ================== 屏幕捕获配置 ==================

# 默认捕获区域（后面用工具重新框选）
CAPTURE_REGION = {
    "left": 100,
    "top": 100,
    "width": 800,
    "height": 600
}

# 快捷键
HOTKEY_SELECT = "ctrl+f1"    # 框选区域
HOTKEY_PAUSE = "ctrl+f2"     # 暂停/继续
HOTKEY_CAPTURE = "ctrl+f3"   # 手动触发一次识别
HOTKEY_EXIT = "ctrl+q"       # 退出

# 扫描间隔（秒）
SCAN_INTERVAL = 1.0
