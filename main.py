"""
主程序 - 基于RapidOCR的屏幕截图答题辅助工具
改进：完善依赖检查、修复线程安全、更健壮的错误处理
"""

import sys
import os
import re
import json
import time
import logging
import threading
from typing import Optional, Dict

from PIL import Image

# PEP 8: Windows + Python 3.14 环境下，onnxruntime 必须在 PyQt5 之前加载，
# 否则会出现 "DLL load failed while importing onnxruntime_pybind11_state"。
if os.name == "nt":
    try:
        import importlib.util

        _ort_spec = importlib.util.find_spec("onnxruntime")
        if _ort_spec and _ort_spec.submodule_search_locations:
            _ort_capi = os.path.join(_ort_spec.submodule_search_locations[0], "capi")
            if os.path.isdir(_ort_capi):
                _ort_dll_dir = os.add_dll_directory(_ort_capi)  # 保持句柄存活
        import onnxruntime  # noqa: F401
    except Exception:
        pass

# 脚本所在目录，用于定位同目录下的文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGION_JSON = os.path.join(BASE_DIR, "capture_region.json")
REGION_SELECTOR_SCRIPT = os.path.join(BASE_DIR, "region_selector.py")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查所有必要的依赖"""
    missing = []

    # 核心依赖（RapidOCR替代PaddleOCR）
    core_deps = {
        'PyQt5': 'PyQt5',
        'mss': 'mss',
        'PIL': 'Pillow',
        'keyboard': 'keyboard',
        'rapidocr_onnxruntime': 'rapidocr_onnxruntime',
    }

    for module, package in core_deps.items():
        try:
            __import__(module)
            logger.debug(f"✓ {package}")
        except ImportError:
            missing.append(package)
            logger.warning(f"✗ {package} 未安装")

    if missing:
        print("\n" + "=" * 50)
        print("⚠️  缺少必要的依赖包")
        print("=" * 50)
        print("\n请运行以下命令安装：")
        print(f"  pip install {' '.join(missing)}")
        print("\n或一次性安装所有依赖：")
        print("  pip install PyQt5 mss Pillow keyboard rapidocr_onnxruntime")
        print("=" * 50)

        input("\n按回车退出...")
        sys.exit(1)

    logger.info("✓ 所有依赖检查通过")


# 延迟导入（依赖检查后）
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from config import (
    CAPTURE_REGION, HOTKEY_SELECT, HOTKEY_PAUSE,
    HOTKEY_CAPTURE, HOTKEY_EXIT, SCAN_INTERVAL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEEPSEEK_ENABLE_SEARCH,
    USE_DEEPSEEK_VISION, VISION_MODEL, VISION_DETAIL,
    VISION_MAX_TOKENS, VISION_TIMEOUT, VISION_MAX_RETRIES,
    VISION_MAX_IMAGE_SIDE, VISION_FAIL_BACKOFF,
    TEXT_MAX_TOKENS, TEXT_TIMEOUT,
)
from overlay import AnswerOverlay
from ocr_engine import OCREngine
from ai_solver import DeepSeekSolver


class HotkeyHandler(QObject):
    """处理热键回调，通过信号确保 UI 调用线程安全"""
    select_region_signal = pyqtSignal()
    toggle_pause_signal = pyqtSignal()
    capture_signal = pyqtSignal()
    exit_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def setup_hotkeys(self):
        try:
            import keyboard
            keyboard.add_hotkey(HOTKEY_SELECT, lambda: self.select_region_signal.emit())
            # 切换类热键在松开时触发，避免按住不放时被键盘重复事件反复切换
            keyboard.add_hotkey(
                HOTKEY_PAUSE,
                lambda: self.toggle_pause_signal.emit(),
                trigger_on_release=True,
            )
            keyboard.add_hotkey(HOTKEY_CAPTURE, lambda: self.capture_signal.emit())
            keyboard.add_hotkey(HOTKEY_EXIT, lambda: self.exit_signal.emit())
            logger.info(f"热键已注册: {HOTKEY_SELECT}, {HOTKEY_PAUSE}, {HOTKEY_CAPTURE}, {HOTKEY_EXIT}")
        except Exception as e:
            logger.error(f"热键注册失败: {e}")


class CaptureThread(QThread):
    """屏幕捕获线程"""
    result_signal = pyqtSignal(dict)
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.paused = True
        self.single_shot = False
        self.sct = None
        # 用于打断 sleep，让暂停/继续、单次识别立即生效
        self._wake = threading.Event()

        # OCR和AI将在run中初始化
        self.ocr = None
        self.ai = None

        # 加载捕获区域
        self.region = self._load_region()

        # 状态变量
        self.last_text = ""              # OCR raw-text 去重
        self.last_result_key = ""        # 题干文本去重
        self.last_attempt_sig = None     # 帧签名（避免重复视觉调用）
        self.vision_disabled_until = 0.0 # 视觉失败冷却截止时间
        self.use_vision = USE_DEEPSEEK_VISION

        self.api_count = 0
        self.vision_count = 0
        self.fallback_count = 0
        self.start_time = time.time()

    def _load_region(self) -> Dict:
        """加载捕获区域配置"""
        try:
            with open(REGION_JSON, "r", encoding="utf-8") as f:
                region = json.load(f)
                logger.info(f"已加载捕获区域: {region}")
                return region
        except FileNotFoundError:
            logger.info(f"使用默认捕获区域: {CAPTURE_REGION}")
            return CAPTURE_REGION
        except Exception as e:
            logger.error(f"加载区域配置失败: {e}")
            return CAPTURE_REGION

    def run(self):
        """线程主循环"""
        try:
            import mss
            self.sct = mss.MSS()

            # 在运行线程中初始化引擎（避免构造时阻塞信号连接）
            self.ocr = self._init_ocr_engine()
            if self.ocr is None:
                logger.warning("OCR引擎初始化失败，视觉模式仍可用，但回退不可用")

            self.ai = self._init_ai_solver()
            if self.ai is None:
                logger.error("AI解题器初始化失败，线程终止")
                return

            vision_mode = "DeepSeek视觉优先 + RapidOCR回退" if self.use_vision else "RapidOCR + 文本DeepSeek"
            logger.info(f"""
{'='*50}
🚀 DeepSeek自动答题助手已启动
{'='*50}
识别模式: {vision_mode}
视觉模型: {VISION_MODEL}
捕获区域: {self.region}
快捷键:
  • Ctrl+F1 = 重新框选区域
  • Ctrl+F2 = 暂停/继续（启动时默认暂停）
  • Ctrl+F3 = 手动识别一次 ⚡
  • Ctrl+Q  = 退出程序
{'='*50}
""")

            while self.running:
                self._wake.clear()
                if self.single_shot:
                    self.single_shot = False
                    try:
                        self._process_frame(force=True)
                    except Exception as e:
                        logger.error(f"处理帧时出错: {e}")
                elif not self.paused:
                    try:
                        self._process_frame(force=False)
                    except Exception as e:
                        logger.error(f"处理帧时出错: {e}")
                # 可被 pause()/trigger_capture()/stop() 立即唤醒
                self._wake.wait(SCAN_INTERVAL)

        except Exception as e:
            logger.error(f"捕获线程异常: {e}")
        finally:
            if self.sct:
                try:
                    self.sct.close()
                except:
                    pass

    def _init_ocr_engine(self) -> Optional[OCREngine]:
        """初始化OCR引擎（可能耗时，放在run中）"""
        try:
            logger.info("正在初始化OCR引擎...")
            engine = OCREngine(use_gpu=False, lang='ch', show_log=False)
            version_info = engine.get_version_info()
            logger.info(f"OCR引擎版本: {version_info}")
            return engine
        except Exception as e:
            logger.error(f"OCR引擎初始化失败: {e}")
            return None

    def _init_ai_solver(self) -> Optional[DeepSeekSolver]:
        """初始化AI解题器"""
        try:
            logger.info("正在初始化AI解题器...")
            solver = DeepSeekSolver(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                model=DEEPSEEK_MODEL,
                enable_search=DEEPSEEK_ENABLE_SEARCH,
                use_vision=USE_DEEPSEEK_VISION,
                vision_model=VISION_MODEL,
                vision_detail=VISION_DETAIL,
                vision_max_tokens=VISION_MAX_TOKENS,
                vision_timeout=VISION_TIMEOUT,
                vision_max_retries=VISION_MAX_RETRIES,
                text_max_tokens=TEXT_MAX_TOKENS,
                text_timeout=TEXT_TIMEOUT,
                vision_max_image_side=VISION_MAX_IMAGE_SIDE,
            )
            logger.info("AI解题器初始化完成")
            return solver
        except Exception as e:
            logger.error(f"AI解题器初始化失败: {e}")
            return None

    def _process_frame(self, force: bool = False):
        """处理单帧：视觉优先 → 本地OCR → 文本DeepSeek"""
        if not self.ai:
            return

        try:
            t0 = time.time()
            screenshot = self.sct.grab(self.region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            t1 = time.time()

            # 帧签名：画面几乎没变时跳过，避免连续扫描反复调用视觉 API
            signature = self._frame_signature(img)
            if not force and self.last_attempt_sig is not None:
                if self._signature_close(signature, self.last_attempt_sig):
                    logger.debug("画面未变化，跳过本帧")
                    return
            self.last_attempt_sig = signature

            # 1) 视觉优先
            vision_elapsed = 0.0
            if self.use_vision and (force or time.time() >= self.vision_disabled_until):
                t2 = time.time()
                vision_result = self.ai.solve_image(img)
                t3 = time.time()
                vision_elapsed = t3 - t2

                if vision_result.get("ok"):
                    self.vision_count += 1
                    self.api_count += 1
                    self.vision_disabled_until = 0.0
                    self._publish(vision_result, force, elapsed=t3 - t1)
                    return

                error_type = vision_result.get("error_type", "api")
                error = vision_result.get("error", "未知错误")
                if error_type == "not_question" and not force:
                    logger.info("视觉判断为非题目画面，跳过")
                    return

                logger.warning(
                    "视觉失败(%s)，回退本地OCR: %s", error_type, error
                )
                if error_type != "not_question":
                    self.vision_disabled_until = time.time() + VISION_FAIL_BACKOFF

            # 2) 本地 OCR 回退
            if not self.ocr:
                self._emit_error("视觉失败且OCR回退不可用")
                return

            t4 = time.time()
            ocr_result = self.ocr.recognize(img)
            t5 = time.time()
            if not ocr_result or not ocr_result.get("is_valid"):
                logger.info(
                    "⏱ 截图:%.2fs | OCR:%.2fs → 无效帧，跳过",
                    t1 - t0, t5 - t4,
                )
                return

            current_text = ocr_result.get("raw_text", "")[:50]
            if current_text == self.last_text:
                return
            self.last_text = current_text

            logger.info("📝 OCR题目: %s...", ocr_result["question"][:60])

            t6 = time.time()
            answer = self.ai.solve(ocr_result)
            t7 = time.time()

            if answer.get("ok"):
                self.fallback_count += 1
                self.api_count += 1
                answer["source"] = (
                    "本地OCR+DeepSeek" if self.use_vision else "DeepSeek"
                )
            else:
                answer.setdefault("source", "error")

            logger.info(
                "⏱ 截图:%.2fs | 视觉:%.2fs | OCR:%.2fs | 文本API:%.2fs",
                t1 - t0, vision_elapsed, t5 - t4, t7 - t6,
            )
            self._publish(answer, force, elapsed=t7 - t4)

        except Exception as e:
            logger.error(f"处理帧失败: {e}")

    @staticmethod
    def _frame_signature(img: Image.Image, size: int = 8) -> int:
        """8x8 差值哈希（dHash），对计时器等小幅变化不敏感。"""
        small = img.convert("L").resize(
            (size + 1, size), Image.Resampling.LANCZOS
        )
        pixels = small.tobytes()
        bits = 0
        for row in range(size):
            base = row * (size + 1)
            for col in range(size):
                bits = (bits << 1) | (
                    pixels[base + col] > pixels[base + col + 1]
                )
        return bits

    @staticmethod
    def _signature_close(a: int, b: int, tolerance: int = 3) -> bool:
        return (a ^ b).bit_count() <= tolerance

    @staticmethod
    def _make_result_key(result: Dict) -> str:
        question = result.get("question") or ""
        if question:
            return re.sub(r'[\s，。、；：？！,.;:?!]+', '', question)[:120]
        raw = result.get("raw_text") or ""
        return re.sub(r'[\s，。、；：？！,.;:?!]+', '', raw)[:80]

    def _publish(self, result: Dict, force: bool, elapsed: float = 0.0):
        """去重、记录日志并通过信号发送结果。"""
        key = self._make_result_key(result)
        if result.get("ok") and key and not force and key == self.last_result_key:
            logger.info("同题已处理，跳过重复显示")
            return
        if result.get("ok") and key:
            self.last_result_key = key

        answer = result.get("answer", "?")
        source = result.get("source", "unknown")
        logger.info(
            "✅ 答案: 【%s】 | 题型: %s | 来源: %s | 耗时: %.2fs",
            answer, result.get("question_type"), source, elapsed,
        )
        detail = result.get("detail", "")
        if detail:
            logger.debug("   解析: %s", detail[:50])

        self.result_signal.emit(result)
        self.status_signal.emit(self.status_text())

    def _emit_error(self, detail: str):
        logger.error(detail)
        self.result_signal.emit({
            "ok": False,
            "answer": "?",
            "answers": [],
            "detail": detail,
            "source": "error",
            "question": "",
            "options": {},
            "question_type": "unknown",
            "raw_text": "",
        })

    def update_region(self, new_region: Dict):
        """更新捕获区域，并重置去重与视觉冷却状态"""
        self.region = new_region
        self.last_text = ""
        self.last_result_key = ""
        self.last_attempt_sig = None
        self.vision_disabled_until = 0.0
        logger.info(f"捕获区域已更新: {new_region}")

    def set_paused(self, paused: bool):
        """设置暂停状态，并唤醒捕获线程立即响应。"""
        if self.paused != paused:
            self.paused = paused
            self._wake.set()

    def pause(self) -> bool:
        """切换暂停状态，返回切换后的状态。"""
        self.set_paused(not self.paused)
        return self.paused

    def stop(self):
        """停止线程"""
        self.running = False
        self._wake.set()
        logger.info("正在停止捕获线程...")

    def trigger_capture(self):
        """触发一次识别"""
        self.single_shot = True
        self._wake.set()

    def status_text(self) -> str:
        """状态栏文本：始终包含当前扫描状态。"""
        runtime = time.time() - self.start_time
        state = "⏸ 已暂停" if self.paused else "▶ 连续扫描"
        return (
            f"{state} | 运行: {int(runtime)}s | API: {self.api_count}次 | "
            f"视觉: {self.vision_count}次 | 回退: {self.fallback_count}次"
        )


class AutoAnswerApp:
    """主应用类"""
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("DeepSeek答题助手")

        # 创建悬浮窗
        self.overlay = AnswerOverlay()

        # 创建捕获线程
        self.thread = CaptureThread()
        self.thread.result_signal.connect(self.overlay.show_answer)
        self.thread.status_signal.connect(self.overlay.update_status)

        # 热键状态：防止重复触发 / 框选期间误切换
        self._last_pause_toggle = 0.0
        self._selecting_region = False

        # 初始化热键处理器（线程安全）
        self.hotkey_handler = HotkeyHandler()
        self.hotkey_handler.select_region_signal.connect(self._on_select_region)
        self.hotkey_handler.toggle_pause_signal.connect(self._on_toggle_pause)
        self.hotkey_handler.capture_signal.connect(self._on_capture)
        self.hotkey_handler.exit_signal.connect(self._on_exit)
        self.hotkey_handler.setup_hotkeys()

        # 启动线程
        self.thread.start()
        self.overlay.update_status("⏸ 已暂停 | Ctrl+F2 开始连续扫描 | Ctrl+F3 单次识别")
        logger.info("应用启动完成")

    def _on_capture(self):
        """手动触发一次识别（主线程）"""
        if self._selecting_region:
            self.overlay.show_message("正在框选区域，请稍候")
            return
        logger.info("⚡ 手动触发识别...")
        self.overlay.show_message("正在识别...")
        self.thread.trigger_capture()

    def _on_select_region(self):
        """重新框选区域（在主线程执行）"""
        logger.info("触发区域选择...")
        self._selecting_region = True
        was_paused = self.thread.paused
        self.thread.set_paused(True)
        self.overlay.update_status("⏸ 正在框选区域...")

        print("\n" + "=" * 40)
        print("请框选要捕获的屏幕区域...")
        print("=" * 40)

        try:
            import subprocess
            subprocess.run(
                [sys.executable, REGION_SELECTOR_SCRIPT],
                timeout=60
            )
            try:
                with open(REGION_JSON, "r", encoding="utf-8") as f:
                    new_region = json.load(f)
                    self.thread.update_region(new_region)
                    self.overlay.show_message("区域已更新 ✓")
            except Exception as e:
                logger.error(f"加载新区域失败: {e}")
                self.overlay.show_message("区域加载失败 ✗")
        except subprocess.TimeoutExpired:
            logger.warning("区域选择超时")
            self.overlay.show_message("选择超时 ✗")
        except Exception as e:
            logger.error(f"区域选择失败: {e}")
            self.overlay.show_message("选择失败 ✗")
        finally:
            self._selecting_region = False
            # 恢复框选前的扫描状态（框选期间 Ctrl+F2 不生效）
            self.thread.set_paused(was_paused)
            self.overlay.update_status(self.thread.status_text())

    def _on_toggle_pause(self):
        """切换暂停状态（主线程）"""
        if self._selecting_region:
            self.overlay.show_message("正在框选区域，暂不能切换扫描状态")
            return

        # 防抖：避免松开热键时产生重复信号
        now = time.time()
        if now - self._last_pause_toggle < 0.25:
            return
        self._last_pause_toggle = now

        is_paused = self.thread.pause()
        if is_paused:
            status = "已暂停 ⏸ | Ctrl+F2 继续 | Ctrl+F3 单次识别"
        else:
            status = "连续扫描中 ▶ | Ctrl+F2 暂停"
        self.overlay.update_status(self.thread.status_text())
        self.overlay.show_message(status)
        logger.info(status)

    def _on_exit(self):
        """退出程序（主线程）"""
        print("\n" + "=" * 40)
        print("正在退出...")
        print("=" * 40)

        runtime = time.time() - self.thread.start_time
        logger.info(f"运行时长: {int(runtime)}秒")
        logger.info(f"API调用次数: {self.thread.api_count}")
        logger.info(f"视觉识别次数: {self.thread.vision_count}")
        logger.info(f"OCR回退次数: {self.thread.fallback_count}")

        self.thread.stop()
        self.thread.wait(3000)
        self.app.quit()

    def run(self):
        """运行应用"""
        logger.info("进入主事件循环...")
        sys.exit(self.app.exec_())


def main():
    print("\n" + "=" * 50)
    print("  DeepSeek 自动答题助手 v2.1")
    print("  DeepSeek 视觉优先 + RapidOCR 回退")
    print("=" * 50 + "\n")

    # 检查依赖
    check_dependencies()

    # 检查API密钥
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-key-here":
        print("\n" + "=" * 50)
        print("❌ 错误：未配置DeepSeek API密钥")
        print("=" * 50)
        print("\n请在环境变量中设置 DEEPSEEK_API_KEY")
        print("Windows: set DEEPSEEK_API_KEY=你的密钥")
        print("Linux/Mac: export DEEPSEEK_API_KEY=你的密钥")
        print("获取地址：https://platform.deepseek.com/")
        print("=" * 50)
        input("\n按回车退出...")
        sys.exit(1)

    try:
        app = AutoAnswerApp()
        app.run()
    except KeyboardInterrupt:
        logger.info("收到键盘中断，正在退出...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"应用异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
