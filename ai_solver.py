"""
AI解题模块 - DeepSeek API（原生视觉优先 + 文本兜底）。

视觉路径一次请求完成“读题 + 判题型 + 作答”；
文本路径处理本地 OCR 的兜底结果。
"""

import base64
import io
import json
import re
import time
import logging
from typing import Any, Dict, List, Optional

import openai

from answer_utils import (
    normalize_answer_labels,
    normalize_option_label,
    normalize_true_false,
    split_answer_tokens,
)

logger = logging.getLogger(__name__)

CHOICE_TYPES = ("choice", "single_choice", "multiple_choice")
KNOWN_TYPES = ("single_choice", "multiple_choice", "true_false", "fill_blank", "open_ended")

VISION_SYSTEM_PROMPT = (
    "你是屏幕答题助手。用户会发送一张答题界面截图。"
    "你必须独立解题，不要依赖界面中已选中、高亮或勾选的状态。"
    "只输出一个合法 JSON 对象，不要输出 Markdown、代码块或任何解释文字。"
)

VISION_PROMPT = """请阅读截图中的题目并作答。

要求：
1. 识别题干、全部选项及其原始标记（A/B/C/D、1/2/3、①/②、一/二等）。
2. 判断题型：single_choice(单选)、multiple_choice(多选)、true_false(判断)、fill_blank(填空)、open_ended(问答)。
3. 单选题只选一个；多选题必须选出所有正确项，不能只选一个；判断题答案为“正确”或“错误”。
4. 选项题 answers 使用选项原始标记，按选项顺序排列；answer 为英文逗号连接的展示字符串。
5. 填空/问答把最简答案写入 answers；多个空用分号分隔。
6. 忽略浏览器导航、倒计时、按钮、已选状态等无关内容。
7. 如果截图明显不是题目，is_question=false，answers=[]。

只输出以下 JSON，不要加注释：
{
  "is_question": true,
  "question": "题干文本",
  "options": {"A": "选项内容", "B": "选项内容"},
  "question_type": "single_choice|multiple_choice|true_false|fill_blank|open_ended",
  "answers": ["A"],
  "answer": "A",
  "detail": "不超过50字的简要解析"
}"""


class DeepSeekSolver:
    """DeepSeek API解题器"""

    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        enable_search: bool = False,
        use_vision: bool = True,
        vision_model: Optional[str] = None,
        vision_detail: str = "high",
        vision_max_tokens: int = 4096,
        vision_timeout: int = 40,
        vision_max_retries: int = 2,
        text_max_tokens: int = 4096,
        text_timeout: int = 40,
        vision_max_image_side: int = 1600,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.enable_search = enable_search

        self.use_vision = use_vision
        self.vision_model = vision_model or model
        self.vision_detail = vision_detail
        self.vision_max_tokens = vision_max_tokens
        self.vision_timeout = vision_timeout
        self.vision_max_retries = max(1, int(vision_max_retries))
        self.text_max_tokens = text_max_tokens
        self.text_timeout = text_timeout
        self.vision_max_image_side = vision_max_image_side

        # 应用层已有重试逻辑，关闭 SDK 自带重试，避免叠加导致长时间阻塞
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

        logger.info(
            "DeepSeek解题器初始化完成，文本模型: %s，视觉模型: %s（%s），联网搜索: %s",
            model, self.vision_model, "开启" if use_vision else "关闭", enable_search,
        )

    @classmethod
    def from_config(cls, config: Dict):
        """从配置字典创建实例"""
        return cls(
            api_key=config.get('DEEPSEEK_API_KEY', ''),
            base_url=config.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            model=config.get('DEEPSEEK_MODEL', 'deepseek-flash'),
            enable_search=config.get('DEEPSEEK_ENABLE_SEARCH', False),
            use_vision=config.get('USE_DEEPSEEK_VISION', True),
            vision_model=config.get(
                'VISION_MODEL', config.get('DEEPSEEK_MODEL', 'deepseek-flash')
            ),
            vision_detail=config.get('VISION_DETAIL', 'high'),
            vision_max_tokens=config.get('VISION_MAX_TOKENS', 4096),
            vision_timeout=config.get('VISION_TIMEOUT', 40),
            vision_max_retries=config.get('VISION_MAX_RETRIES', 2),
            text_max_tokens=config.get('TEXT_MAX_TOKENS', 4096),
            text_timeout=config.get('TEXT_TIMEOUT', 40),
        )

    # ================== 结果契约 ==================

    def _error(
        self,
        detail: str,
        source: str = "error",
        error_type: str = "api",
        **extra: Any,
    ) -> Dict:
        return {
            "ok": False,
            "answer": "?",
            "answers": [],
            "detail": detail,
            "source": source,
            "question": extra.get("question", ""),
            "options": extra.get("options", {}),
            "question_type": extra.get("question_type", "unknown"),
            "raw_text": extra.get("raw_text", ""),
            "error": detail,
            "error_type": error_type,
        }

    @staticmethod
    def _ok(
        answer: str,
        answers: List[str],
        detail: str,
        source: str,
        question: str,
        question_type: str,
        options: Dict,
        raw_text: str = "",
    ) -> Dict:
        return {
            "ok": True,
            "answer": answer,
            "answers": answers,
            "detail": detail[:50],
            "source": source,
            "question": question,
            "options": options,
            "question_type": question_type,
            "raw_text": raw_text,
        }

    # ================== 文本入口（本地 OCR 兜底） ==================

    def solve(self, question_data: Dict) -> Dict:
        """解题主入口（文本路径）"""
        if not question_data or not question_data.get('is_valid'):
            return self._error("未识别题目", error_type="invalid")

        question = question_data.get('question', '')
        options = question_data.get('options', {}) or {}
        raw_text = question_data.get('raw_text', '')
        question_type = self._normalize_question_type(
            question_data.get('question_type', ''), options
        )

        if not question:
            return self._error("题目为空", error_type="invalid")

        logger.debug("开始解题: %s...", question[:50])

        result = self._call_api_with_retry(
            question, options, question_type, raw_text
        )
        result.setdefault("question", question)
        result.setdefault("options", options)
        result.setdefault("question_type", question_type)
        result.setdefault("raw_text", raw_text)
        return result

    def _normalize_question_type(
        self, question_type: Any, options: Optional[Dict] = None
    ) -> str:
        text = str(question_type or "").strip().lower()
        aliases = {
            "choice": "single_choice",
            "single": "single_choice",
            "单选": "single_choice",
            "单选题": "single_choice",
            "multiple": "multiple_choice",
            "多选": "multiple_choice",
            "多选题": "multiple_choice",
            "judge": "true_false",
            "判断": "true_false",
            "判断题": "true_false",
            "填空": "fill_blank",
            "填空题": "fill_blank",
            "问答": "open_ended",
            "简答": "open_ended",
        }
        text = aliases.get(text, text)
        if text in KNOWN_TYPES:
            return text
        if options:
            if self._looks_true_false(options):
                return "true_false"
            return "single_choice"
        return "open_ended"

    @staticmethod
    def _looks_true_false(options: Dict) -> bool:
        if len(options) != 2:
            return False
        values = "".join(str(v) for v in options.values()).replace(" ", "")
        return (
            ("正确" in values and "错误" in values)
            or ("对" in values and "错" in values)
            or ("√" in values and "×" in values)
        )

    def _call_api_with_retry(
        self, question: str, options: Dict, question_type: str, raw_text: str
    ) -> Dict:
        """调用文本 API，带重试；第 2 次起关闭联网搜索。"""
        last_error = "未知错误"

        for attempt in range(self.MAX_RETRIES):
            use_search = self.enable_search and attempt == 0
            try:
                t0 = time.time()
                result = self._call_api(
                    question, options, question_type, raw_text,
                    use_search=use_search,
                )
                t1 = time.time()
                logger.info(
                    "⏱ API单次调用: %.2fs (尝试%d/%d, 联网=%s)",
                    t1 - t0, attempt + 1, self.MAX_RETRIES, use_search,
                )
                if result.get("ok"):
                    return result
                last_error = result.get("detail", "未知错误")
            except Exception as e:  # pylint: disable=broad-except
                last_error = str(e)
                logger.warning(
                    "API调用失败 (尝试 %d/%d): %s",
                    attempt + 1, self.MAX_RETRIES, e,
                )

            if attempt < self.MAX_RETRIES - 1:
                wait_time = self.RETRY_DELAY * (attempt + 1)
                logger.debug("等待 %s 秒后重试...", wait_time)
                time.sleep(wait_time)

        logger.error("API调用全部失败: %s", last_error)
        return self._error(
            "API失败: %s" % str(last_error)[:40], error_type="api",
            question=question, options=options, question_type=question_type,
            raw_text=raw_text,
        )

    def _call_api(
        self,
        question: str,
        options: Dict,
        question_type: str,
        raw_text: str,
        use_search: bool = True,
    ) -> Dict:
        """调用DeepSeek文本 API 获取答案"""
        options_text = self._format_options(options)
        prompt = self._build_prompt(
            question, options_text, question_type, raw_text
        )

        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是答题助手。请根据题型直接输出标准答案，并补一行简短解析。"
                        "单选题只输出一个选项标记；多选题必须输出全部正确选项，禁止只给一个；"
                        "判断题输出“正确”或“错误”；填空题直接输出应填内容。"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": self.text_max_tokens,
            "timeout": self.text_timeout,
        }

        if use_search:
            kwargs["extra_body"] = {"enable_search": True}

        response = self.client.chat.completions.create(**kwargs)
        content = self._extract_message_text(response, "文本API", allow_reasoning=True)
        if not content:
            return self._error("空响应", error_type="empty")

        result = self._parse_response(content, question_type, options, question, raw_text)

        if use_search and result.get("ok"):
            search_info = self._extract_search_citations(response)
            if search_info:
                result["detail"] = (result.get("detail", "") + search_info)[:80]

        return result

    @staticmethod
    def _extract_message_text(
        response: Any, tag: str, allow_reasoning: bool = False
    ) -> str:
        """安全读取 content，并记录 finish_reason/usage 便于诊断空响应。"""
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                logger.warning("%s 响应没有 choices", tag)
                return ""
            choice = choices[0]
            message = getattr(choice, "message", None)
            content = (getattr(message, "content", None) or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(response, "usage", None)
            reasoning = (getattr(message, "reasoning_content", None) or "").strip()

            if content:
                return content

            logger.warning(
                "%s 空响应: finish_reason=%s, usage=%s, reasoning_len=%d",
                tag, finish_reason, usage, len(reasoning),
            )
            if allow_reasoning and reasoning:
                match = re.search(r'答案[：:]\s*.+', reasoning)
                if match:
                    return match.group(0)
            return ""
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("%s 读取响应失败: %s", tag, e)
            return ""

    def _extract_search_citations(self, response) -> str:
        """从响应中提取联网搜索引用"""
        try:
            citations = getattr(response.choices[0].message, 'search_results', None)
            if citations and len(citations) > 0:
                sources = []
                for c in citations[:3]:
                    title = getattr(c, 'title', '') or getattr(c, 'name', '') or ''
                    url = getattr(c, 'url', '') or ''
                    if title and url:
                        sources.append(f"{title[:30]}: {url[:50]}")
                if sources:
                    return " | 参考: " + "; ".join(sources)
        except Exception:  # pylint: disable=broad-except
            pass
        return ""

    def _format_options(self, options: Dict) -> str:
        if not options:
            return "无选项"
        lines = []
        # 保持 OCR/视觉返回的选项顺序，不按字母排序
        for key, value in options.items():
            value = str(value)
            if len(value) > 80:
                value = value[:77] + "..."
            lines.append(f"{key}. {value}")
        return '\n'.join(lines)

    def _build_prompt(
        self, question: str, options_text: str, question_type: str, raw_text: str
    ) -> str:
        answer_rule = {
            "single_choice": "这是单选题，只输出一个选项标记，例如 A。",
            "choice": "这是单选题，只输出一个选项标记，例如 A。",
            "multiple_choice": (
                "这是多选题，必须输出所有正确选项标记，不能只输出一个；"
                "按选项顺序用英文逗号分隔，例如 A,C。"
            ),
            "true_false": "这是判断题，请只输出“正确”或“错误”。",
            "fill_blank": "这是填空题，请直接输出应填内容；多个空用分号分隔。",
            "open_ended": "这不是标准选择题，请输出最精简的直接答案。",
        }.get(question_type, "请输出最精简的直接答案。")
        return f"""题型：{question_type}
题目：{question}

选项：
{options_text}

OCR原文：
{raw_text[:800]}

要求：
1. {answer_rule}
2. 严格按以下格式输出两行：
答案：...
解析：..."""

    def _parse_response(
        self, text: str, question_type: str, options: Dict,
        question: str = "", raw_text: str = "",
    ) -> Dict:
        if not text:
            return self._error("空响应", error_type="empty", question=question,
                               options=options, question_type=question_type,
                               raw_text=raw_text)

        answer = self._extract_answer(text, question_type, options)
        detail = ""

        detail_patterns = [
            r'解析[：:]\s*(.+?)(?:\n|$)',
            r'原因[：:]\s*(.+?)(?:\n|$)',
            r'因为[：:]\s*(.+?)(?:\n|$)',
        ]
        for pattern in detail_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                detail = match.group(1).strip()
                break

        if not detail:
            colon_match = re.search(r'[：:](.+?)(?:\n|$)', text)
            if colon_match:
                potential = colon_match.group(1).strip()
                if potential and potential.upper() not in ['A', 'B', 'C', 'D']:
                    detail = potential

        detail = detail[:50] if detail else ""
        ok = answer not in ("", "?")
        if not ok:
            return self._error("无法解析答案", error_type="invalid_answer",
                               question=question, options=options,
                               question_type=question_type, raw_text=raw_text)

        answers = answer.split(",") if "," in answer else [answer]
        return self._ok(answer, answers, detail, "DeepSeek",
                        question, question_type, options, raw_text)

    def _extract_answer(self, text: str, question_type: str, options: Dict) -> str:
        answer_line_match = re.search(
            r'答案[：:]\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL
        )
        answer_line = answer_line_match.group(1).strip() if answer_line_match else ""
        first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
        candidate = answer_line or first_line
        candidate = re.split(r'解析[：:]', candidate)[0].strip()

        if question_type == "true_false":
            normalized = normalize_true_false(candidate)
            if normalized:
                return normalized

        if question_type in CHOICE_TYPES and options:
            labels = normalize_answer_labels(
                split_answer_tokens(candidate), options
            )
            if not labels:
                raw_labels = re.findall(
                    r'(?<![A-Za-z0-9])[A-Za-z](?![A-Za-z0-9])|[①-⑳]|\d{1,2}',
                    candidate,
                )
                labels = normalize_answer_labels(raw_labels, options)
            if labels:
                if question_type != "multiple_choice":
                    if len(labels) > 1:
                        logger.warning(
                            "单选题匹配到多个标签 %s，只取第一个", labels
                        )
                    labels = labels[:1]
                return ",".join(labels)

        if candidate:
            candidate = candidate.strip('，,。.;； ')
            if candidate:
                return candidate[:80]

        return "?"

    # ================== 视觉入口 ==================

    def solve_image(self, image: Any) -> Dict:
        """
        将截图发给视觉模型，一次完成读题、判题型与作答。
        永不向上抛异常；失败时返回 ok=False 的契约，由 main.py 回退 OCR。
        """
        if not self.use_vision:
            return self._error("视觉模式未启用", error_type="disabled")

        try:
            data_url = self._encode_image_data_url(image)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("图片编码失败: %s", e)
            return self._error("图片编码失败: %s" % str(e)[:40], error_type="image")

        max_attempts = self.vision_max_retries
        last_error = "视觉识别失败"
        last_type = "api"
        use_json_mode = True
        repair_hint = ""

        for attempt in range(max_attempts):
            try:
                content = self._call_vision_once(
                    data_url, repair_hint, use_json_mode
                )
            except Exception as e:  # pylint: disable=broad-except
                message = str(e)
                last_error = message[:60]
                if use_json_mode and (
                    "response_format" in message or "json" in message.lower()
                ):
                    logger.warning(
                        "视觉JSON模式不可用，改为普通模式重试: %s", message
                    )
                    use_json_mode = False
                    continue
                last_type = "timeout" if "timeout" in message.lower() else "api"
                logger.warning(
                    "视觉请求失败 (尝试 %d/%d): %s",
                    attempt + 1, max_attempts, message,
                )
                if attempt < max_attempts - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                continue

            if not content:
                last_error = "视觉空响应"
                last_type = "empty"
                repair_hint = "上次输出为空，请直接输出一个 JSON 对象。"
                if attempt < max_attempts - 1:
                    time.sleep(self.RETRY_DELAY)
                continue

            payload = self._parse_vision_response(content)
            if payload is None:
                last_error = "视觉返回非法JSON"
                last_type = "json"
                repair_hint = (
                    "上次输出不是合法 JSON，只输出一个 JSON 对象，"
                    "不要 Markdown 代码块。"
                )
                if attempt < max_attempts - 1:
                    time.sleep(self.RETRY_DELAY)
                continue

            result = self._normalize_vision_payload(payload)
            if result.get("ok"):
                return result
            if result.get("error_type") == "not_question":
                return result

            last_error = result.get("error", last_error)
            last_type = result.get("error_type", "invalid_answer")
            if attempt < max_attempts - 1:
                time.sleep(self.RETRY_DELAY)
            continue

        logger.warning("视觉识别失败，交给本地OCR回退: %s", last_error)
        return self._error(last_error, error_type=last_type)

    def _call_vision_once(
        self, data_url: str, repair_hint: str, use_json_mode: bool
    ) -> str:
        prompt = VISION_PROMPT
        if repair_hint:
            prompt = prompt + "\n\n" + repair_hint

        kwargs = {
            "model": self.vision_model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": self.vision_detail,
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.vision_max_tokens,
            "timeout": self.vision_timeout,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # 视觉请求不发送 enable_search：题目已在图中，搜索只会增加延迟。
        response = self.client.chat.completions.create(**kwargs)
        return self._extract_message_text(response, "视觉API", allow_reasoning=False)

    def _encode_image_data_url(self, image: Any) -> str:
        from PIL import Image  # pylint: disable=import-outside-toplevel

        if isinstance(image, Image.Image):
            img = image
        else:
            import numpy as np  # pylint: disable=import-outside-toplevel
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = Image.open(image)

        img = img.convert("RGB")
        max_side = max(img.size)
        if max_side > self.vision_max_image_side:
            scale = self.vision_max_image_side / float(max_side)
            new_size = (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "data:image/png;base64," + encoded

    @staticmethod
    def _parse_vision_response(content: str) -> Optional[Dict]:
        """去掉 Markdown 围栏并提取第一个 JSON 对象。"""
        if not content:
            return None

        text = content.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass

        start = text.find('{')
        if start == -1:
            return None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        end = text.rfind('}')
        if end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return None

    def _normalize_vision_payload(self, payload: Dict) -> Dict:
        """校验视觉返回的 JSON 并转换为统一契约。"""
        if not isinstance(payload, dict):
            return self._error("视觉结果格式错误", error_type="json")

        is_question = payload.get("is_question", True)
        if is_question is False or str(is_question).strip().lower() in (
            "false", "0", "no", "否",
        ):
            return self._error("非题目画面", error_type="not_question")

        question = str(payload.get("question") or "").strip()
        options = self._normalize_vision_options(payload.get("options"))

        question_type = self._normalize_question_type(
            payload.get("question_type") or payload.get("type") or "", options
        )

        raw_answers = payload.get("answers")
        if raw_answers is None:
            raw_answers = payload.get("correct_options")
        if raw_answers is None:
            raw_answers = payload.get("answer")
        if raw_answers is None:
            raw_answers = []

        raw_detail = payload.get("detail") or payload.get("explanation") or ""
        detail = str(raw_detail).strip()[:50]

        if not question:
            return self._error("视觉未识别到题干", error_type="invalid_answer")

        if question_type in CHOICE_TYPES and options:
            labels = normalize_answer_labels(raw_answers, options)
            if not labels:
                answer_text = payload.get("answer")
                if answer_text:
                    labels = normalize_answer_labels(
                        split_answer_tokens(answer_text), options
                    )
            if not labels:
                return self._error(
                    "视觉答案无法映射到选项", error_type="invalid_answer",
                    question=question, options=options,
                    question_type=question_type,
                )

            # 模型把多选标成单选时，按标签数量强制纠正
            if len(labels) >= 2:
                question_type = "multiple_choice"
            elif question_type != "multiple_choice":
                labels = labels[:1]

            return self._ok(
                ",".join(labels), labels, detail, "DeepSeek视觉",
                question, question_type, options,
            )

        if question_type == "true_false":
            normalized = normalize_true_false(raw_answers)
            if not normalized:
                normalized = normalize_true_false(payload.get("answer"))
            if not normalized:
                return self._error(
                    "视觉判断题答案无法识别", error_type="invalid_answer",
                    question=question, options=options,
                    question_type=question_type,
                )
            return self._ok(
                normalized, [normalized], detail, "DeepSeek视觉",
                question, question_type, options,
            )

        if isinstance(raw_answers, (list, tuple)):
            items = [str(item).strip() for item in raw_answers if str(item).strip()]
        else:
            items = split_answer_tokens(raw_answers)
        if not items and payload.get("answer"):
            items = [str(payload["answer"]).strip()]
        if not items:
            return self._error(
                "视觉答案为空", error_type="invalid_answer",
                question=question, options=options,
                question_type=question_type,
            )

        answer = "；".join(items)
        return self._ok(
            answer, items, detail, "DeepSeek视觉",
            question, question_type, options,
        )

    @staticmethod
    def _normalize_vision_options(raw_options: Any) -> Dict[str, str]:
        options: Dict[str, str] = {}
        if isinstance(raw_options, dict):
            for key, value in raw_options.items():
                label = normalize_option_label(key)
                content = str(value).strip()
                if label and content:
                    options[label] = content[:200]
        elif isinstance(raw_options, list):
            for index, value in enumerate(raw_options):
                content = str(value).strip()
                if content:
                    options[chr(ord('A') + index)] = content[:200]
        return options


if __name__ == "__main__":
    print("=" * 50)
    print("DeepSeek解题器测试")
    print("=" * 50)

    test_question = {
        "question": "2025年联合国气候变化大会(COP30)在哪举办？",
        "options": {
            "A": "日内瓦",
            "B": "纽约",
            "C": "里约热内卢",
            "D": "贝伦"
        },
        "question_type": "single_choice",
        "is_valid": True,
        "raw_text": "",
    }

    try:
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        solver = DeepSeekSolver(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
        )
        print("\n测试解题...")
        result = solver.solve(test_question)
        print("\n结果:")
        print(f"  ok: {result['ok']}")
        print(f"  答案: {result['answer']}")
        print(f"  解析: {result['detail']}")
        print(f"  来源: {result['source']}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"\n❌ 测试失败: {e}")
