"""
OCR engine module - RapidOCR (ONNX Runtime) text recognition.
Lightweight, no PaddlePaddle dependency, fast inference.
"""
import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from answer_utils import normalize_option_label

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCREngine:
    """RapidOCR engine: ONNX Runtime backend, lightweight and fast."""

    OPTION_LINE_RE = re.compile(
        r'^\s*(?:[☐□☑✔√■]\s*)?'
        r'(?P<label>[A-Za-z]|\d{1,2}|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|'
        r'[一二三四五六七八九十])\s*[\.\、．:：\)）]\s*(?P<content>.+?)\s*$'
    )
    INLINE_LETTER_OPTION_RE = re.compile(
        r'(?<![A-Za-z0-9])(?P<label>[A-Z])\s*[\.\、．:：\)）]\s*'
    )
    FILL_BLANK_PATTERNS: List[str] = [
        r'_{2,}',
        r'（\s*）',
        r'\(\s*\)',
        r'【\s*】',
        r'\[\s*\]',
        r'____+',
    ]
    TRUE_FALSE_HINTS: Tuple[str, ...] = (
        '判断题', '判断下列', '对错', '是否正确', '是否错误',
        '是非题'
    )
    MULTIPLE_CHOICE_HINTS: Tuple[str, ...] = (
        '多选题', '多选', '多项选择', '不定项', '至少两项',
        '全部正确', '选出所有',
    )
    CHECKBOX_GLYPHS = '□☐'
    OPTION_PATTERNS: Dict[str, List[str]] = {
        'A': [
            r'A[．、.、:：]\s*(.+?)(?=(?:[B-D][．、.、:：]|\Z))',
            r'A[）)]\s*(.+?)(?=(?:[B-D][）)]|\Z))',
            r'A\s{2,}(.+?)(?=(?:[B-D]\s{2,}|\Z))',
            r'①\s*[Aa]\.?\s*(.+?)(?=(?:②\s*[Bb]|\Z))',
        ],
        'B': [
            r'B[．、.、:：]\s*(.+?)(?=(?:[C-D][．、.、:：]|\Z))',
            r'B[）)]\s*(.+?)(?=(?:[C-D][）)]|\Z))',
            r'B\s{2,}(.+?)(?=(?:[C-D]\s{2,}|\Z))',
            r'②\s*[Bb]\.?\s*(.+?)(?=(?:③\s*[Cc]|\Z))',
        ],
        'C': [
            r'C[．、.、:：]\s*(.+?)(?=(?:D[．、.、:：]|\Z))',
            r'C[）)]\s*(.+?)(?=(?:D[）)]|\Z))',
            r'C\s{2,}(.+?)(?=(?:D\s{2,}|\Z))',
            r'③\s*[Cc]\.?\s*(.+?)(?=(?:④\s*[Dd]|\Z))',
        ],
        'D': [
            r'D[．、.、:：]\s*(.+?)(?=\Z)',
            r'D[）)]\s*(.+?)(?=\Z)',
            r'D\s{2,}(.+?)(?=\Z)',
            r'④\s*[Dd]\.?\s*(.+?)(?=\Z)',
        ]
    }

    QUESTION_CLEAN_PATTERNS: List[Tuple[str, str]] = [
        (r'^\d+[\.\u3001\s]*', ''),
        (r'^[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]+[\.\u3001\s]*', ''),
        (r'^(\u9898\u76ee|\u95ee\u9898|\u8bf7\u56de\u7b54)'
         r'[\uff1a:]?\s*', ''),
        (r'^\s*', ''),
        (r'\s*$', ''),
    ]

    def __init__(self, use_gpu: bool = False, lang: str = 'ch',
                 show_log: bool = True) -> None:
        self.ocr: Any = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._show_log = show_log
        if not show_log:
            logging.getLogger('rapidocr_onnxruntime').setLevel(logging.WARNING)
        self._init_model()

    def _init_model(self) -> None:
        try:
            logger.info("Loading RapidOCR [ONNX Runtime]...")
            from rapidocr_onnxruntime import RapidOCR  # pylint: disable=import-outside-toplevel

            self.ocr = RapidOCR()

            # Warm-up with a blank image
            warm = np.zeros((60, 120, 3), dtype=np.uint8)
            warm[:] = 255
            _ = self.ocr(warm)
            logger.info("OCR model warm-up completed")

            logger.info("RapidOCR loaded successfully")
        except ImportError as exc:
            logger.error("RapidOCR import failed: %s", exc)
            raise ImportError(
                "pip install rapidocr_onnxruntime"
            ) from exc
        except Exception as exc:
            logger.error("RapidOCR init failed: %s", exc)
            raise RuntimeError(
                "OCR engine init failed: %s" % exc
            ) from exc

    def recognize(self, image: Any) -> Dict[str, Any]:
        try:
            img_array = self._prepare_image(image)
            if img_array is None:
                return self._empty_result("unsupported image type")

            t0 = time.time()
            result, _ = self.ocr(img_array)
            t1 = time.time()

            if not result:
                logger.info("OCR: %.2fs -> no text", t1 - t0)
                return self._empty_result("no text detected")

            texts, confidences = self._extract_texts(result)
            t2 = time.time()

            if not texts:
                logger.info("OCR: %.2fs extract: %.2fs -> low confidence",
                            t1 - t0, t2 - t1)
                return self._empty_result("text confidence too low")

            full_text = '\n'.join(texts)
            avg_confidence = sum(confidences) / len(confidences)

            parsed = self._parse_question(full_text)
            t3 = time.time()
            parsed["confidence"] = avg_confidence
            logger.info(
                "OCR: inference=%.2fs extract=%.2fs parse=%.3fs | "
                "%d lines conf=%.2f",
                t1 - t0, t2 - t1, t3 - t2,
                len(texts), avg_confidence
            )
            return parsed
        except Exception as exc:
            logger.error("OCR recognition error: %s", exc)
            return self._empty_result(str(exc))

    def _prepare_image(self, image: Any) -> Optional[np.ndarray]:
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                return None
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            return img_array
        except Exception as exc:
            logger.error("Image conversion failed: %s", exc)
            return None

    def _extract_texts(self, ocr_result: Any
                       ) -> Tuple[List[str], List[float]]:
        """
        RapidOCR returns a list of lists:
        [[box_coords, text, confidence], ...]
        e.g. [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "识别文字", 0.95]
        """
        texts: List[str] = []
        confidences: List[float] = []
        if not ocr_result:
            return texts, confidences
        for item in ocr_result:
            # Each item is [box, text, score]
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = str(item[1])
                score = float(item[2])
                if score > 0.2:
                    texts.append(text)
                    confidences.append(score)
        return texts, confidences

    def _parse_question(self, text: str) -> Dict[str, Any]:
        if not text:
            return self._empty_result("text is empty")
        lines = self._normalize_lines(text)
        text = '\n'.join(lines)

        options, option_start_index = self._extract_options_from_lines(lines)
        if options:
            question = self._extract_question_from_lines(
                lines, option_start_index
            )
        else:
            options = self._extract_inline_options(text)
            question = self._extract_question(text, options)

        question = self._clean_question(question)
        question_type = self._detect_question_type(question, options, text)
        return {
            "question": question[:300],
            "options": options,
            "raw_text": text,
            "question_type": question_type,
            "is_valid": self._is_valid_question(question, options, question_type),
            "confidence": 0.0,
        }

    def _normalize_lines(self, text: str) -> List[str]:
        lines: List[str] = []
        for line in text.split('\n'):
            cleaned = re.sub(r'[^\S\n]+', ' ', line).strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    def _normalize_option_label(self, label: str) -> str:
        return normalize_option_label(label)

    def _extract_options_from_lines(
        self, lines: List[str]
    ) -> Tuple[Dict[str, str], int]:
        options: Dict[str, str] = {}
        current_label: Optional[str] = None
        option_start_index = len(lines)

        for index, line in enumerate(lines):
            match = self.OPTION_LINE_RE.match(line)
            if match:
                label = self._normalize_option_label(match.group('label'))
                content = match.group('content').strip()
                if option_start_index == len(lines):
                    option_start_index = index
                if content:
                    options[label] = content[:200]
                    current_label = label
                continue

            if current_label and option_start_index < len(lines):
                options[current_label] = (
                    options[current_label] + ' ' + line
                )[:200]

        return options, option_start_index

    def _extract_inline_options(self, text: str) -> Dict[str, str]:
        options: Dict[str, str] = {}
        matches = list(self.INLINE_LETTER_OPTION_RE.finditer(text))
        if len(matches) < 2:
            for letter in ['A', 'B', 'C', 'D']:
                option_content = self._extract_option(text, letter)
                if option_content:
                    options[letter] = option_content[:200].strip()
            return options

        for index, match in enumerate(matches):
            label = match.group('label').upper()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[start:end].strip(" \n\t,，;；")
            if content:
                options[label] = content[:200]
        return options

    def _extract_option(self, text: str, letter: str) -> Optional[str]:
        patterns = self.OPTION_PATTERNS.get(letter, [])
        for pattern in patterns:
            try:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if content and len(content) >= 1:
                        return content
            except re.error as exc:
                logger.warning("Regex error: %s", exc)
                continue
        return None

    def _extract_question(self, text: str,
                          options: Dict[str, str]) -> str:
        if not options:
            return text
        first_pos = len(text)
        for label in options:
            pattern = re.compile(
                r'(?<![A-Za-z0-9])%s\s*[\.\、．:：\)）]' % re.escape(label),
                re.IGNORECASE
            )
            match = pattern.search(text)
            if match and match.start() < first_pos:
                first_pos = match.start()
        if first_pos < len(text):
            question = text[:first_pos].strip()
        else:
            question = text
        return self._clean_question(question)

    def _extract_question_from_lines(
        self, lines: List[str], option_start_index: int
    ) -> str:
        if option_start_index <= 0:
            return '\n'.join(lines)
        return '\n'.join(lines[:option_start_index]).strip()

    def _clean_question(self, question: str) -> str:
        for pat, replacement in self.QUESTION_CLEAN_PATTERNS:
            question = re.sub(pat, replacement, question,
                              flags=re.IGNORECASE)
        return question.strip()

    def _detect_question_type(
        self, question: str, options: Dict[str, str], raw_text: str
    ) -> str:
        combined_text = f"{question}\n{raw_text}".strip()
        if self._is_true_false_question(question, options, combined_text):
            return "true_false"
        if self._is_fill_blank_question(combined_text, options):
            return "fill_blank"
        if self._is_multiple_choice_question(question, options, combined_text):
            return "multiple_choice"
        if options:
            return "single_choice"
        return "open_ended"

    def _is_multiple_choice_question(
        self, question: str, options: Dict[str, str], combined_text: str
    ) -> bool:
        """保守判断多选：仅凭明确提示词或复选框符号，不单看“全部/正确的有”。"""
        if len(options) < 2:
            return False
        if any(hint in combined_text for hint in self.MULTIPLE_CHOICE_HINTS):
            return True
        checkbox_count = sum(combined_text.count(c) for c in self.CHECKBOX_GLYPHS)
        return checkbox_count >= 2

    def _is_true_false_question(
        self, question: str, options: Dict[str, str], combined_text: str
    ) -> bool:
        option_values = ''.join(options.values())
        normalized = option_values.replace(' ', '')
        if len(options) > 2:
            return False
        if any(hint in combined_text for hint in self.TRUE_FALSE_HINTS):
            return True
        if '正确' in normalized and '错误' in normalized:
            return True
        if ('对' in normalized and '错' in normalized) or ('√' in normalized and '×' in normalized):
            return True
        if not options and re.search(r'(正确|错误|对|错|是否)', question):
            return True
        return False

    def _is_fill_blank_question(
        self, combined_text: str, options: Dict[str, str]
    ) -> bool:
        if options:
            return False
        if '填空题' in combined_text or '填空' in combined_text:
            return True
        return any(
            re.search(pattern, combined_text)
            for pattern in self.FILL_BLANK_PATTERNS
        )

    def _is_valid_question(
        self, question: str, options: Dict[str, str], question_type: str
    ) -> bool:
        question_length = len(question.replace('\n', '').strip())
        if question_type in ("choice", "single_choice", "multiple_choice"):
            return question_length >= 4 and len(options) >= 2
        if question_type == "true_false":
            return question_length >= 4
        if question_type in ("fill_blank", "open_ended"):
            return question_length >= 6
        return question_length >= 6 or len(options) >= 2

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "question": "",
            "options": {},
            "raw_text": "",
            "question_type": "unknown",
            "is_valid": False,
            "confidence": 0.0,
            "error": reason
        }

    def get_version_info(self) -> Dict[str, Any]:
        try:
            import rapidocr_onnxruntime  # pylint: disable=import-outside-toplevel
            return {
                "rapidocr_version": getattr(
                    rapidocr_onnxruntime, '__version__', 'unknown'
                ),
                "ocr_type": "RapidOCR (ONNX Runtime)",
                "language": self.lang,
                "gpu_enabled": self.use_gpu
            }
        except Exception as exc:
            logger.warning("Get version failed: %s", exc)
            return {"error": str(exc)}


if __name__ == "__main__":
    print("=" * 50)
    print("RapidOCR Fast Mode Test")
    print("=" * 50)
    try:
        engine = OCREngine(use_gpu=False, lang='ch')
        info = engine.get_version_info()
        print("\nEngine version: %s" % info)
        test_path = "test_question.png"
        if os.path.exists(test_path):
            print("\nRecognizing: %s" % test_path)
            img = Image.open(test_path)
            result = engine.recognize(img)
            print("\nResult:")
            q_text = str(result['question'])[:50]
            print("  question: %s..." % q_text)
            print("  options: %s" % result['options'])
            print("  valid: %s" % result['is_valid'])
            print("  confidence: %.3f" % result.get('confidence', 0))
        else:
            print("\nTest file not found: %s" % test_path)
    except Exception as exc:
        print("\nTest failed: %s" % exc)
