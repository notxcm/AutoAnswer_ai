"""
答案与选项标签归一化工具。

OCR（本地兜底）和 DeepSeek（视觉/文本）产出的标签口径可能不同：
A/B/C/D、1/2/3、①/②、一/二…… 这里统一转换成可比较的形式。
"""

import re
from typing import Any, Dict, List

CIRCLED_NUM_MAP: Dict[str, str] = {
    '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
    '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
    '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
    '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20',
}
CN_NUM_MAP: Dict[str, str] = {
    '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
    '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
}

_TRIM_CHARS = " \t\r\n.,，、;；:：()（）[]【】<>《》"


def normalize_option_label(label: Any) -> str:
    """把单个选项标记统一成大写字母或数字字符串。"""
    if label is None:
        return ""
    text = str(label).strip().strip(_TRIM_CHARS)
    if not text:
        return ""
    if text in CIRCLED_NUM_MAP:
        return CIRCLED_NUM_MAP[text]
    if text in CN_NUM_MAP:
        return CN_NUM_MAP[text]
    if text.isdigit():
        return str(int(text))
    if text.isalpha() and len(text) == 1:
        return text.upper()
    return text


def _label_ordinal(label: str) -> int:
    """A->1, B->2, ..., Z->26; '3'->3; 其他返回 0。"""
    normalized = normalize_option_label(label)
    if not normalized:
        return 0
    if normalized.isdigit():
        return int(normalized)
    if len(normalized) == 1 and "A" <= normalized <= "Z":
        return ord(normalized) - ord("A") + 1
    return 0


def _normalize_separators(text: str) -> str:
    """A和C / A与C / A及C 这类写法先转成逗号。"""
    return re.sub(
        r'(?<=[A-Za-z0-9①-⑳])\s*[和与及]\s*(?=[A-Za-z0-9①-⑳])',
        ',',
        text,
    )


def split_answer_tokens(value: Any) -> List[str]:
    """把 list/str 形式的答案拆成 token，兼容中英文分隔符。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tokens: List[str] = []
        for item in value:
            tokens.extend(split_answer_tokens(item))
        return tokens

    text = _normalize_separators(str(value).strip())
    if not text:
        return []
    return [token for token in re.split(r'[,，、;；/|\\\s]+', text) if token]


def _expand_label_token(token: str) -> List[str]:
    """把 AC / A③ 这种连写标签拆成单个标签候选。"""
    if len(token) <= 1:
        return []
    chars = list(token)
    if not all(("A" <= c.upper() <= "Z" and c.isalpha()) or c in CIRCLED_NUM_MAP
               for c in chars):
        return []
    expanded: List[str] = []
    for char in chars:
        if char in CIRCLED_NUM_MAP:
            expanded.append(CIRCLED_NUM_MAP[char])
        else:
            expanded.append(char.upper())
    return expanded


def _match_token(
    token: str,
    option_labels: List[str],
    normalized_map: Dict[str, str],
    ordinal_map: Dict[int, str],
    normalized_option_text: Dict[str, str],
) -> List[str]:
    """把一个 token 映射到实际选项标签。"""
    normalized = normalize_option_label(token)
    if not normalized:
        return []

    # 1) 精确匹配
    if normalized in normalized_map:
        return [normalized_map[normalized]]

    # 2) 连写标签：AC -> A, C
    expanded = _expand_label_token(normalized)
    if expanded:
        matched = [normalized_map[x] for x in expanded if x in normalized_map]
        if len(matched) == len(expanded):
            return matched

    # 3) 字母与数字按序号等价：A <-> 1
    ordinal = _label_ordinal(normalized)
    if ordinal and ordinal in ordinal_map:
        return [ordinal_map[ordinal]]

    # 4) 按选项文本匹配
    for label in option_labels:
        option_text = normalized_option_text.get(label, "")
        if not option_text:
            continue
        if normalized == option_text:
            return [label]
        if len(normalized) >= 2 and option_text.startswith(normalized):
            return [label]

    return []


def normalize_answer_labels(raw: Any, options: Dict[str, str]) -> List[str]:
    """
    把模型输出映射为 options 中真实存在的标签，并按选项原始顺序返回。
    无法映射的 token 会被丢弃。
    """
    if not options:
        return []

    option_labels = list(options.keys())
    normalized_map: Dict[str, str] = {}
    ordinal_map: Dict[int, str] = {}
    normalized_option_text: Dict[str, str] = {}

    for label in option_labels:
        normalized = normalize_option_label(label)
        if normalized:
            normalized_map.setdefault(normalized, label)
        ordinal = _label_ordinal(label)
        if ordinal:
            ordinal_map.setdefault(ordinal, label)
        normalized_option_text[label] = normalize_option_label(options[label])

    matched = set()
    for token in split_answer_tokens(raw):
        for label in _match_token(
            token, option_labels, normalized_map, ordinal_map,
            normalized_option_text,
        ):
            matched.add(label)

    return [label for label in option_labels if label in matched]


def normalize_true_false(value: Any) -> str:
    """判断题答案归一化为“正确”/“错误”，无法识别返回空串。"""
    text = str(value or "").strip().replace(" ", "")
    if not text:
        return ""
    if re.search(r'不正确|错误|不对|错|×|✗|false', text, re.IGNORECASE):
        return "错误"
    if re.search(r'正确|对|√|✓|true', text, re.IGNORECASE):
        return "正确"
    return ""
