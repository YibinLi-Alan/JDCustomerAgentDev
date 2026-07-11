"""输出侧防护 —— 敏感信息脱敏(阶段六 P-B)。

答复发给用户前过一道检查:密钥、手机号、身份证号等模式脱敏替换。
**诚实边界**(安全报告如实写):朴素正则只防低级泄漏(原样复读),
防不了改写变形(如"密钥前六位是…");PII 识别是业界难题,本实现是纵深防御
的一层,不是保证。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: (名称, 模式, 替换文本)。顺序即检查顺序。
_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "[已脱敏:疑似密钥]"),
    (
        "id_card",
        re.compile(r"\b\d{17}[\dXx]\b"),
        "[已脱敏:疑似身份证号]",
    ),
    (
        "phone",
        re.compile(r"\b1[3-9]\d{9}\b"),
        "[已脱敏:疑似手机号]",
    ),
)


@dataclass(frozen=True)
class OutputCheck:
    """一次输出过滤的结果。

    Attributes:
        text: 脱敏后的文本。
        redactions: 命中的规则名列表(空 = 干净;trace/安全报告用)。
    """

    text: str
    redactions: list[str] = field(default_factory=list)


def filter_output(text: str) -> OutputCheck:
    """对即将发给用户的答复做敏感信息脱敏(出口必经;永不抛)。"""
    redactions: list[str] = []
    for name, pattern, replacement in _RULES:
        if pattern.search(text):
            redactions.append(name)
            text = pattern.sub(replacement, text)
    return OutputCheck(text=text, redactions=redactions)
