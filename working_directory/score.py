"""成绩等级判断。"""

import numbers


def classify_score(score):
    """根据成绩返回等级；成绩必须是 0 到 100 的数值。"""
    if isinstance(score, bool) or not isinstance(score, numbers.Real):
        raise TypeError("score 必须是数字")
    if not 0 <= score <= 100:
        raise ValueError("score 必须在 0 到 100 之间")

    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 60:
        return "C"
    return "D"
