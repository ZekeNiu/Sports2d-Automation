from __future__ import annotations


def parse_float_list(text: str, default: list[float]) -> list[float]:
    stripped = text.strip()
    if not stripped:
        return default
    values = []
    for part in stripped.replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values or default


def parse_str_list(text: str, default: list[str]) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return default
    values = [part.strip() for part in stripped.replace(";", ",").split(",") if part.strip()]
    return values or default
