import re

class PIISanitizer:
    """Фильтрация персональных данных перед отправкой в Langfuse."""

    PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "CARD": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        "PHONE_RU": r'\b\+?[78][\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b',
        "INN": r'\b\d{10,12}\b',  # ИНН (упрощённый)
    }

    def sanitize(self, text: str) -> str:
        """Маскирует PII в тексте."""
        for name, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f'[{name}_REDACTED]', text)
        return text

    def has_pii(self, text: str) -> bool:
        """Проверяет наличие PII."""
        for pattern in self.PATTERNS.values():
            if re.search(pattern, text):
                return True
        return False


class InjectionDetector:
    """Детектор prompt injection атак."""

    PATTERNS = [
        (r"ignore\s+(previous|above|all)\s+(instructions?|rules?|prompts?)", 0.9),
        (r"forget\s+(everything|all|previous)", 0.8),
        (r"you\s+are\s+now\s+", 0.7),
        (r"(system|admin)\s*:\s*(override|reset|ignore)", 0.9),
        (r"SYSTEM\s*:", 0.8),
        (r"reveal\s+(your|the)\s+(system\s+)?prompt", 0.9),
        (r"what\s+(is|are)\s+your\s+(instructions?|rules?|prompt)", 0.7),
        (r"(игнорируй|забудь|отмени)\s+(предыдущие|все|прежние)", 0.9),
        (r"ты\s+теперь\s+", 0.7),
        (r"выведи\s+(системный\s+)?промпт", 0.9),
        (r"DAN|Do\s+Anything\s+Now", 0.8),
    ]

    def detect(self, text: str) -> dict:
        """Анализирует текст на наличие injection-паттернов."""
        max_score = 0.0
        matched_patterns = []

        for pattern, weight in self.PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                max_score = max(max_score, weight)
                matched_patterns.append(pattern)

        return {
            "risk_score": max_score,
            "is_suspicious": max_score >= 0.7,
            "matched_patterns": len(matched_patterns),
        }

