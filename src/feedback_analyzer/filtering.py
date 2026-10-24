"""Conservative text checks before a later semantic category classifier."""

import re


_GREETING_ONLY = re.compile(r"(?:hi|hello|hey)[\s!?.…,]*", flags=re.IGNORECASE)


def normalize_message(message: str) -> str:
    """Collapse whitespace for model input without modifying the original."""
    return " ".join(message.split())


def ignore_reason(message: str) -> str | None:
    """Ignore only blanks and an exact simple greeting, never semantic content.

    Longer social greetings, spam, and promotions remain candidates until a
    trained classifier is available. Short complaints and URLs are allowed.
    """
    normalized = normalize_message(message)
    if not normalized:
        return "Blank message."
    if _GREETING_ONLY.fullmatch(normalized):
        return "Exact greeting-only message."
    return None
