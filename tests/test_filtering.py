"""Qualification boundaries are conservative text checks, not AI predictions."""

import pytest

from feedback_analyzer.filtering import ignore_reason, normalize_message


@pytest.mark.parametrize("message", ["", " \t\r\n ", "hi", " HELLO! ", "Hey...", "hello,", "Hi?!"])
def test_blank_and_exact_greetings_are_ignored(message):
    assert ignore_reason(message) is not None


@pytest.mark.parametrize(
    "message",
    [
        "Hi, checkout crashes",
        "Hello app crashes",
        "App crashes",
        "Charged twice",
        "https://example.com/account does not load",
        "Please stop sending spam emails",
        "Hello, hope you are having a nice day!",
        "hello world",
        "Get a free phone at https://example.com",
    ],
)
def test_substantive_or_semantically_uncertain_text_remains_candidate(message):
    assert ignore_reason(message) is None


def test_normalization_does_not_change_the_original_string():
    original = "  Hi,\tcheckout\r\n crashes.  "
    assert normalize_message(original) == "Hi, checkout crashes."
    assert original == "  Hi,\tcheckout\r\n crashes.  "
