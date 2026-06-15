"""
Smoke tests for backend/generator.py.
These run on every Codex-generated PR to catch regressions before merge.
"""
import re


def test_suggest_prompt_has_no_angle_bracket_placeholders():
    """
    The SYSTEM_PROMPT must not instruct the model to output angle-bracket placeholders.
    Regression test for issues #12 and #11.
    """
    with open("backend/generator.py") as f:
        source = f.read()

    # Must not contain <q1>/<q2>/<q3> style tokens
    assert "<q1>" not in source, "SYSTEM_PROMPT must not contain literal placeholder <q1>"
    assert "<q2>" not in source, "SYSTEM_PROMPT must not contain literal placeholder <q2>"
    assert "<q3>" not in source, "SYSTEM_PROMPT must not contain literal placeholder <q3>"
    # Must explicitly tell the model NOT to use angle brackets
    assert "NO angle brackets" in source or "no angle brackets" in source, \
        "SYSTEM_PROMPT must explicitly forbid angle brackets around suggestions"


def test_suggest_marker_present_in_prompt():
    """The SYSTEM_PROMPT must still instruct the model to append the |||SUGGEST marker."""
    with open("backend/generator.py") as f:
        source = f.read()
    assert "|||SUGGEST" in source, "SYSTEM_PROMPT must reference the |||SUGGEST marker"


def test_generator_compiles():
    """backend/generator.py must be importable without syntax errors."""
    import py_compile
    py_compile.compile("backend/generator.py", doraise=True)
