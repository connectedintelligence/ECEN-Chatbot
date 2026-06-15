"""
Smoke tests for backend/generator.py.
These run on every Codex-generated PR to catch regressions before merge.
"""
import re


def test_suggest_prompt_has_no_angle_bracket_placeholders():
    """
    The SYSTEM_PROMPT must not instruct the model to output literal <q1>/<q2>/<q3>.
    Regression test for issue #12.
    """
    with open("backend/generator.py") as f:
        source = f.read()

    # Find the SYSTEM_PROMPT string (everything between the triple-quote delimiters)
    match = re.search(r'SYSTEM_PROMPT\s*=\s*["\']+(.*?)["\']+\s*\n', source, re.DOTALL)
    # Simpler check: the literal tokens <q1>, <q2>, <q3> must not appear in generator.py
    assert "<q1>" not in source, "SYSTEM_PROMPT must not contain literal placeholder <q1>"
    assert "<q2>" not in source, "SYSTEM_PROMPT must not contain literal placeholder <q2>"
    assert "<q3>" not in source, "SYSTEM_PROMPT must not contain literal placeholder <q3>"


def test_suggest_marker_present_in_prompt():
    """The SYSTEM_PROMPT must still instruct the model to append the |||SUGGEST marker."""
    with open("backend/generator.py") as f:
        source = f.read()
    assert "|||SUGGEST" in source, "SYSTEM_PROMPT must reference the |||SUGGEST marker"


def test_generator_compiles():
    """backend/generator.py must be importable without syntax errors."""
    import py_compile
    py_compile.compile("backend/generator.py", doraise=True)
