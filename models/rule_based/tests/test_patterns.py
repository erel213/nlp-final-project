"""Unit tests for the custom confidential recognizers (comment 007).

Per model-rule-based.md testing requirements, every pattern has at least a
true-positive (canonical), a true-positive (obfuscated), and a true-negative
(near-miss that must NOT fire). Tests run the recognizers directly against
plain text so they do not require the full spaCy pipeline / model download.
"""

from __future__ import annotations

import pytest

from models.rule_based.patterns import (
    GENERIC_ID,
    PASSWORD_SECRET,
    PROPRIETARY_MARK,
    build_generic_id_recognizer,
    build_password_recognizer,
    build_proprietary_recognizer,
)


def _fires(recognizer, text: str, entity: str) -> bool:
    results = recognizer.analyze(text=text, entities=[entity], nlp_artifacts=None)
    return len(results) > 0


# --------------------------------------------------------------------------
# Password / secret recognizer
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def password_rec():
    return build_password_recognizer()


def test_password_canonical(password_rec):
    assert _fires(password_rec, "password: r]iD1#8", PASSWORD_SECRET)


def test_password_obfuscated_equals(password_rec):
    assert _fires(password_rec, "pwd=Be~o}.zq8", PASSWORD_SECRET)


def test_password_api_token(password_rec):
    assert _fires(
        password_rec, "api_key: AKIA1234567890ABCDEF", PASSWORD_SECRET
    )


def test_password_bare_word_negative(password_rec):
    # bare keyword with no value must NOT fire (avoids "reset your password")
    assert not _fires(password_rec, "please reset your password soon", PASSWORD_SECRET)


# --------------------------------------------------------------------------
# Generic passport / ID / driver-license recognizer
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def id_rec():
    return build_generic_id_recognizer()


def test_id_canonical_shape(id_rec):
    # ai4privacy IDCARD example
    assert _fires(id_rec, "ID number KB90324ER on file", GENERIC_ID)


def test_id_with_context_keyword(id_rec):
    assert _fires(id_rec, "passport no. L898902C3 issued", GENERIC_ID)


def test_id_obfuscated_driver_license(id_rec):
    assert _fires(id_rec, "Driving Licence: AB1234567", GENERIC_ID)


def test_id_plain_word_negative(id_rec):
    # ordinary short number / word must NOT fire
    assert not _fires(id_rec, "the meeting is at room 12", GENERIC_ID)


# --------------------------------------------------------------------------
# Proprietary vocabulary recognizer
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prop_rec():
    return build_proprietary_recognizer()


def test_proprietary_canonical(prop_rec):
    assert _fires(prop_rec, "This document is CONFIDENTIAL.", PROPRIETARY_MARK)


def test_proprietary_multiword(prop_rec):
    assert _fires(prop_rec, "Internal only — do not distribute", PROPRIETARY_MARK)


def test_proprietary_negative(prop_rec):
    assert not _fires(prop_rec, "This is a public newsletter.", PROPRIETARY_MARK)
