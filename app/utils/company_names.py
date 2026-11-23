"""Shared normalization helpers for company-name matching and deduplication."""

import re
import unicodedata


LEGAL_SUFFIXES = {
    "ag", "aps", "bv", "co", "company", "corp", "corporation",
    "gmbh", "inc", "incorporated", "llc", "llp", "lp", "ltd",
    "limited", "plc", "pty", "sa", "sarl"
}


def normalize_company_name(value: str) -> str:
    """Normalize punctuation, accents, case, and trailing legal suffixes."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[a-z0-9]+", normalized.casefold())
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def company_name_matches_text(company_name: str, text: str) -> bool:
    """Match the normalized company name as a complete phrase in text."""
    name = normalize_company_name(company_name)
    body = normalize_company_name(text)
    if not name or not body:
        return False
    return f" {name} " in f" {body} "
