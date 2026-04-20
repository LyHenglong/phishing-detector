"""
feature_extraction.py
---------------------
Extracts all 30 features from a raw URL string.
Used at both training time (via generate_dataset) and inference time (predict.py / API).
Keeping extraction in one place guarantees training ↔ inference parity.
"""

import re
import math
import ipaddress
from urllib.parse import urlparse, parse_qs
from collections import Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "logon", "verify", "verification",
    "account", "update", "banking", "secure", "security", "alert",
    "confirm", "password", "credential", "wallet", "paypal", "ebay",
    "amazon", "apple", "microsoft", "google", "facebook", "netflix",
    "support", "service", "help", "recover", "unlock", "suspend",
]

SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top",
    ".click", ".link", ".online", ".site", ".win", ".bid",
    ".loan", ".party", ".trade", ".work", ".racing", ".review",
}

PHISHING_EXTENSIONS = {".php", ".asp", ".aspx", ".cgi", ".do", ".action"}

FEATURE_COLUMNS = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_at_symbols", "num_ampersands", "num_equals_signs", "num_question_marks",
    "num_special_chars", "num_digits", "digit_ratio",
    "has_https", "has_ip_address", "num_subdomains", "domain_entropy",
    "has_suspicious_keyword", "num_suspicious_keywords",
    "has_suspicious_tld", "has_phishing_extension",
    "path_depth", "has_port", "url_entropy",
    "token_count", "longest_token_length",
    "has_double_slash_redirect", "has_hex_encoding",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string in bits."""
    if not s:
        return 0.0
    counts = Counter(s)
    total  = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _is_ip(host: str) -> bool:
    """Return True if host is a raw IPv4 address."""
    try:
        ipaddress.IPv4Address(host.split(":")[0])
        return True
    except ValueError:
        return False


def _count_special(url: str) -> int:
    """Non-alphanumeric, non-standard characters (beyond . - _ / @ & = ? %)."""
    return len(re.findall(r"[^a-zA-Z0-9.\-_/@&=?%:/#+]", url))


def _tokenize(url: str):
    """Split URL into alphanumeric tokens."""
    return re.findall(r"[a-zA-Z0-9]+", url)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract all 30 features from a URL string.

    Parameters
    ----------
    url : str
        Raw URL (with or without scheme).

    Returns
    -------
    dict
        Mapping of feature name → numeric value.
    """
    # Normalise scheme
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed      = urlparse(url)
    host        = parsed.hostname or ""
    path        = parsed.path or ""
    query       = parsed.query or ""
    url_lower   = url.lower()
    tokens      = _tokenize(url)

    # ── Size / layout ──────────────────────────────────────────────────────
    url_length    = len(url)
    domain_length = len(host)
    path_length   = len(path)
    query_length  = len(query)

    # ── Character counts ───────────────────────────────────────────────────
    num_dots          = url.count(".")
    num_hyphens       = url.count("-")
    num_underscores   = url.count("_")
    num_slashes       = url.count("/")
    num_at_symbols    = url.count("@")
    num_ampersands    = url.count("&")
    num_equals_signs  = url.count("=")
    num_question_marks = url.count("?")
    num_special_chars = _count_special(url)
    num_digits        = sum(c.isdigit() for c in url)
    digit_ratio       = num_digits / url_length if url_length else 0.0

    # ── Scheme / host flags ────────────────────────────────────────────────
    has_https     = int(parsed.scheme == "https")
    has_ip_address = int(_is_ip(host))

    # Subdomains: parts before the registered domain
    host_parts    = host.split(".")
    num_subdomains = max(0, len(host_parts) - 2)

    domain_entropy = _shannon_entropy(host)

    # ── Content signals ────────────────────────────────────────────────────
    kw_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
    has_suspicious_keyword  = int(len(kw_hits) > 0)
    num_suspicious_keywords = len(kw_hits)

    tld = "." + host_parts[-1].lower() if host_parts else ""
    has_suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    path_ext = re.search(r"\.[a-z]{2,5}$", path.lower())
    has_phishing_extension = int(
        path_ext is not None and path_ext.group() in PHISHING_EXTENSIONS
    )

    # ── Path depth ─────────────────────────────────────────────────────────
    path_depth = len([p for p in path.split("/") if p])

    # ── Port ───────────────────────────────────────────────────────────────
    has_port = int(parsed.port is not None and parsed.port not in (80, 443))

    # ── Entropy (full URL) ─────────────────────────────────────────────────
    url_entropy = _shannon_entropy(url)

    # ── Token analysis ─────────────────────────────────────────────────────
    token_count          = len(tokens)
    longest_token_length = max((len(t) for t in tokens), default=0)

    # ── Obfuscation ────────────────────────────────────────────────────────
    # double-slash in path (after removing the scheme ://)
    path_only = url.split("://", 1)[-1]
    has_double_slash_redirect = int("//" in path_only)

    has_hex_encoding = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))

    return {
        "url_length":               url_length,
        "domain_length":            domain_length,
        "path_length":              path_length,
        "query_length":             query_length,
        "num_dots":                 num_dots,
        "num_hyphens":              num_hyphens,
        "num_underscores":          num_underscores,
        "num_slashes":              num_slashes,
        "num_at_symbols":           num_at_symbols,
        "num_ampersands":           num_ampersands,
        "num_equals_signs":         num_equals_signs,
        "num_question_marks":       num_question_marks,
        "num_special_chars":        num_special_chars,
        "num_digits":               num_digits,
        "digit_ratio":              digit_ratio,
        "has_https":                has_https,
        "has_ip_address":           has_ip_address,
        "num_subdomains":           num_subdomains,
        "domain_entropy":           domain_entropy,
        "has_suspicious_keyword":   has_suspicious_keyword,
        "num_suspicious_keywords":  num_suspicious_keywords,
        "has_suspicious_tld":       has_suspicious_tld,
        "has_phishing_extension":   has_phishing_extension,
        "path_depth":               path_depth,
        "has_port":                 has_port,
        "url_entropy":              url_entropy,
        "token_count":              token_count,
        "longest_token_length":     longest_token_length,
        "has_double_slash_redirect": has_double_slash_redirect,
        "has_hex_encoding":         has_hex_encoding,
    }


def extract_feature_vector(url: str) -> list:
    """Return features as an ordered list matching FEATURE_COLUMNS."""
    feats = extract_features(url)
    return [feats[col] for col in FEATURE_COLUMNS]
