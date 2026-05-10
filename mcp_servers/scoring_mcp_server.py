"""
ScoringMCP — content quality scoring and compliance validation.

Tools:
  score_content(content, rules)    → quality score 0-100 with breakdown
  check_compliance(content, ret)   → retailer compliance validation
  check_brand_safety(content)      → brand safety flag detection
  diff_content(old, new)           → structured diff with improvement metrics
"""
import re
from dataclasses import dataclass, field

HAS_TEXTSTAT = False
try:
    import textstat
    # Verify NLTK data is available (textstat >= 0.7.4 requires cmudict)
    textstat.flesch_reading_ease("The quick brown fox.")
    HAS_TEXTSTAT = True
except Exception:
    HAS_TEXTSTAT = False


def _count_syllables_simple(word: str) -> int:
    """Syllable counter that works without NLTK data."""
    word = word.lower().strip(".,!?;:")
    if not word:
        return 0
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_simple(text: str) -> float:
    """Flesch Reading Ease without NLTK dependency."""
    import re
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = text.split()
    if not words:
        return 60.0
    syllables = sum(_count_syllables_simple(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return max(0.0, min(100.0, score))


# ── Scoring logic ─────────────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    total: float
    title_compliance: float
    bullet_compliance: float
    keyword_inclusion: float
    readability: float
    specificity: float
    description_quality: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def score_content(content: dict, rules: dict) -> dict:
    """
    Score product listing content against retailer rules.

    Returns a 0-100 quality score with component breakdown.
    Each dimension is independently scored and weighted.

    Weights:
      title_compliance    25%
      bullet_compliance   25%
      keyword_inclusion   20%
      readability         15%
      specificity         15%
    """
    issues = []
    suggestions = []
    scores = {}

    title = content.get("title", "")
    bullets = content.get("bullet_points", [])
    description = content.get("description", "")

    # ── Title scoring (25%) ───────────────────────────────────────────────────
    title_score = 100.0
    title_rules = rules.get("title", {})
    title_max = title_rules.get("max_chars", 200)
    title_min = title_rules.get("min_chars", 20)
    prohibited = title_rules.get("prohibited_words", [])

    if len(title) > title_max:
        excess = len(title) - title_max
        title_score -= min(40, excess * 0.5)
        issues.append(f"Title too long: {len(title)}/{title_max} chars (+{excess})")
    elif len(title) < title_min:
        title_score -= 30
        issues.append(f"Title too short: {len(title)}/{title_min} chars minimum")
    else:
        # Reward using title space well (80%+ of limit = ideal)
        utilization = len(title) / title_max
        if utilization < 0.6:
            title_score -= 15
            suggestions.append(f"Title underutilizes character limit ({len(title)}/{title_max})")

    for word in prohibited:
        if word.lower() in title.lower():
            title_score -= 20
            issues.append(f"Prohibited word in title: '{word}'")

    scores["title_compliance"] = max(0, title_score)

    # ── Bullet scoring (25%) ─────────────────────────────────────────────────
    bullet_score = 100.0
    bullet_rules = rules.get("bullet_points", {})
    expected_count = bullet_rules.get("count", 5)
    bullet_max = bullet_rules.get("max_chars_each", 255)

    if len(bullets) != expected_count:
        bullet_score -= 30
        issues.append(f"Wrong bullet count: {len(bullets)} (expected {expected_count})")

    for i, bullet in enumerate(bullets):
        if len(bullet) > bullet_max:
            bullet_score -= 10
            issues.append(f"Bullet {i+1} too long: {len(bullet)}/{bullet_max} chars")
        if bullet_rules.get("start_with_capital") and bullet and not bullet[0].isupper():
            bullet_score -= 5
            issues.append(f"Bullet {i+1} should start with capital letter")
        # Penalize weak bullets (too short = not informative)
        if len(bullet) < 40:
            bullet_score -= 8
            suggestions.append(f"Bullet {i+1} is too brief ({len(bullet)} chars) — add more detail")

    scores["bullet_compliance"] = max(0, bullet_score)

    # ── Keyword inclusion (20%) ───────────────────────────────────────────────
    keyword_score = 0.0
    search_benchmarks = rules.get("search_volume_benchmarks", {})
    if search_benchmarks:
        top_keywords = sorted(
            search_benchmarks.items(), key=lambda x: x[1], reverse=True
        )[:5]
        full_text = f"{title} {' '.join(bullets)} {description}".lower()
        matched = sum(1 for kw, _ in top_keywords if kw.lower() in full_text)
        keyword_score = (matched / len(top_keywords)) * 100
        if matched < len(top_keywords):
            missing = [kw for kw, _ in top_keywords if kw.lower() not in full_text]
            suggestions.append(f"Missing high-volume keywords: {', '.join(missing[:3])}")
    else:
        keyword_score = 70  # Neutral if no benchmark data

    scores["keyword_inclusion"] = keyword_score

    # ── Readability (15%) ─────────────────────────────────────────────────────
    full_text = f"{title}. {' '.join(bullets)}. {description}"
    if full_text.strip():
        try:
            fk_score = textstat.flesch_reading_ease(full_text) if HAS_TEXTSTAT else _flesch_simple(full_text)
        except Exception:
            fk_score = _flesch_simple(full_text)
        # Target 60-80 (standard/fairly easy) for consumer products
        if 60 <= fk_score <= 80:
            readability_score = 100
        elif 50 <= fk_score < 60 or 80 < fk_score <= 90:
            readability_score = 80
        elif 30 <= fk_score < 50 or 90 < fk_score <= 100:
            readability_score = 60
        else:
            readability_score = 40
            issues.append(f"Readability score {fk_score:.0f} is outside ideal range (60-80)")
    else:
        readability_score = 65  # Neutral fallback

    scores["readability"] = readability_score

    # ── Specificity (15%) ─────────────────────────────────────────────────────
    # Detect concrete vs vague language
    vague_phrases = [
        "high quality", "great", "amazing", "awesome", "perfect", "best",
        "good", "nice", "excellent", "superior", "premium quality",
        "state of the art", "cutting edge",
    ]
    specific_patterns = [
        r"\d+\s*(hours?|hrs?|minutes?|mins?)",  # time specs
        r"\d+\s*(mm|cm|inch|inches|ft|gb|mb|mhz|ghz|ohm|db)",  # measurements
        r"\d+\s*(count|pack|pieces?|ct\b)",  # quantities
        r"ip[x\d]\d",  # IP ratings
        r"bluetooth\s*\d+",  # bluetooth version
    ]

    full_lower = full_text.lower()
    vague_count = sum(1 for phrase in vague_phrases if phrase in full_lower)
    specific_count = sum(
        1 for pattern in specific_patterns if re.search(pattern, full_lower)
    )

    specificity_score = min(100, max(0, 60 + (specific_count * 8) - (vague_count * 10)))
    if vague_count > 3:
        suggestions.append(
            f"Replace vague language ({vague_count} instances) with specific claims and measurements"
        )

    scores["specificity"] = specificity_score

    # ── Weighted total ────────────────────────────────────────────────────────
    weights = {
        "title_compliance": 0.25,
        "bullet_compliance": 0.25,
        "keyword_inclusion": 0.20,
        "readability": 0.15,
        "specificity": 0.15,
    }
    total = sum(scores[k] * w for k, w in weights.items())

    return {
        "total_score": round(total, 1),
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "issues": issues,
        "suggestions": suggestions,
        "grade": _score_to_grade(total),
    }


def check_compliance(content: dict, retailer: str = "amazon") -> dict:
    """
    Hard compliance validation against retailer rules.
    Returns pass/fail per requirement with specific violation details.
    """
    from mcp_servers.retailer_mcp_server import get_retailer_requirements

    title = content.get("title", "")
    bullets = content.get("bullet_points", [])
    description = content.get("description", "")
    category = content.get("category", "electronics")

    rules = get_retailer_requirements(category, retailer)
    violations = []
    checks = {}

    # Title length
    title_max = rules.get("title", {}).get("max_chars", 200)
    title_min = rules.get("title", {}).get("min_chars", 20)
    checks["title_length"] = len(title) <= title_max and len(title) >= title_min
    if not checks["title_length"]:
        violations.append(f"Title length {len(title)} out of range [{title_min}, {title_max}]")

    # Prohibited words in title
    prohibited = rules.get("title", {}).get("prohibited_words", [])
    prohibited_found = [w for w in prohibited if w.lower() in title.lower()]
    checks["title_prohibited_words"] = len(prohibited_found) == 0
    if prohibited_found:
        violations.append(f"Prohibited words in title: {prohibited_found}")

    # Bullet count
    expected_bullets = rules.get("bullet_points", {}).get("count", 5)
    checks["bullet_count"] = len(bullets) == expected_bullets
    if not checks["bullet_count"]:
        violations.append(f"Expected {expected_bullets} bullets, got {len(bullets)}")

    # Bullet length
    bullet_max = rules.get("bullet_points", {}).get("max_chars_each", 255)
    long_bullets = [i + 1 for i, b in enumerate(bullets) if len(b) > bullet_max]
    checks["bullet_lengths"] = len(long_bullets) == 0
    if long_bullets:
        violations.append(f"Bullets {long_bullets} exceed {bullet_max} char limit")

    # Description length
    desc_max = rules.get("description", {}).get("max_chars", 2000)
    checks["description_length"] = len(description) <= desc_max
    if not checks["description_length"]:
        violations.append(f"Description length {len(description)} exceeds {desc_max} limit")

    return {
        "compliant": len(violations) == 0,
        "retailer": retailer,
        "checks": checks,
        "violations": violations,
        "violation_count": len(violations),
    }


def check_brand_safety(content: dict, brand_guidelines: dict | None = None) -> dict:
    """
    Detect brand safety issues in generated content.

    Checks for:
    - Prohibited claims from brand guidelines
    - Generic brand safety terms (unsubstantiated superlatives, medical claims)
    - Competitor name mentions
    """
    title = content.get("title", "")
    bullets = content.get("bullet_points", [])
    description = content.get("description", "")
    full_text = f"{title} {' '.join(bullets)} {description}".lower()

    flags = []

    # Generic prohibited terms (always unsafe)
    always_prohibited = [
        "cures", "treats", "prevents", "heals",     # medical claims
        "fda approved",                              # false regulatory claims
        "#1 best", "number one best",               # unsubstantiated rank
        "guaranteed for life",                      # vague lifetime guarantees
    ]
    for term in always_prohibited:
        if term in full_text:
            flags.append({"type": "prohibited_claim", "term": term, "severity": "high"})

    # Brand-specific prohibited claims
    if brand_guidelines:
        for claim in brand_guidelines.get("prohibited_claims", []):
            # Remove parenthetical notes for matching
            clean_claim = re.sub(r"\s*\(.*?\)", "", claim).strip().lower()
            if clean_claim and clean_claim in full_text:
                flags.append({
                    "type": "brand_prohibited",
                    "term": claim,
                    "severity": "medium",
                })

    # PII detection
    pii_patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b\d{16}\b", "credit card number"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
        (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone number"),
    ]
    for pattern, pii_type in pii_patterns:
        if re.search(pattern, full_text):
            flags.append({"type": "pii", "term": pii_type, "severity": "critical"})

    return {
        "passed": len(flags) == 0,
        "flags": flags,
        "flag_count": len(flags),
        "critical_count": sum(1 for f in flags if f["severity"] == "critical"),
        "high_count": sum(1 for f in flags if f["severity"] == "high"),
    }


def diff_content(old_content: dict, new_content: dict) -> dict:
    """
    Compare old and new content versions with structured improvement metrics.
    """
    def _field_diff(old: str, new: str, field_name: str) -> dict:
        return {
            "field": field_name,
            "old_length": len(old),
            "new_length": len(new),
            "length_delta": len(new) - len(old),
            "changed": old.strip() != new.strip(),
        }

    diffs = []
    fields = ["title", "description", "backend_keywords"]
    for field_name in fields:
        diffs.append(_field_diff(
            old_content.get(field_name, ""),
            new_content.get(field_name, ""),
            field_name,
        ))

    # Bullet diff
    old_bullets = old_content.get("bullet_points", [])
    new_bullets = new_content.get("bullet_points", [])
    bullet_changes = sum(
        1 for i in range(max(len(old_bullets), len(new_bullets)))
        if (old_bullets[i] if i < len(old_bullets) else "") !=
           (new_bullets[i] if i < len(new_bullets) else "")
    )

    return {
        "field_diffs": diffs,
        "bullets_changed": bullet_changes,
        "total_bullets": len(new_bullets),
        "summary": f"{sum(1 for d in diffs if d['changed'])} fields changed, {bullet_changes}/{len(new_bullets)} bullets rewritten",
    }


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


# ── MCP Server entrypoint ─────────────────────────────────────────────────────

TOOLS = {
    "score_content": score_content,
    "check_compliance": check_compliance,
    "check_brand_safety": check_brand_safety,
    "diff_content": diff_content,
}


def call_tool(tool_name: str, **kwargs) -> dict:
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}"}
    try:
        return TOOLS[tool_name](**kwargs)
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    from config import settings

    app = FastAPI(title="ScoringMCP", version="1.0")

    @app.get("/tools")
    def list_tools():
        return {"tools": list(TOOLS.keys())}

    @app.post("/call/{tool_name}")
    def call(tool_name: str, args: dict):
        return call_tool(tool_name, **args)

    uvicorn.run(app, host="0.0.0.0", port=settings.scoring_mcp_port)
