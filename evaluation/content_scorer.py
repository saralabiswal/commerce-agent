"""
Content scorer — automated quality metrics for generated content.
Runs on every output. No manual review step required.

Metrics tracked:
  title_compliance    — character count vs retailer limit
  keyword_inclusion   — top keywords present in content
  bullet_count        — exactly N bullets
  bullet_length       — each bullet within char limits
  readability         — Flesch-Kincaid score 60-80
  brand_safety        — 0 prohibited term flags
  specificity         — concrete claims vs vague language
  improvement_delta   — new score vs old score

Owner: Sarala Biswal
"""
import re
from dataclasses import dataclass

HAS_TEXTSTAT = False
try:
    import textstat
    textstat.flesch_reading_ease("The quick brown fox.")
    HAS_TEXTSTAT = True
except Exception:
    HAS_TEXTSTAT = False


def _count_syllables_simple(word: str) -> int:
    """Estimate syllable count without external dependencies."""
    word = word.lower().strip(".,!?;:")
    if not word:
        return 0
    count, vowels, prev_vowel = 0, "aeiouy", False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_simple(text: str) -> float:
    """Calculate a fallback Flesch reading ease score."""
    import re
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = text.split()
    if not words:
        return 60.0
    syllables = sum(_count_syllables_simple(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return max(0.0, min(100.0, score))


@dataclass
class MetricResult:
    """Result for one automated content quality metric."""
    name: str
    score: float          # 0-100
    passed: bool          # Met target threshold
    target: str           # Human-readable target
    actual: str           # Human-readable actual value
    details: str = ""


@dataclass
class ContentScoreReport:
    """Aggregate scoring report returned by the content scorer."""
    overall_score: float
    grade: str
    metrics: list[MetricResult]
    improvement_delta: float
    passed_quality_gate: bool
    quality_gate_threshold: float

    @property
    def passed_metrics(self) -> list[MetricResult]:
        """Return metrics that met their target thresholds."""
        return [m for m in self.metrics if m.passed]

    @property
    def failed_metrics(self) -> list[MetricResult]:
        """Return metrics that failed their target thresholds."""
        return [m for m in self.metrics if not m.passed]


class ContentScorer:
    """
    Runs all automated quality metrics against generated content.

    This is the evaluation framework the quality gate depends on,
    and also the one displayed in the Streamlit dashboard.
    """

    TARGETS = {
        "title_compliance": 100,     # percent
        "keyword_inclusion": 80,     # percent of top-5 keywords present
        "bullet_count": 100,         # exactly right count
        "bullet_length": 100,        # all bullets within limit
        "readability": 65,           # Flesch score in acceptable range
        "brand_safety": 100,         # 0 flags
        "specificity": 70,           # percent specific vs vague
        "improvement_delta": 0,      # any improvement over baseline
    }

    def score(
        self,
        content: dict,
        requirements: dict,
        previous_score: float = 0.0,
        brand_guidelines: dict | None = None,
    ) -> ContentScoreReport:
        """
        Score content across all quality dimensions.

        Args:
            content: Dict with title, bullet_points, description, backend_keywords
            requirements: Retailer-specific requirements dict
            previous_score: The score before optimization (for delta calculation)
            brand_guidelines: Optional brand guidelines for safety check

        Returns:
            ContentScoreReport with per-metric breakdown
        """
        from config import settings

        metrics = []

        # 1. Title compliance
        metrics.append(self._score_title_compliance(content, requirements))

        # 2. Keyword inclusion
        metrics.append(self._score_keyword_inclusion(content, requirements))

        # 3. Bullet count
        metrics.append(self._score_bullet_count(content, requirements))

        # 4. Bullet length
        metrics.append(self._score_bullet_length(content, requirements))

        # 5. Readability
        metrics.append(self._score_readability(content))

        # 6. Brand safety
        metrics.append(self._score_brand_safety(content, brand_guidelines))

        # 7. Specificity
        metrics.append(self._score_specificity(content))

        # 8. Improvement delta
        overall = self._compute_weighted_score(metrics)
        metrics.append(self._score_improvement_delta(overall, previous_score))

        # Recompute overall with delta included (minor weight)
        overall = self._compute_weighted_score(metrics)

        return ContentScoreReport(
            overall_score=round(overall, 1),
            grade=self._to_grade(overall),
            metrics=metrics,
            improvement_delta=round(overall - previous_score, 1),
            passed_quality_gate=overall >= settings.quality_gate_threshold,
            quality_gate_threshold=settings.quality_gate_threshold,
        )

    def _score_title_compliance(self, content: dict, requirements: dict) -> MetricResult:
        """Score title length and prohibited-word compliance."""
        title = content.get("title", "")
        title_rules = requirements.get("title", {})
        max_chars = title_rules.get("max_chars", 200)
        min_chars = title_rules.get("min_chars", 20)
        prohibited = title_rules.get("prohibited_words", [])

        violations = []
        if len(title) > max_chars:
            violations.append(f"over limit by {len(title) - max_chars} chars")
        if len(title) < min_chars:
            violations.append(f"under minimum by {min_chars - len(title)} chars")
        for word in prohibited:
            if word.lower() in title.lower():
                violations.append(f"prohibited: '{word}'")

        passed = len(violations) == 0
        score = 100.0 if passed else max(0, 100 - len(violations) * 25)
        utilization = round(len(title) / max_chars * 100, 1)

        return MetricResult(
            name="title_compliance",
            score=score,
            passed=passed,
            target="100%",
            actual=f"{len(title)}/{max_chars} chars ({utilization}% utilization)",
            details="; ".join(violations) if violations else "All checks passed",
        )

    def _score_keyword_inclusion(self, content: dict, requirements: dict) -> MetricResult:
        """Score whether high-volume keywords appear in generated content."""
        benchmarks = requirements.get("search_volume_benchmarks", {})
        if not benchmarks:
            return MetricResult(
                name="keyword_inclusion",
                score=70, passed=True,
                target="≥ 80%", actual="No benchmark data",
            )

        top_keywords = sorted(benchmarks.items(), key=lambda x: x[1], reverse=True)[:5]
        full_text = " ".join([
            content.get("title", ""),
            " ".join(content.get("bullet_points", [])),
            content.get("description", ""),
        ]).lower()

        matched = [kw for kw, _ in top_keywords if kw.lower() in full_text]
        pct = round(len(matched) / len(top_keywords) * 100, 1)
        missing = [kw for kw, _ in top_keywords if kw.lower() not in full_text]

        return MetricResult(
            name="keyword_inclusion",
            score=pct,
            passed=pct >= 80,
            target="≥ 80%",
            actual=f"{pct}% ({len(matched)}/{len(top_keywords)} top keywords)",
            details=f"Missing: {', '.join(missing)}" if missing else "All top keywords present",
        )

    def _score_bullet_count(self, content: dict, requirements: dict) -> MetricResult:
        """Score whether the listing has the required number of bullets."""
        bullets = content.get("bullet_points", [])
        expected = requirements.get("bullet_points", {}).get("count", 5)
        passed = len(bullets) == expected
        score = 100.0 if passed else max(0, 100 - abs(len(bullets) - expected) * 30)
        return MetricResult(
            name="bullet_count",
            score=score, passed=passed,
            target=f"Exactly {expected}",
            actual=str(len(bullets)),
            details="" if passed else f"Expected {expected}, got {len(bullets)}",
        )

    def _score_bullet_length(self, content: dict, requirements: dict) -> MetricResult:
        """Score whether each bullet fits retailer character limits."""
        bullets = content.get("bullet_points", [])
        max_chars = requirements.get("bullet_points", {}).get("max_chars_each", 255)
        violations = [
            f"Bullet {i+1}: {len(b)}/{max_chars} chars"
            for i, b in enumerate(bullets)
            if len(b) > max_chars
        ]
        passed = len(violations) == 0
        score = 100.0 if passed else max(0, 100 - len(violations) * 20)
        return MetricResult(
            name="bullet_length",
            score=score, passed=passed,
            target=f"Each ≤ {max_chars} chars",
            actual=f"Max: {max(len(b) for b in bullets) if bullets else 0} chars",
            details="; ".join(violations) if violations else "All within limit",
        )

    def _score_readability(self, content: dict) -> MetricResult:
        """Score generated content readability for consumer-facing copy."""
        title = content.get("title", "")
        bullets = content.get("bullet_points", [])
        description = content.get("description", "")
        full_text = f"{title}. {' '.join(bullets)}. {description}"

        if not HAS_TEXTSTAT or not full_text.strip():
            return MetricResult(
                name="readability", score=65, passed=True,
                target="Flesch 60-80", actual="Not measured (textstat unavailable)",
            )
        try:
            fk = textstat.flesch_reading_ease(full_text)
        except Exception:
            fk = _flesch_simple(full_text)
        in_range = 60 <= fk <= 80
        score = 100 if in_range else (80 if 50 <= fk < 60 or 80 < fk <= 90 else 50)

        return MetricResult(
            name="readability",
            score=float(score), passed=in_range,
            target="Flesch 60-80",
            actual=f"Flesch {fk:.0f}",
            details=(
                "In target range"
                if in_range
                else f"{'Too complex' if fk < 60 else 'Too simple'} for consumer products"
            ),
        )

    def _score_brand_safety(
        self, content: dict, brand_guidelines: dict | None
    ) -> MetricResult:
        """Score brand safety using configured prohibited terms and claims."""
        from mcp_servers.scoring_mcp_server import check_brand_safety
        result = check_brand_safety(content, brand_guidelines)
        passed = result["passed"]
        score = 100.0 if passed else max(0, 100 - result["flag_count"] * 20)
        return MetricResult(
            name="brand_safety",
            score=score, passed=passed,
            target="0 flags",
            actual=f"{result['flag_count']} flags",
            details=str(result["flags"]) if result["flags"] else "No issues found",
        )

    def _score_specificity(self, content: dict) -> MetricResult:
        """Score the ratio of specific claims to vague marketing language."""
        title = content.get("title", "")
        bullets = content.get("bullet_points", [])
        description = content.get("description", "")
        full_text = f"{title} {' '.join(bullets)} {description}".lower()

        vague = ["high quality", "great", "amazing", "awesome", "perfect",
                 "good", "nice", "excellent", "superior", "premium quality"]
        specific_patterns = [
            r"\d+\s*(hours?|hrs?|minutes?|mins?)",
            r"\d+\s*(mm|cm|inch|inches|ft|gb|mb|mhz|ghz|ohm|db)",
            r"\d+\s*(count|pack|pieces?|ct\b)",
            r"ip[x\d]\d", r"bluetooth\s*\d+",
        ]

        vague_count = sum(1 for phrase in vague if phrase in full_text)
        specific_count = sum(1 for p in specific_patterns if re.search(p, full_text))

        score = min(100, max(0, 60 + specific_count * 8 - vague_count * 10))
        passed = score >= 70

        return MetricResult(
            name="specificity",
            score=float(score), passed=passed,
            target="≥ 70%",
            actual=f"{specific_count} specific claims, {vague_count} vague phrases",
            details=f"Specificity score: {score:.0f}",
        )

    def _score_improvement_delta(
        self, new_score: float, previous_score: float
    ) -> MetricResult:
        """Score whether generated content improves over the baseline."""
        delta = new_score - previous_score
        passed = delta > 0
        return MetricResult(
            name="improvement_delta",
            score=min(100, max(0, 50 + delta)),
            passed=passed,
            target="> 0 (any improvement)",
            actual=f"{delta:+.1f} points ({previous_score:.1f} → {new_score:.1f})",
            details="Improved" if passed else "No improvement over baseline",
        )

    def _compute_weighted_score(self, metrics: list[MetricResult]) -> float:
        """Combine metric scores using the scorer's fixed weights."""
        weights = {
            "title_compliance": 0.20,
            "keyword_inclusion": 0.20,
            "bullet_count": 0.15,
            "bullet_length": 0.10,
            "readability": 0.10,
            "brand_safety": 0.10,
            "specificity": 0.10,
            "improvement_delta": 0.05,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for m in metrics:
            w = weights.get(m.name, 0.05)
            weighted_sum += m.score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _to_grade(self, score: float) -> str:
        """Convert a numeric score into a letter grade."""
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"
