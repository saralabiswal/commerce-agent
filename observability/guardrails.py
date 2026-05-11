"""
Guardrails — detect and flag unsafe or inaccurate generated content.

Checks:
  hallucination_detection  — specs in output not present in input
  brand_safety             — prohibited claims and terms
  pii_detection            — personal data in product content
  retailer_compliance      — hard rule violations

Owner: Sarala Biswal
"""
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of running safety and accuracy guardrail checks."""
    passed: bool
    checks: dict[str, bool]
    violations: list[dict]
    severity: str  # "ok" | "warning" | "critical"

    @property
    def critical_violations(self) -> list[dict]:
        """Return guardrail violations that must block content."""
        return [v for v in self.violations if v.get("severity") == "critical"]

    @property
    def warning_violations(self) -> list[dict]:
        """Return guardrail violations that should be surfaced as warnings."""
        return [v for v in self.violations if v.get("severity") == "warning"]


class Guardrails:
    """
    Runs safety and accuracy guardrails on generated content before it ships.

    In production, content that fails critical guardrails is blocked.
    Warnings are logged and surfaced in the UI but don't block output.
    """

    def __init__(self):
        """Load guardrail settings from application configuration."""
        from config import settings
        self._settings = settings

    def run_all(
        self,
        generated_content: dict,
        product_specs: dict,
        brand_guidelines: dict | None = None,
    ) -> GuardrailResult:
        """
        Run all enabled guardrails.

        Args:
            generated_content: The generated listing content dict
            product_specs: Authoritative product specs (source of truth)
            brand_guidelines: Optional brand guidelines

        Returns:
            GuardrailResult with pass/fail status and violation details
        """
        violations = []
        checks = {}

        if self._settings.hallucination_detection:
            hall_violations = self._check_hallucination(
                generated_content, product_specs
            )
            checks["hallucination"] = len(hall_violations) == 0
            violations.extend(hall_violations)

        if self._settings.brand_safety_enabled:
            safety_violations = self._check_brand_safety(
                generated_content, brand_guidelines
            )
            checks["brand_safety"] = len(safety_violations) == 0
            violations.extend(safety_violations)

        if self._settings.pii_detection_enabled:
            pii_violations = self._check_pii(generated_content)
            checks["pii"] = len(pii_violations) == 0
            violations.extend(pii_violations)

        critical = any(v.get("severity") == "critical" for v in violations)
        any_fail = len(violations) > 0

        severity = "critical" if critical else ("warning" if any_fail else "ok")
        passed = not critical  # Warnings don't block; criticals do

        if violations:
            logger.warning(f"Guardrail violations: {violations}")

        return GuardrailResult(
            passed=passed,
            checks=checks,
            violations=violations,
            severity=severity,
        )

    def _check_hallucination(
        self, content: dict, specs: dict
    ) -> list[dict]:
        """
        Detect numeric claims in generated content not present in product specs.

        Strategy: extract all numeric values from generated content and verify
        each appears somewhere in the authoritative product specs.
        """
        violations = []
        specs_text = str(specs).lower()

        title = content.get("title", "")
        bullets = content.get("bullet_points", [])
        description = content.get("description", "")
        full_content = f"{title} {' '.join(bullets)} {description}".lower()

        # Extract specific numeric claims (hours, mm, mAh, etc.)
        numeric_patterns = [
            (r"(\d+)\s*hour", "battery/hour claim"),
            (r"(\d+)\s*mm\b", "size claim"),
            (r"(\d+)\s*mah", "battery capacity claim"),
            (r"(\d+)\s*meter", "range claim"),
            (r"(\d+)\s*db\b", "audio level claim"),
            (r"(\d+)\s*gram", "weight claim"),
        ]

        for pattern, claim_type in numeric_patterns:
            for match in re.finditer(pattern, full_content):
                number = match.group(1)
                # Check if this number appears anywhere in the specs
                if number not in specs_text:
                    violations.append({
                        "type": "hallucination",
                        "severity": "critical",
                        "claim": match.group(0),
                        "claim_type": claim_type,
                        "message": f"Generated {claim_type} '{match.group(0)}' not found in product specs",
                    })

        return violations

    def _check_brand_safety(
        self, content: dict, brand_guidelines: dict | None
    ) -> list[dict]:
        """Check for prohibited terms and unsafe claims."""
        violations = []
        title = content.get("title", "")
        bullets = content.get("bullet_points", [])
        description = content.get("description", "")
        full_text = f"{title} {' '.join(bullets)} {description}".lower()

        # Always-prohibited terms
        always_prohibited = [
            ("cures ", "medical claim"),
            ("treats ", "medical claim"),
            ("fda approved", "false regulatory claim"),
        ]
        for term, term_type in always_prohibited:
            if term in full_text:
                violations.append({
                    "type": "brand_safety",
                    "severity": "critical",
                    "term": term,
                    "message": f"Prohibited {term_type}: '{term}'",
                })

        # Brand-specific prohibited claims
        if brand_guidelines:
            for claim in brand_guidelines.get("prohibited_claims", []):
                clean = re.sub(r"\s*\(.*?\)", "", claim).strip().lower()
                if clean and len(clean) > 3 and clean in full_text:
                    violations.append({
                        "type": "brand_safety",
                        "severity": "warning",
                        "term": claim,
                        "message": f"Brand-prohibited claim: '{claim}'",
                    })

        return violations

    def _check_pii(self, content: dict) -> list[dict]:
        """Detect personally identifiable information in product content."""
        violations = []
        title = content.get("title", "")
        bullets = content.get("bullet_points", [])
        description = content.get("description", "")
        full_text = f"{title} {' '.join(bullets)} {description}"

        pii_patterns = [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
            (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone number"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        ]

        for pattern, pii_type in pii_patterns:
            if re.search(pattern, full_text):
                violations.append({
                    "type": "pii",
                    "severity": "critical",
                    "pii_type": pii_type,
                    "message": f"PII detected in product content: {pii_type}",
                })

        return violations
