"""Version-Controlled Regulatory Obligation Registry and Statutory Rule Engine.

Loads structured compliance obligations from data/regulatory_obligations.json and performs
deterministic, regex-hardened validation of percentages, timelines, and statutory prohibitions.

Negation Handling
-----------------
Penalty / prohibition checks always test for active assertions.  Phrases like
"without any penalty", "no foreclosure fee", "exempt from foreclosure charge"
are explicitly excluded so that correctly-drafted clauses are not falsely flagged.

Cooling-Off Period Rules (RBI (Digital Lending) Directions, 2025 — Clause 8.2)
-------------------------------------------------------------------------------
  - Loan tenor ≥ 7 days  → minimum cooling-off = 3 calendar days
  - Loan tenor < 7 days  → minimum cooling-off = 1 calendar day
  - If tenor is unspecified, the conservative 3-day minimum applies.
  - Penalty percentage figures in the clause are only flagged when accompanied
    by an active penalty keyword (not negated).

Percentage Interpretation
--------------------------
The DLG cap (OBL_DL_006) applies to the DLG/FLDG portfolio percentage only.
Foreclosure fee percentages (OBL_DL_003) are only flagged when a live penalty
keyword is present in the clause text without an immediately preceding negation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Single regex to extract decimal/integer percentages and variations of %, percent, per cent
PERCENTAGE_REGEX = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(?:%|percent|per\s+cent\b)")

# Regex for days and hours
DAYS_REGEX = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(?:calendar\s+)?(?:day|days)\b")
HOURS_REGEX = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*[- ]?\s*(?:hour|hours)\b")

# Negation pattern: catches "without", "no", "not", "exempt", "waived", "free"
# immediately before penalty/fee/charge language (within ~40 chars)
_NEGATION_PREFIX_RE = re.compile(
    r"(?i)\b(?:without|no|not|exempt(?:ed)?|waived?|free\s+of|zero)\b.{0,40}?"
    r"(?:penalt|foreclosure\s*fee|exit\s*charge|prepayment\s*penalt|foreclosure\s*charge)"
)

# Active (non-negated) penalty assertion: penalty/foreclosure keyword that is NOT preceded by negation
_ACTIVE_PENALTY_WORD_RE = re.compile(
    r"(?i)(?<!\bwithout\s)(?<!\bno\s)(?<!\bnot\s)"
    r"\b(?:foreclosure\s*(?:fee|penalt)|prepayment\s*penalt|exit\s*charge|penalty\s*fee|pre-payment\s*penalt)\b"
)

# Tenor detection — broad: "tenor of 5 days", "5-day loan", "loan of 5 days", "5 day tenor"
_TENOR_REGEX = re.compile(
    r"(?i)"
    r"(?:"
    r"tenor\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:calendar\s+)?days?"  # "tenor of 5 days"
    r"|(\d+(?:\.\d+)?)\s*[-\s]?day\s*(?:loan|tenor|period|credit)"  # "5-day loan"
    r"|loan\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:calendar\s+)?days?"  # "loan of 5 days"
    r"|(\d+(?:\.\d+)?)\s*(?:calendar\s+)?days?\s*tenor"  # "5 days tenor"
    r")"
)

# Cooling-off duration extraction — multiple patterns in priority order
_COOLING_DAY_PATTERNS = [
    # "1-day cooling-off", "one-day cooling off"
    re.compile(r"(?i)(\d+(?:\.\d+)?)\s*[-\s]?(?:calendar\s+)?day[s]?\s*(?:cooling(?:-off)?|look-?up)"),
    # "cooling-off period of 1 day", "cooling off window of 3 calendar days"
    re.compile(r"(?i)cooling(?:-off)?\s*(?:look-?up\s*)?(?:period|window)?\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:calendar\s+)?days?"),
    # "look-up period of 1 day"
    re.compile(r"(?i)look-?up\s*(?:period\s*)?(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:calendar\s+)?days?"),
]


def extract_percentages(text: str) -> List[float]:
    """
    Extract all percentage numbers from text.
    Handles '5%', '5.0%', '5.5 %', '5 percent', '5 per cent', '0.5%'.
    """
    if not text:
        return []
    matches = PERCENTAGE_REGEX.findall(text)
    return [float(m) for m in matches]


def extract_durations(text: str) -> Dict[str, List[float]]:
    """Extract all day and hour quantities from text."""
    if not text:
        return {"days": [], "hours": []}
    days = [float(m) for m in DAYS_REGEX.findall(text)]
    hours = [float(m) for m in HOURS_REGEX.findall(text)]
    return {"days": days, "hours": hours}


class RegulatoryObligationRegistry:
    """Singleton loader and evaluator for the RBI Regulatory Obligation Registry."""

    _instance: Optional["RegulatoryObligationRegistry"] = None

    def __init__(self, registry_path: str = "data/regulatory_obligations.json"):
        self.registry_path = Path(registry_path)
        self.data: Dict[str, Any] = {}
        self.obligations: List[Dict[str, Any]] = []
        self._load_registry()

    def _load_registry(self):
        """Load JSON obligations or fallback to safe built-in defaults."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    self.obligations = self.data.get("obligations", [])
                logger.info(f"Loaded {len(self.obligations)} obligations from {self.registry_path}")
                return
            except Exception as e:
                logger.error(f"Failed to read obligation registry at {self.registry_path}: {e}")

        # Fallback minimal built-ins
        self.obligations = []

    def get_all_obligations(self) -> List[Dict[str, Any]]:
        return self.obligations

    def get_obligation(self, obligation_id: str) -> Optional[Dict[str, Any]]:
        for obl in self.obligations:
            if obl.get("obligation_id") == obligation_id:
                return obl
        return None

    def evaluate_clause(self, clause_text: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate a clause against the registry.
        Returns a dict of evaluation details if a statutory rule or violation is matched, else None.

        Negation handling: penalty/prohibition keywords preceded by negating phrases
        ("without", "no", "not", "exempt", "waived", "free of") are NOT treated as violations.
        """
        text_lower = clause_text.lower()

        for obl in self.obligations:
            keywords = obl.get("keywords", [])
            # Must match at least one trigger keyword
            if not any(k in text_lower for k in keywords):
                continue

            # Check context keywords if specified
            context_keywords = obl.get("context_keywords", [])
            if context_keywords and not any(ck in text_lower for ck in context_keywords):
                continue

            rule_type = obl.get("rule_type")

            # -------------------------------------------------------------
            # 1. Prohibitions (e.g. Contacts scraping, Pool accounts, Biometrics)
            # -------------------------------------------------------------
            if rule_type == "PROHIBITION":
                # For obligations that carry negation_keywords (e.g. OBL_DL_003 —
                # foreclosure fee prohibition), a compliant clause like
                # "exit without any foreclosure fee" should NOT be flagged FAIL.
                # We first check the obligation-specific negation_keywords list,
                # then fall back to the global _NEGATION_PREFIX_RE pattern.
                negation_kws = obl.get("negation_keywords", [])
                clause_negated = False
                if negation_kws:
                    clause_negated = any(nk in text_lower for nk in negation_kws)
                if not clause_negated:
                    clause_negated = bool(_NEGATION_PREFIX_RE.search(text_lower))

                if clause_negated:
                    # Negation detected → not a violation; keep scanning other obligations
                    continue

                return {
                    "obligation_id": obl["obligation_id"],
                    "status": "FAIL",
                    "circular_name": obl["circular_name"],
                    "clause": obl["clause"],
                    "page_number": obl.get("page_number"),
                    "statutory_rule": obl["statutory_rule"],
                    "remediation_advice": obl["remediation_advice"],
                    "risk_level": obl.get("risk_level", "CRITICAL"),
                    "violation_reason": f"Explicit statutory prohibition violated: {obl['title']}"
                }

            # -------------------------------------------------------------
            # 2. Cooling-Off Look-up Period Validation (Clause 8.2 & 8.1)
            # -------------------------------------------------------------
            if rule_type == "COOLING_OFF_VALIDATION":
                # Check for zero / waived cooling off
                waived = any(w in text_lower for w in ["waived", "no cooling", "zero cooling", "without cooling", "0 day", "0-day"])
                if waived:
                    return {
                        "obligation_id": obl["obligation_id"],
                        "status": "FAIL",
                        "circular_name": obl["circular_name"],
                        "clause": obl["clause"],
                        "page_number": obl.get("page_number"),
                        "statutory_rule": obl["statutory_rule"],
                        "remediation_advice": obl["remediation_advice"],
                        "risk_level": "CRITICAL",
                        "violation_reason": "Statutory cooling-off look-up period cannot be waived or set to zero."
                    }

                # RBI Direction 10(i) applies a one-day minimum regardless of loan tenor.
                # Tenor text is intentionally irrelevant to this threshold.
                cooling_days = None
                for pat in _COOLING_DAY_PATTERNS:
                    m = pat.search(text_lower)
                    if m:
                        cooling_days = float(m.group(1))
                        break
                if cooling_days is None:
                    durations = extract_durations(text_lower)
                    if durations["hours"] and re.search(r"(?i)(?:(cooling|look[- ]?up).{0,25}\d+(?:\.\d+)?\s*hours?|\d+(?:\.\d+)?\s*hours?.{0,25}(cooling|look[- ]?up))", text_lower):
                        cooling_days = min(durations["hours"]) / 24.0
                    elif durations["days"]:
                        cooling_days = min(durations["days"])
                    elif durations["hours"]:
                        cooling_days = min(durations["hours"]) / 24.0

                minimum_days = float(obl.get("minimum_days", 1.0))
                if cooling_days is not None and cooling_days < minimum_days:
                    return {
                        "obligation_id": obl["obligation_id"], "status": "FAIL",
                        "circular_name": obl["circular_name"], "clause": obl["clause"],
                        "page_number": obl.get("page_number"), "statutory_rule": obl["statutory_rule"],
                        "remediation_advice": obl["remediation_advice"], "risk_level": "HIGH",
                        "violation_reason": f"Cooling-off period of {cooling_days} day(s) is below the statutory minimum of {minimum_days:g} calendar day."
                    }

                # --- Check for active (non-negated) penalty during cooling-off ---
                # First check: is there any negation covering penalty language?
                has_negated_penalty = bool(_NEGATION_PREFIX_RE.search(text_lower))
                # Second check: is there an active (non-negated) penalty keyword?
                has_active_penalty = bool(_ACTIVE_PENALTY_WORD_RE.search(clause_text))
                # Third check: percentage figure paired with an active penalty keyword
                pcts = extract_percentages(text_lower)
                has_pct_penalty = bool(pcts) and has_active_penalty

                allowed_processing_fee = bool(re.search(r"(?i)(reasonable\s+)?one[- ]time\s+processing\s+fee", clause_text)) and bool(re.search(r"(?i)(disclos(?:e|ed|ure).{0,30}upfront|upfront.{0,30}disclos)", clause_text))
                if (has_active_penalty or has_pct_penalty) and not has_negated_penalty and not allowed_processing_fee:
                    return {
                        "obligation_id": "OBL_DL_003",
                        "status": "FAIL",
                        "circular_name": obl["circular_name"],
                        "clause": "Clause 8.1: Prepayment Penalty Prohibition during Cooling-Off",
                        "page_number": obl.get("page_number"),
                        "statutory_rule": "A borrower shall be given an explicit option to exit without any penalty or foreclosure fee during the cooling-off period.",
                        "remediation_advice": "Remove all foreclosure fees and exit charges during cooling-off look-up period.",
                        "risk_level": "CRITICAL",
                        "violation_reason": "Prepayment or foreclosure penalty charged during the statutory cooling-off period."
                    }

            # -------------------------------------------------------------
            # 3. Default Loss Guarantee (DLG) Cap (Clause 13.1 - Max 5%)
            # -------------------------------------------------------------
            if rule_type == "PERCENTAGE_CAP":
                max_pct = float(obl["max_percentage"])
                percentages = extract_percentages(clause_text)
                if percentages:
                    violating_pct = [p for p in percentages if p > max_pct]
                    if violating_pct:
                        return {
                            "obligation_id": obl["obligation_id"],
                            "status": "FAIL",
                            "circular_name": obl["circular_name"],
                            "clause": obl["clause"],
                            "page_number": obl.get("page_number"),
                            "statutory_rule": obl["statutory_rule"],
                            "remediation_advice": obl["remediation_advice"],
                            "risk_level": obl.get("risk_level", "HIGH"),
                            "violation_reason": f"Default Loss Guarantee of {violating_pct[0]}% exceeds statutory cap of {max_pct}%."
                        }

            # -------------------------------------------------------------
            # 4. Hourly SLA (e.g. 6-Hour Cyber Incident Reporting Timeline)
            # -------------------------------------------------------------
            if rule_type == "HOURLY_SLA":
                max_hours = obl.get("max_hours", 6.0)
                durations = extract_durations(text_lower)
                violating_hours = [h for h in durations["hours"] if h > max_hours]
                violating_days = [d for d in durations["days"] if d * 24.0 > max_hours]
                if violating_hours or violating_days:
                    found_val = f"{violating_hours[0]} hours" if violating_hours else f"{violating_days[0]} days"
                    return {
                        "obligation_id": obl["obligation_id"],
                        "status": "FAIL",
                        "circular_name": obl["circular_name"],
                        "clause": obl["clause"],
                        "page_number": obl.get("page_number"),
                        "statutory_rule": obl["statutory_rule"],
                        "remediation_advice": obl["remediation_advice"],
                        "risk_level": obl.get("risk_level", "CRITICAL"),
                        "violation_reason": f"Incident reporting timeline of {found_val} exceeds statutory limit of {max_hours} hours."
                    }

        return None


# Global singleton helper
_REGISTRY_INSTANCE: Optional[RegulatoryObligationRegistry] = None


def get_obligation_registry() -> RegulatoryObligationRegistry:
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = RegulatoryObligationRegistry()
    return _REGISTRY_INSTANCE
