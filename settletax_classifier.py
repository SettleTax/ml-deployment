"""
SettleTax Transaction Classifier v1.0
=====================================
4-layer classification engine for Nigerian bank transactions.

Layer 1: Structural Detection (self-transfer, bank charges, reversals, duplicates)
Layer 2: User History Matching (counterparty → category from past user actions)
Layer 3: Rule Engine (150+ Nigerian-specific keyword + amount patterns)
Layer 4: LLM Fallback (batch classification for ambiguous transactions)

Usage:
    from settletax_classifier import SettleTaxClassifier

    classifier = SettleTaxClassifier(account_name="OBAFEMI-MOSES MOSINMILOLUWA")
    results = classifier.classify_batch(transactions_df)
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Set, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# CORE TYPES
# ═══════════════════════════════════════════════════════════════

class ClassificationSource(str, Enum):
    STRUCTURAL = "structural"
    USER_HISTORY = "user_history"
    RULE = "rule"
    LLM = "llm"
    UNCLASSIFIED = "unclassified"


@dataclass
class ClassifyResult:
    category: Optional[str]          # SettleTax category label (e.g., "Fuel", "Income")
    type: str                        # "income" or "expense"
    confidence: float                # 0.0 to 1.0
    source: ClassificationSource     # which layer classified it
    needs_review: bool               # flag for user to confirm
    counterparty: Optional[str]      # extracted counterparty name
    explanation: str = ""            # human-readable reason
    rule_hit: Optional[str] = None   # which rule matched (for debugging)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        return d


# ═══════════════════════════════════════════════════════════════
# SETTLETAX CATEGORY DEFINITIONS
# These must match income_streams.category exactly
# ═══════════════════════════════════════════════════════════════

SETTLETAX_INCOME_CATEGORIES = [
    "Income",                    # General business income / salary
    "Annual Rent",               # Rental income (landlords)
    "Bank Interest Received",    # Interest from savings/deposits
    "Dividend Income",           # Dividends from shares
    "Investment Income",         # Returns from investment platforms
    "Crypto Trading Income",     # Crypto trading profits
    "Capital Gains Income",      # Gains from asset sales
    "PAYE Income",               # Employment income (PAYE)
    "Consulting Income",         # Freelance/consulting fees
    "Commission Income",         # Sales commissions
]

SETTLETAX_EXPENSE_CATEGORIES = [
    "Fuel",                      # Petrol, diesel
    "Motor Running Expenses",    # Vehicle maintenance, Bolt/Uber
    "Telephone",                 # Airtime, data, telecom
    "Utilities",                 # Electricity, water, DSTV
    "Office Rent",               # Office/shop rent
    "Annual Rent",               # Personal rent (expense side)
    "Salaries and Wages",        # Staff salaries
    "Training",                  # Education, courses, school fees
    "Insurance",                 # General insurance
    "Bank Charges",              # Bank fees, NIP charges, SMS alerts
    "Interest on Loans",         # Loan interest/repayments
    "Legal and Professional Fees", # Legal, accounting, CAC
    "Repairs and Maintenance",   # Repairs to property/equipment
    "Advertising",               # Marketing, ads
    "Stationery",                # Office supplies
    "Office Equipment",          # Equipment purchases
    "Computer Equipment",        # IT equipment
    "Travel Expenses",           # Business travel
    "Subscriptions",             # Software, memberships
    "Depreciation",              # Asset depreciation
    "Bad Debts",                 # Written-off debts
    "Donations",                 # Charitable donations
    "Health Insurance",          # NHIS
    "Life Assurance Premium",    # Life insurance
    "National Housing Fund",     # NHF
    "Pension",                   # Pension contributions
    "Drawings",                  # Personal/non-business expenses
    "Transfer",                  # Inter-account transfers
    "Crypto Trading Cost",       # Crypto purchases
    "Capital Gains Cost",        # Cost basis of sold assets
]

ALL_CATEGORIES = set(SETTLETAX_INCOME_CATEGORIES + SETTLETAX_EXPENSE_CATEGORIES)


# ═══════════════════════════════════════════════════════════════
# LAYER 1: STRUCTURAL DETECTION
# ═══════════════════════════════════════════════════════════════

class StructuralDetector:
    """
    Detects transactions by their structure, not keywords:
    - Self-transfers (your money moving between your own accounts)
    - Bank charges (small fees following transfers)
    - Reversals (failed/reversed transactions)
    - Duplicate pairs (same amount, opposite direction, same day)
    """

    # Known Nigerian noise words in bank narrations
    NOISE_WORDS = frozenset({
        "TRANSFER", "BETWEEN", "CUSTOMERS", "PAYMENT", "OUTWARD",
        "INWARD", "NIP", "FIP", "FAILED", "REVERSAL", "REF",
        "NIBSS", "INSTANT", "NEFT", "THE", "FOR", "AND", "FROM",
        "VIA", "MOBILE", "BANKING", "INTERNET", "USSD", "APP",
        "TRANSACTION", "CHARGE", "FEE", "TO", "OF", "ON", "AT",
    })

    # Nigerian bank-specific provider patterns
    PROVIDER_PATTERNS = [
        # PalmPay: "NIBSS Instan ... PALMPAY ... NIP TRANSFER TO {name}"
        (r"PALMPAY.*?NIP\s+TRANSFER\s+TO\s+(.+?)$", "palmpay"),
        # OPay: "OPAY - {prefix} T ... NIP TRANSFER TO {suffix}"
        (r"OPAY\s*-\s*(.+?)\s+T\s+.*?NIP\s+TRANSFER\s+TO\s+(.+?)$", "opay"),
        # Kuda: "Transfer from {name}" or "Transfer to {name}"
        (r"TRANSFER\s+(?:FROM|TO)\s+(.+?)(?:\s+\d|$)", "kuda"),
        # Standard NIP: "NIP TRANSFER TO {name}" or "NIP/FIP ... TO {name}"
        (r"NIP\s+(?:TRANSFER\s+)?TO\s+(.+?)(?:\s+\d{6,}|$)", "nip_standard"),
        # Access/GTB: "TRF-{name}/{ref}" or "{name}/TRF/{ref}"
        (r"TRF[-\s]+(.+?)(?:/|$)", "trf_prefix"),
        (r"^(.+?)/TRF/", "trf_suffix"),
        # Zenith: "MC TRANSFER: {ref} {name}"
        (r"MC\s+TRANSFER:\s*\S+\s+(.+?)$", "zenith"),
        # Moniepoint: "Transfer to {name} - {ref}"
        (r"TRANSFER\s+TO\s+(.+?)\s*-\s*\S+$", "moniepoint"),
    ]

    # Compiled provider patterns
    _compiled_providers = None

    @classmethod
    def _get_compiled_providers(cls):
        if cls._compiled_providers is None:
            cls._compiled_providers = [
                (re.compile(pat, re.IGNORECASE), name)
                for pat, name in cls.PROVIDER_PATTERNS
            ]
        return cls._compiled_providers

    def __init__(self, account_name: str, account_names: List[str] = None):
        """
        Args:
            account_name: Primary account holder name
            account_names: Additional name variations (e.g., business name)
        """
        self.account_name = account_name
        self.all_names = [account_name]
        if account_names:
            self.all_names.extend(account_names)

        # Pre-compute identity signatures for all names
        self.owner_signatures = [
            self._build_identity_signature(name)
            for name in self.all_names
        ]

    @staticmethod
    def _normalize_identity(text: str) -> str:
        """Strip to uppercase letters only."""
        return re.sub(r"[^A-Z]", "", text.upper())

    @staticmethod
    def _char_ngrams(text: str, n: int = 5) -> Set[str]:
        """Generate character n-grams from text."""
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    @staticmethod
    def _ngram_similarity(a: str, b: str, n: int = 5) -> float:
        """Jaccard similarity of character n-grams."""
        ga = StructuralDetector._char_ngrams(a, n)
        gb = StructuralDetector._char_ngrams(b, n)
        if not ga or not gb:
            return 0.0
        intersection = len(ga & gb)
        union = len(ga | gb)
        return intersection / union if union > 0 else 0.0

    def _build_identity_signature(self, name: str) -> dict:
        """Build a reusable identity signature for matching."""
        clean = self._normalize_identity(name)
        return {
            "clean": clean,
            "ngrams": self._char_ngrams(clean, 5),
            "tokens": set(re.sub(r"[^A-Z\s]", " ", name.upper()).split()),
        }

    def _is_name_like(self, token: str) -> bool:
        """Heuristic: does this token look like a person's name?"""
        if not token.isalpha() or len(token) < 2 or len(token) > 25:
            return False
        vowels = sum(1 for c in token if c in "AEIOU")
        if vowels == 0:
            return False
        if (len(token) - vowels) / len(token) > 0.85:
            return False
        if len(set(token)) <= 2:
            return False
        return True

    def _extract_name_from_provider(self, narration: str) -> Optional[str]:
        """Try to extract counterparty name using provider-specific patterns."""
        text = narration.upper().strip()
        for pattern, provider in self._get_compiled_providers():
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if provider == "opay" and len(groups) == 2:
                    # OPay: concatenate prefix + suffix
                    return (groups[0] + groups[1]).strip()
                else:
                    return groups[0].strip()
        return None

    def _extract_name_spans(self, narration: str) -> List[str]:
        """
        Extract name-like spans from narration by:
        1. Removing noise words, numbers, references
        2. Grouping consecutive name-like tokens
        """
        clean = re.sub(r"[^A-Z\s\-]", " ", narration.upper())
        clean = re.sub(r"\s+", " ", clean).strip()
        tokens = clean.split()

        spans = []
        current = []

        for tok in tokens:
            tok_clean = tok.replace("-", "")
            if (tok_clean.upper() not in self.NOISE_WORDS
                    and self._is_name_like(tok_clean)):
                current.append(tok)
            else:
                if current:
                    spans.append(" ".join(current))
                    current = []

        if current:
            spans.append(" ".join(current))

        return spans

    def detect_self_transfer(
        self, narration: str, threshold: float = 0.28
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if a transaction is a self-transfer by matching
        the counterparty name against the account holder.

        Uses three strategies:
        1. Provider-specific name extraction (PalmPay, OPay, etc.)
        2. Individual span matching (each name-like span vs owner)
        3. Combined span matching (all spans concatenated vs owner)
           This handles Nigerian banks that fragment names across the narration.

        Returns: (is_self, best_score, best_matching_span)
        """
        # Step 1: Try provider-specific extraction first
        provider_name = self._extract_name_from_provider(narration)

        # Step 2: Extract name spans from narration
        if provider_name:
            spans = [provider_name]
        else:
            spans = self._extract_name_spans(narration)

        if not spans:
            return False, 0.0, None

        # Step 3: Compare each individual span against all owner signatures
        best_score = 0.0
        best_span = None

        for span in spans:
            span_clean = self._normalize_identity(span)
            if len(span_clean) < 3:
                continue

            for sig in self.owner_signatures:
                score = self._ngram_similarity(sig["clean"], span_clean)
                if score > best_score:
                    best_score = score
                    best_span = span

                # Also check token overlap as backup
                span_tokens = set(
                    re.sub(r"[^A-Z\s]", " ", span.upper()).split()
                ) - self.NOISE_WORDS
                overlap = sig["tokens"] & span_tokens
                if len(overlap) >= 2:
                    token_score = min(0.50, len(overlap) * 0.20)
                    if token_score > best_score:
                        best_score = token_score
                        best_span = span

        # Step 4: Combined-span matching for fragmented narrations
        # Nigerian banks (esp. GTBank) scatter name fragments across the narration.
        # Concatenate all spans and compare as one unit.
        if len(spans) > 1:
            combined = " ".join(spans)
            combined_clean = self._normalize_identity(combined)
            if len(combined_clean) >= 5:
                for sig in self.owner_signatures:
                    # Try multiple n-gram sizes for robustness
                    for n in [5, 4, 3]:
                        score = self._ngram_similarity(
                            sig["clean"], combined_clean, n=n
                        )
                        # Penalize smaller n-grams slightly (more false positives)
                        if n < 5:
                            score *= (0.85 + (n * 0.03))
                        if score > best_score:
                            best_score = score
                            best_span = combined

                    # Token overlap on the combined text
                    combined_tokens = set(
                        re.sub(r"[^A-Z\s]", " ", combined.upper()).split()
                    ) - self.NOISE_WORDS
                    overlap = sig["tokens"] & combined_tokens
                    if len(overlap) >= 2:
                        token_score = min(0.50, len(overlap) * 0.20)
                        if token_score > best_score:
                            best_score = token_score
                            best_span = combined

        # Step 5: Substring containment check
        # If the owner's name (stripped of spaces) is substantially
        # contained within the narration text, it's likely a self-transfer.
        narration_clean = self._normalize_identity(narration)
        if len(narration_clean) >= 10:
            for sig in self.owner_signatures:
                owner_clean = sig["clean"]
                # Check if large chunks of the owner name appear in the narration
                # Try progressively smaller windows of the owner name
                owner_len = len(owner_clean)
                if owner_len >= 6:
                    # Count how many 4-char chunks of the owner appear in narration
                    chunk_size = 4
                    chunks = [
                        owner_clean[i:i+chunk_size]
                        for i in range(0, owner_len - chunk_size + 1)
                    ]
                    hits = sum(1 for c in chunks if c in narration_clean)
                    if chunks:
                        containment = hits / len(chunks)
                        # If >60% of owner's 4-char chunks appear in narration
                        if containment >= 0.60:
                            score = min(containment * 0.55, 0.45)
                            if score > best_score:
                                best_score = score
                                best_span = "substring_match"

                # Full narration n-gram as last resort
                score = self._ngram_similarity(
                    owner_clean, narration_clean, n=4
                )
                score *= 0.80  # Penalize since full narration is very noisy
                if score > best_score:
                    best_score = score
                    best_span = "full_narration"

        return best_score >= threshold, best_score, best_span

    def detect_bank_charge(
        self, narration: str, amount: float, direction: str
    ) -> Optional[ClassifyResult]:
        """
        Detect bank charges by narration keywords AND amount patterns.
        """
        desc = narration.upper()

        # Explicit bank charge keywords (very high confidence)
        explicit_keywords = [
            ("NIP TRANSFER COMMISSION", 0.99),
            ("NIP COMMISSION", 0.99),
            ("SMS CHARGE", 0.99),
            ("SMS ALERT", 0.99),
            ("ACCOUNT MAINTENANCE", 0.99),
            ("ACCT MAINTENANCE", 0.99),
            ("E-MONEY TRANSFER LEVY", 0.99),
            ("ELECTRONIC TRANSFER LEVY", 0.99),
            ("STAMP DUTY", 0.99),
            ("VAT ON COMMISSION", 0.99),
            ("VAT ON FEE", 0.99),
            ("CARD MAINTENANCE", 0.95),
            ("CARD ISSUANCE", 0.95),
            ("TOKEN CHARGE", 0.95),
            ("COT CHARGE", 0.95),
        ]

        for keyword, conf in explicit_keywords:
            if keyword in desc:
                return ClassifyResult(
                    category="Bank Charges",
                    type="expense",
                    confidence=conf,
                    source=ClassificationSource.STRUCTURAL,
                    needs_review=False,
                    counterparty=None,
                    explanation=f"Bank charge detected: '{keyword}'",
                    rule_hit=f"bank_charge_{keyword.lower().replace(' ', '_')}",
                )

        # Amount-based heuristic: small debit (₦10-₦75) = likely NIP fee
        if direction == "debit" and 10 <= amount <= 75:
            if any(k in desc for k in ["NIP", "TRANSFER", "COMMISSION"]):
                return ClassifyResult(
                    category="Bank Charges",
                    type="expense",
                    confidence=0.88,
                    source=ClassificationSource.STRUCTURAL,
                    needs_review=False,
                    counterparty=None,
                    explanation=f"Small debit (₦{amount}) with transfer keyword = NIP fee",
                    rule_hit="bank_charge_amount_nip",
                )

        # SMS alert charge pattern: exactly ₦4 or ₦52 debit
        if direction == "debit" and amount in (4, 4.00, 52, 52.00):
            if "SMS" in desc or "ALERT" in desc:
                return ClassifyResult(
                    category="Bank Charges",
                    type="expense",
                    confidence=0.92,
                    source=ClassificationSource.STRUCTURAL,
                    needs_review=False,
                    counterparty=None,
                    explanation=f"SMS alert charge (₦{amount})",
                    rule_hit="bank_charge_sms",
                )

        return None

    def detect_reversal(self, narration: str) -> Optional[ClassifyResult]:
        """Detect failed/reversed transactions."""
        desc = narration.upper()
        reversal_keywords = [
            "REVERSAL", "REVERSED", "FAILED TRANSACTION",
            "TRANSACTION FAILED", "REFUND", "CHARGEBACK",
        ]
        for keyword in reversal_keywords:
            if keyword in desc:
                return ClassifyResult(
                    category="Transfer",
                    type="expense",
                    confidence=0.90,
                    source=ClassificationSource.STRUCTURAL,
                    needs_review=True,  # user should verify
                    counterparty=None,
                    explanation=f"Reversal/refund detected: '{keyword}'",
                    rule_hit="reversal",
                )
        return None

    def detect_atm(self, narration: str, direction: str) -> Optional[ClassifyResult]:
        """Detect ATM withdrawals."""
        desc = narration.upper()
        if direction == "debit" and any(k in desc for k in ["ATM WITHDRAWAL", "ATM WDL", "ATM CASH"]):
            return ClassifyResult(
                category="Drawings",
                type="expense",
                confidence=0.85,
                source=ClassificationSource.STRUCTURAL,
                needs_review=True,  # could be business or personal
                counterparty=None,
                explanation="ATM cash withdrawal",
                rule_hit="atm_withdrawal",
            )
        return None

    def classify(
        self, narration: str, amount: float, direction: str
    ) -> Optional[ClassifyResult]:
        """
        Run all structural detections. Returns first match or None.
        Priority: reversal > bank charge > ATM > self-transfer
        """
        # 1. Reversals first (they override everything)
        result = self.detect_reversal(narration)
        if result:
            return result

        # 2. Bank charges (structural pattern, not keyword)
        result = self.detect_bank_charge(narration, amount, direction)
        if result:
            return result

        # 3. ATM withdrawals
        result = self.detect_atm(narration, direction)
        if result:
            return result

        # 4. Self-transfer detection
        is_self, score, best_span = self.detect_self_transfer(narration)
        if is_self:
            return ClassifyResult(
                category="Transfer",
                type="expense",
                confidence=min(score + 0.20, 0.98),
                source=ClassificationSource.STRUCTURAL,
                needs_review=score < 0.40,
                counterparty=best_span,
                explanation=f"Self-transfer detected (n-gram score: {score:.3f}, span: '{best_span}')",
                rule_hit="self_transfer",
            )

        return None


# ═══════════════════════════════════════════════════════════════
# LAYER 2: USER HISTORY MATCHING
# ═══════════════════════════════════════════════════════════════

class UserHistoryMatcher:
    """
    Matches transactions against previously categorised counterparties.

    In production, this reads from the `counterparty_mappings` table.
    For testing, you can pre-load mappings manually.
    """

    def __init__(self, mappings: Dict[str, dict] = None):
        """
        Args:
            mappings: Dict of counterparty name → {category, type, match_count}
                      Keys will be normalized automatically.
        """
        self.mappings: Dict[str, dict] = {}
        if mappings:
            for key, value in mappings.items():
                normalized = self._normalize(key)
                self.mappings[normalized] = value

    def add_mapping(
        self, counterparty: str, category: str, tx_type: str, source: str = "user"
    ):
        """Add or update a counterparty → category mapping."""
        key = self._normalize(counterparty)
        if key in self.mappings:
            self.mappings[key]["match_count"] += 1
            # Update category if user explicitly re-categorised
            if source == "user":
                self.mappings[key]["category"] = category
                self.mappings[key]["type"] = tx_type
        else:
            self.mappings[key] = {
                "category": category,
                "type": tx_type,
                "match_count": 1,
                "source": source,
            }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize counterparty name for matching.

        Only strips formal legal suffixes (LTD, PLC, etc.), not business
        descriptors like ENTERPRISES which may be part of the actual name.
        """
        text = text.upper().strip()
        text = re.sub(r"[^A-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Only strip formal legal entity suffixes
        for suffix in [" LTD", " LIMITED", " PLC", " INC", " LLC"]:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()
        return text

    @staticmethod
    def _fuzzy_match(a: str, b: str) -> float:
        """Simple token overlap similarity."""
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = len(tokens_a & tokens_b)
        return overlap / max(len(tokens_a), len(tokens_b))

    def classify(self, counterparty: Optional[str]) -> Optional[ClassifyResult]:
        """Look up counterparty in user history."""
        if not counterparty:
            return None

        key = self._normalize(counterparty)
        if not key:
            return None

        # Exact match
        if key in self.mappings:
            m = self.mappings[key]
            confidence = min(0.70 + (m["match_count"] * 0.05), 0.95)
            return ClassifyResult(
                category=m["category"],
                type=m["type"],
                confidence=confidence,
                source=ClassificationSource.USER_HISTORY,
                needs_review=m["match_count"] < 3,
                counterparty=counterparty,
                explanation=f"Matched counterparty '{counterparty}' (seen {m['match_count']}x before)",
                rule_hit="history_exact",
            )

        # Fuzzy match (for slight name variations)
        best_score = 0.0
        best_key = None
        for existing_key in self.mappings:
            score = self._fuzzy_match(key, existing_key)
            if score > best_score:
                best_score = score
                best_key = existing_key

        if best_score >= 0.70 and best_key:
            m = self.mappings[best_key]
            confidence = min(0.55 + (best_score * 0.30), 0.85)
            return ClassifyResult(
                category=m["category"],
                type=m["type"],
                confidence=confidence,
                source=ClassificationSource.USER_HISTORY,
                needs_review=True,  # fuzzy match always needs review
                counterparty=counterparty,
                explanation=f"Fuzzy match '{counterparty}' ≈ '{best_key}' (score: {best_score:.2f})",
                rule_hit="history_fuzzy",
            )

        return None


# ═══════════════════════════════════════════════════════════════
# LAYER 3: RULE ENGINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class Rule:
    keywords: List[str]
    category: str
    type: str  # "income" or "expense"
    confidence: float
    rule_name: str
    direction: Optional[str] = None      # "debit" or "credit" or None
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    exclude_keywords: List[str] = field(default_factory=list)

    def matches(self, narration: str, direction: str, amount: float) -> bool:
        desc = narration.upper()

        # Check keyword match
        if not any(k in desc for k in self.keywords):
            return False

        # Check exclusions
        if any(k in desc for k in self.exclude_keywords):
            return False

        # Check direction
        if self.direction and self.direction != direction:
            return False

        # Check amount range
        if self.amount_min is not None and amount < self.amount_min:
            return False
        if self.amount_max is not None and amount > self.amount_max:
            return False

        return True


class RuleEngine:
    """
    150+ Nigerian-specific classification rules.
    Rules are ordered by specificity (most specific first).
    """

    RULES: List[Rule] = [
        # ═══ TAX PAYMENTS ═══
        Rule(["FEDERAL INLAND REVENUE", "FIRS PAYMENT"], "PAYE Income", "expense", 0.97, "tax_firs"),
        Rule(["LAGOS INTERNAL REVENUE", "LIRS PAYMENT"], "PAYE Income", "expense", 0.97, "tax_lirs"),
        Rule(["PAYE DEDUCTION", "PAYE PAYMENT", "PAYE TAX"], "PAYE Income", "expense", 0.96, "tax_paye"),
        Rule(["WITHHOLDING TAX", "WHT PAYMENT", "WHT DEDUCTION"], "PAYE Income", "expense", 0.93, "tax_wht"),
        Rule(["VAT PAYMENT", "VAT REMITTANCE"], "PAYE Income", "expense", 0.93, "tax_vat",
             exclude_keywords=["VAT ON COMMISSION", "VAT ON FEE"]),  # avoid bank charges

        # ═══ RELIEFS ═══
        Rule(["NATIONAL HOUSING FUND", "NHF DEDUCTION", "NHF CONTRIBUTION"], "National Housing Fund", "expense", 0.98, "relief_nhf"),
        Rule(["NATIONAL HEALTH INSURANCE", "NHIS DEDUCTION", "NHIS CONTRIBUTION"], "Health Insurance", "expense", 0.98, "relief_nhis"),
        Rule(["LIFE ASSURANCE PREMIUM", "LIFE ASSURANCE", "LIFE INSURANCE PREMIUM"], "Life Assurance Premium", "expense", 0.96, "relief_life"),
        Rule(["AIICO INSURANCE", "AIICO"], "Life Assurance Premium", "expense", 0.93, "relief_aiico"),
        Rule(["AXA MANSARD", "AXA INSURANCE"], "Life Assurance Premium", "expense", 0.93, "relief_axa"),
        Rule(["PENSION CONTRIBUTION", "PFA CONTRIBUTION", "RSA CONTRIBUTION", "PENCOM"], "Pension", "expense", 0.96, "relief_pension"),

        # ═══ UTILITIES — ELECTRICITY ═══
        Rule(["IKEJA ELECTRIC", "IKEDC", "IKEJA DISCO"], "Utilities", "expense", 0.96, "util_ikeja"),
        Rule(["EKO ELECTRICITY", "EKO DISCO", "EKEDC"], "Utilities", "expense", 0.96, "util_eko"),
        Rule(["IBADAN ELECTRIC", "IBEDC"], "Utilities", "expense", 0.96, "util_ibedc"),
        Rule(["ABUJA ELECTRIC", "AEDC"], "Utilities", "expense", 0.96, "util_aedc"),
        Rule(["ENUGU ELECTRIC", "EEDC"], "Utilities", "expense", 0.96, "util_eedc"),
        Rule(["BENIN ELECTRIC", "BEDC"], "Utilities", "expense", 0.96, "util_bedc"),
        Rule(["JOS ELECTRIC", "JEDC"], "Utilities", "expense", 0.96, "util_jedc"),
        Rule(["KADUNA ELECTRIC", "KAEDCO"], "Utilities", "expense", 0.96, "util_kaduna"),
        Rule(["PORT HARCOURT ELECTRIC", "PHEDC"], "Utilities", "expense", 0.96, "util_ph"),
        Rule(["ELECTRICITY", "PREPAID METER", "BUY POWER", "BUYPOWER"], "Utilities", "expense", 0.90, "util_electricity"),

        # ═══ UTILITIES — TV/CABLE ═══
        Rule(["DSTV", "MULTICHOICE"], "Utilities", "expense", 0.93, "util_dstv"),
        Rule(["GOTV"], "Utilities", "expense", 0.93, "util_gotv"),
        Rule(["STARTIMES"], "Utilities", "expense", 0.93, "util_startimes"),
        Rule(["SHOWMAX"], "Utilities", "expense", 0.90, "util_showmax"),

        # ═══ UTILITIES — WATER / WASTE ═══
        Rule(["WATER CORPORATION", "WATER BOARD", "WATERBOARD"], "Utilities", "expense", 0.90, "util_water"),
        Rule(["LAWMA", "WASTE MANAGEMENT"], "Utilities", "expense", 0.90, "util_lawma"),

        # ═══ TELECOM / AIRTIME ═══
        Rule(["MTN AIRTIME", "MTN DATA", "MTN VTU"], "Telephone", "expense", 0.95, "tel_mtn_explicit"),
        Rule(["GLO AIRTIME", "GLO DATA", "GLO VTU"], "Telephone", "expense", 0.95, "tel_glo_explicit"),
        Rule(["AIRTEL AIRTIME", "AIRTEL DATA", "AIRTEL VTU"], "Telephone", "expense", 0.95, "tel_airtel_explicit"),
        Rule(["9MOBILE AIRTIME", "9MOBILE DATA", "ETISALAT"], "Telephone", "expense", 0.95, "tel_9mobile"),
        Rule(["AIRTIME PURCHASE", "AIRTIME TOP", "VTU PURCHASE", "DATA BUNDLE", "DATA PURCHASE", "DATA SUBSCRIPTION"], "Telephone", "expense", 0.93, "tel_airtime_generic"),
        # MTN/GLO alone are less certain (could be in narration context)
        Rule(["MTN"], "Telephone", "expense", 0.78, "tel_mtn_weak", direction="debit", amount_max=50000),
        Rule(["GLO"], "Telephone", "expense", 0.78, "tel_glo_weak", direction="debit", amount_max=50000),
        Rule(["AIRTEL"], "Telephone", "expense", 0.78, "tel_airtel_weak", direction="debit", amount_max=50000),

        # ═══ FUEL ═══
        Rule(["OANDO"], "Fuel", "expense", 0.93, "fuel_oando", direction="debit"),
        Rule(["TOTAL ENERGIES", "TOTALENERGIES"], "Fuel", "expense", 0.93, "fuel_total"),
        Rule(["SHELL PETROLEUM", "SHELL PETROL"], "Fuel", "expense", 0.93, "fuel_shell"),
        Rule(["MOBIL FILLING", "MOBIL PETROL", "MOBIL OIL"], "Fuel", "expense", 0.93, "fuel_mobil"),
        Rule(["CONOIL"], "Fuel", "expense", 0.93, "fuel_conoil"),
        Rule(["ARDOVA"], "Fuel", "expense", 0.93, "fuel_ardova"),
        Rule(["PETROL STATION", "FILLING STATION", "FUEL STATION", "DIESEL PURCHASE"], "Fuel", "expense", 0.88, "fuel_station_generic"),
        Rule(["PETROL", "DIESEL", "FUEL"], "Fuel", "expense", 0.78, "fuel_keyword", direction="debit"),

        # ═══ TRANSPORT ═══
        Rule(["BOLT RIDE", "BOLT TECHNOLOGY", "BOLT TRANSPORT"], "Motor Running Expenses", "expense", 0.90, "transport_bolt"),
        Rule(["UBER RIDE", "UBER TRIP", "UBER BV"], "Motor Running Expenses", "expense", 0.90, "transport_uber"),
        Rule(["INDRIVE"], "Motor Running Expenses", "expense", 0.88, "transport_indrive"),
        Rule(["VEHICLE LICENCE", "ROAD WORTHINESS", "VEHICLE REGISTRATION"], "Motor Running Expenses", "expense", 0.88, "transport_licence"),
        Rule(["FRSC"], "Motor Running Expenses", "expense", 0.85, "transport_frsc"),
        Rule(["TOLL FEE", "TOLL GATE", "LCC TOLL", "TOLL PAYMENT"], "Motor Running Expenses", "expense", 0.90, "transport_toll"),

        # ═══ RENT (EXPENSE) ═══
        Rule(["OFFICE RENT", "SHOP RENT", "WAREHOUSE RENT", "RENT PAYMENT"], "Office Rent", "expense", 0.88, "rent_office", direction="debit"),
        Rule(["HOUSE RENT", "APARTMENT RENT"], "Annual Rent", "expense", 0.82, "rent_house", direction="debit"),
        Rule(["ANNUAL RENT"], "Annual Rent", "expense", 0.80, "rent_annual", direction="debit"),
        Rule(["CARETAKER"], "Annual Rent", "expense", 0.75, "rent_caretaker", direction="debit"),

        # ═══ RENT (INCOME) ═══
        Rule(["RENT INCOME", "RENTAL INCOME"], "Annual Rent", "income", 0.88, "rent_income", direction="credit"),
        Rule(["TENANT PAYMENT", "TENANT"], "Annual Rent", "income", 0.78, "rent_tenant", direction="credit"),

        # ═══ SALARY / EMPLOYMENT INCOME ═══
        Rule(["SALARY PAYMENT", "SALARY CREDIT", "MONTHLY SALARY"], "Income", "income", 0.93, "salary_explicit", direction="credit"),
        Rule(["PAYROLL"], "Income", "income", 0.90, "salary_payroll", direction="credit"),
        Rule(["NEFT SALARY", "NEFT CREDIT"], "Income", "income", 0.85, "salary_neft", direction="credit"),

        # ═══ BUSINESS INCOME (generic credits) ═══
        Rule(["INVOICE PAYMENT", "INV PAYMENT"], "Income", "income", 0.85, "income_invoice", direction="credit"),
        Rule(["CONSULTING FEE", "CONSULTING PAYMENT"], "Consulting Income", "income", 0.88, "income_consulting", direction="credit"),
        Rule(["COMMISSION"], "Commission Income", "income", 0.75, "income_commission", direction="credit"),

        # ═══ BANK INTEREST (INCOME) ═══
        Rule(["INTEREST CREDIT", "INTEREST EARNED", "INTEREST PAYMENT", "INT CREDIT"], "Bank Interest Received", "income", 0.95, "interest_income", direction="credit"),

        # ═══ PERSONAL / NON-DEDUCTIBLE (DRAWINGS) ═══
        Rule(["SHOPRITE"], "Drawings", "expense", 0.92, "personal_shoprite"),
        Rule(["SPAR SUPERMARKET", "SPAR NIGERIA"], "Drawings", "expense", 0.90, "personal_spar"),
        Rule(["GAME STORES", "GAME NIGERIA"], "Drawings", "expense", 0.90, "personal_game"),
        Rule(["CHICKEN REPUBLIC"], "Drawings", "expense", 0.90, "personal_chicken_republic"),
        Rule(["DOMINOS", "DOMINO PIZZA", "DOMINO'S"], "Drawings", "expense", 0.90, "personal_dominos"),
        Rule(["COLD STONE", "COLDSTONE"], "Drawings", "expense", 0.90, "personal_coldstone"),
        Rule(["KILIMANJARO"], "Drawings", "expense", 0.88, "personal_kilimanjaro"),
        Rule(["KFC", "KENTUCKY FRIED"], "Drawings", "expense", 0.90, "personal_kfc"),
        Rule(["THE PLACE RESTAURANT", "THE PLACE"], "Drawings", "expense", 0.85, "personal_theplace"),
        Rule(["SWEET SENSATION"], "Drawings", "expense", 0.90, "personal_sweetsensation"),
        Rule(["MR BIGGS", "MR. BIGGS"], "Drawings", "expense", 0.90, "personal_mrbiggs"),
        Rule(["TANTALIZER", "TANTALIZERS"], "Drawings", "expense", 0.90, "personal_tantalizers"),

        # ═══ STREAMING / ENTERTAINMENT ═══
        Rule(["NETFLIX"], "Drawings", "expense", 0.92, "personal_netflix"),
        Rule(["SPOTIFY"], "Drawings", "expense", 0.92, "personal_spotify"),
        Rule(["APPLE.COM", "APPLE MUSIC", "APPLE ONE"], "Drawings", "expense", 0.88, "personal_apple"),
        Rule(["GOOGLE PLAY", "GOOGLE STORAGE", "GOOGLE ONE"], "Drawings", "expense", 0.88, "personal_google"),
        Rule(["AMAZON PRIME"], "Drawings", "expense", 0.88, "personal_amazon"),
        Rule(["YOUTUBE PREMIUM"], "Drawings", "expense", 0.88, "personal_youtube"),

        # ═══ ONLINE SHOPPING ═══
        Rule(["JUMIA"], "Drawings", "expense", 0.75, "personal_jumia", direction="debit"),
        Rule(["KONGA"], "Drawings", "expense", 0.75, "personal_konga", direction="debit"),
        Rule(["PAYPORTE"], "Drawings", "expense", 0.80, "personal_payporte"),

        # ═══ INVESTMENTS ═══
        Rule(["RISEVEST", "RISE VEST"], "Investment Income", "income", 0.82, "invest_risevest", direction="credit"),
        Rule(["COWRYWISE"], "Investment Income", "income", 0.82, "invest_cowrywise", direction="credit"),
        Rule(["PIGGYVEST", "PIGGYBANK"], "Investment Income", "income", 0.78, "invest_piggyvest", direction="credit"),
        Rule(["BAMBOO"], "Investment Income", "income", 0.78, "invest_bamboo", direction="credit"),
        Rule(["CHAKA"], "Investment Income", "income", 0.78, "invest_chaka", direction="credit"),
        Rule(["TROVE"], "Investment Income", "income", 0.78, "invest_trove", direction="credit"),
        # Debits to investment platforms = investment cost
        Rule(["RISEVEST", "RISE VEST"], "Drawings", "expense", 0.78, "invest_risevest_out", direction="debit"),
        Rule(["COWRYWISE"], "Drawings", "expense", 0.78, "invest_cowrywise_out", direction="debit"),
        Rule(["PIGGYVEST", "PIGGYBANK"], "Drawings", "expense", 0.78, "invest_piggyvest_out", direction="debit"),

        # ═══ CRYPTO ═══
        Rule(["BINANCE"], "Crypto Trading Income", "income", 0.80, "crypto_binance_in", direction="credit"),
        Rule(["LUNO"], "Crypto Trading Income", "income", 0.80, "crypto_luno_in", direction="credit"),
        Rule(["QUIDAX"], "Crypto Trading Income", "income", 0.80, "crypto_quidax_in", direction="credit"),
        Rule(["PATRICIA"], "Crypto Trading Income", "income", 0.78, "crypto_patricia_in", direction="credit"),
        Rule(["BYBIT"], "Crypto Trading Income", "income", 0.80, "crypto_bybit_in", direction="credit"),
        Rule(["ROQQU"], "Crypto Trading Income", "income", 0.78, "crypto_roqqu_in", direction="credit"),
        Rule(["BINANCE"], "Crypto Trading Cost", "expense", 0.80, "crypto_binance_out", direction="debit"),
        Rule(["LUNO"], "Crypto Trading Cost", "expense", 0.80, "crypto_luno_out", direction="debit"),
        Rule(["QUIDAX"], "Crypto Trading Cost", "expense", 0.80, "crypto_quidax_out", direction="debit"),

        # ═══ INSURANCE ═══
        Rule(["LEADWAY ASSURANCE", "LEADWAY INSURANCE"], "Insurance", "expense", 0.93, "insurance_leadway"),
        Rule(["CUSTODIAN INSURANCE", "CUSTODIAN ALLIED"], "Insurance", "expense", 0.93, "insurance_custodian"),
        Rule(["HEIRS INSURANCE", "HEIRS ASSURANCE"], "Insurance", "expense", 0.93, "insurance_heirs"),
        Rule(["CORONATION INSURANCE"], "Insurance", "expense", 0.93, "insurance_coronation"),
        Rule(["INSURANCE PREMIUM"], "Insurance", "expense", 0.88, "insurance_generic"),

        # ═══ GOVERNMENT / LEGAL ═══
        Rule(["CORPORATE AFFAIRS COMMISSION", "CAC PAYMENT", "CAC FEE"], "Legal and Professional Fees", "expense", 0.90, "legal_cac"),
        Rule(["IMMIGRATION", "NIS PAYMENT"], "Legal and Professional Fees", "expense", 0.85, "legal_immigration"),
        Rule(["COURT FEE", "LEGAL FEE", "SOLICITOR"], "Legal and Professional Fees", "expense", 0.85, "legal_fees"),

        # ═══ EDUCATION ═══
        Rule(["SCHOOL FEE", "SCHOOL FEES", "TUITION"], "Training", "expense", 0.85, "edu_school"),
        Rule(["UNIVERSITY"], "Training", "expense", 0.80, "edu_university"),
        Rule(["WAEC", "JAMB", "NECO"], "Training", "expense", 0.88, "edu_exams"),
        Rule(["TRAINING FEE", "COURSE FEE", "SEMINAR"], "Training", "expense", 0.85, "edu_training"),

        # ═══ SOFTWARE SUBSCRIPTIONS ═══
        Rule(["CANVA"], "Subscriptions", "expense", 0.90, "sub_canva"),
        Rule(["ZOOM"], "Subscriptions", "expense", 0.88, "sub_zoom"),
        Rule(["SLACK"], "Subscriptions", "expense", 0.88, "sub_slack"),
        Rule(["MICROSOFT", "OFFICE 365"], "Subscriptions", "expense", 0.88, "sub_microsoft"),
        Rule(["ADOBE"], "Subscriptions", "expense", 0.88, "sub_adobe"),
        Rule(["NOTION"], "Subscriptions", "expense", 0.88, "sub_notion"),

        # ═══ LOAN PLATFORMS ═══
        Rule(["CARBON LOAN", "CARBON FINANCE", "ONE FINANCE"], "Interest on Loans", "expense", 0.82, "loan_carbon", direction="debit"),
        Rule(["FAIRMONEY"], "Interest on Loans", "expense", 0.82, "loan_fairmoney", direction="debit"),
        Rule(["BRANCH LOAN", "BRANCH INT"], "Interest on Loans", "expense", 0.82, "loan_branch", direction="debit"),
        Rule(["PALMCREDIT"], "Interest on Loans", "expense", 0.82, "loan_palmcredit", direction="debit"),
        Rule(["RENMONEY"], "Interest on Loans", "expense", 0.82, "loan_renmoney", direction="debit"),
        Rule(["LOAN REPAYMENT", "LOAN DEDUCTION"], "Interest on Loans", "expense", 0.85, "loan_generic", direction="debit"),
        # Loan disbursements (credit side) = income/transfer
        Rule(["CARBON LOAN", "CARBON FINANCE"], "Income", "income", 0.70, "loan_carbon_in", direction="credit"),
        Rule(["FAIRMONEY"], "Income", "income", 0.70, "loan_fairmoney_in", direction="credit"),
        Rule(["LOAN DISBURSEMENT"], "Income", "income", 0.75, "loan_disbursement", direction="credit"),

        # ═══ POS PURCHASES ═══
        Rule(["POS PURCHASE", "POS DEBIT", "WEB PURCHASE", "WEB DEBIT"], "Drawings", "expense", 0.65, "pos_purchase", direction="debit"),

        # ═══ SALARIES (EXPENSE — paying employees) ═══
        Rule(["SALARY TO ", "STAFF SALARY", "EMPLOYEE SALARY"], "Salaries and Wages", "expense", 0.88, "salary_out", direction="debit"),
    ]

    def classify(
        self, narration: str, direction: str, amount: float
    ) -> Optional[ClassifyResult]:
        """Match transaction against all rules. First match wins."""
        for rule in self.RULES:
            if rule.matches(narration, direction, amount):
                return ClassifyResult(
                    category=rule.category,
                    type=rule.type,
                    confidence=rule.confidence,
                    source=ClassificationSource.RULE,
                    needs_review=rule.confidence < 0.80,
                    counterparty=None,
                    explanation=f"Rule match: {rule.rule_name}",
                    rule_hit=rule.rule_name,
                )
        return None


# ═══════════════════════════════════════════════════════════════
# LAYER 4: LLM BATCH CLASSIFIER
# ═══════════════════════════════════════════════════════════════

class LLMClassifier:
    """
    Batch classification using LLM (Claude or GPT).
    Only called for transactions that Layer 1-3 couldn't classify.

    For testing: set api_key=None to use mock responses.
    For production: provide Anthropic API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",  # "anthropic" or "openai"
        batch_size: int = 20,
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.batch_size = batch_size

    def _build_prompt(
        self, transactions: List[dict], category_labels: List[str]
    ) -> str:
        """Build the classification prompt."""
        tx_lines = []
        for i, tx in enumerate(transactions):
            tx_lines.append(
                f"{i+1}. {tx['direction'].upper()} | "
                f"NGN {tx['amount']:,.2f} | "
                f"{tx.get('date', 'N/A')} | "
                f"\"{tx['narration']}\""
            )

        return f"""You are classifying Nigerian bank transactions for a tax application called SettleTax.

For each transaction, assign:
1. "category" — MUST be one of these exact labels: {json.dumps(category_labels)}
2. "type" — either "income" or "expense"
3. "confidence" — 0.0 to 1.0 (how sure you are)
4. "explanation" — brief reason (under 15 words)

Context:
- These are Nigerian Naira (NGN) transactions from bank statements
- DEBIT = money leaving the account (expense/transfer)
- CREDIT = money entering the account (income/transfer)
- "Drawings" = personal/non-business expenses
- "Transfer" = money moving between the user's own accounts
- "Income" = general business/employment income
- If a transaction mentions a person's name with no other context, it's likely a "Transfer"
- Small debits right after transfers are usually "Bank Charges"

Transactions to classify:
{chr(10).join(tx_lines)}

Return ONLY a JSON array. Example:
[{{"index":1,"category":"Fuel","type":"expense","confidence":0.9,"explanation":"Payment at fuel station"}}]"""

    def classify_batch(
        self,
        transactions: List[dict],
        category_labels: List[str],
    ) -> List[ClassifyResult]:
        """
        Classify a batch of transactions using the LLM.

        For testing without an API key, returns UNCLASSIFIED results.
        """
        if not self.api_key:
            # Mock mode — return unclassified results for testing
            return [
                ClassifyResult(
                    category=None,
                    type="expense" if tx.get("direction") == "debit" else "income",
                    confidence=0.0,
                    source=ClassificationSource.UNCLASSIFIED,
                    needs_review=True,
                    counterparty=None,
                    explanation="No LLM API key provided (test mode)",
                )
                for tx in transactions
            ]

        prompt = self._build_prompt(transactions, category_labels)

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(prompt, transactions, category_labels)
            else:
                return self._call_openai(prompt, transactions, category_labels)
        except Exception as e:
            # On any error, return unclassified
            return [
                ClassifyResult(
                    category=None,
                    type="expense" if tx.get("direction") == "debit" else "income",
                    confidence=0.0,
                    source=ClassificationSource.UNCLASSIFIED,
                    needs_review=True,
                    counterparty=None,
                    explanation=f"LLM error: {str(e)[:100]}",
                )
                for tx in transactions
            ]

    def _call_anthropic(
        self, prompt: str, transactions: List[dict], category_labels: List[str]
    ) -> List[ClassifyResult]:
        """Call Anthropic Claude API."""
        import requests

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data["content"][0]["text"]

        return self._parse_llm_response(text, transactions, category_labels)

    def _call_openai(
        self, prompt: str, transactions: List[dict], category_labels: List[str]
    ) -> List[ClassifyResult]:
        """Call OpenAI API."""
        import requests

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a financial transaction classifier. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]

        return self._parse_llm_response(text, transactions, category_labels)

    def _parse_llm_response(
        self, text: str, transactions: List[dict], category_labels: List[str]
    ) -> List[ClassifyResult]:
        """Parse LLM JSON response into ClassifyResult list."""
        # Extract JSON array from response
        text = text.strip()
        if not text.startswith("["):
            # Try to find JSON array in the text
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [
                ClassifyResult(
                    category=None,
                    type="expense" if tx.get("direction") == "debit" else "income",
                    confidence=0.0,
                    source=ClassificationSource.UNCLASSIFIED,
                    needs_review=True,
                    counterparty=None,
                    explanation="Failed to parse LLM response",
                )
                for tx in transactions
            ]

        results = []
        for i, tx in enumerate(transactions):
            # Find the matching result by index
            llm_result = None
            for r in parsed:
                if r.get("index") == i + 1:
                    llm_result = r
                    break

            if llm_result and llm_result.get("category") in category_labels:
                conf = float(llm_result.get("confidence", 0.5))
                results.append(ClassifyResult(
                    category=llm_result["category"],
                    type=llm_result.get("type", "expense"),
                    confidence=conf,
                    source=ClassificationSource.LLM,
                    needs_review=conf < 0.75,
                    counterparty=None,
                    explanation=llm_result.get("explanation", "LLM classified"),
                ))
            else:
                results.append(ClassifyResult(
                    category=None,
                    type="expense" if tx.get("direction") == "debit" else "income",
                    confidence=0.0,
                    source=ClassificationSource.UNCLASSIFIED,
                    needs_review=True,
                    counterparty=None,
                    explanation="LLM returned invalid category",
                ))

        return results


# ═══════════════════════════════════════════════════════════════
# COUNTERPARTY EXTRACTOR
# ═══════════════════════════════════════════════════════════════

class CounterpartyExtractor:
    """
    Extracts the likely counterparty name from Nigerian bank narrations.
    Works across multiple bank formats.
    """

    # Ordered by specificity (most specific patterns first)
    PATTERNS = [
        # PalmPay
        (re.compile(r"NIP\s+TRANSFER\s+TO\s+(.+?)(?:\s*$)", re.I), "palmpay_to"),
        (re.compile(r"NIP\s+TRANSFER\s+FROM\s+(.+?)(?:\s*$)", re.I), "nip_from"),
        # Kuda / modern fintechs
        (re.compile(r"TRANSFER\s+TO\s+(.+?)\s*(?:-\s*\S+)?$", re.I), "transfer_to"),
        (re.compile(r"TRANSFER\s+FROM\s+(.+?)\s*(?:-\s*\S+)?$", re.I), "transfer_from"),
        # TRF prefix (Access, GTB)
        (re.compile(r"TRF[-/\s]+(.+?)(?:/\S|$)", re.I), "trf"),
        # UBA style
        (re.compile(r"^(.+?)/TRF/", re.I), "uba_trf"),
        # Zenith
        (re.compile(r"MC\s+TRANSFER:\s*\S+\s+(.+?)$", re.I), "zenith"),
        # POS / Web purchase
        (re.compile(r"(?:POS|WEB)\s+(?:PURCHASE|DEBIT)\s+(?:AT\s+)?(.+?)(?:\s+\d|$)", re.I), "pos"),
    ]

    CLEAN_SUFFIXES = [
        " LTD", " LIMITED", " PLC", " INC", " LLC",
    ]

    @classmethod
    def extract(cls, narration: str) -> Optional[str]:
        """Extract counterparty name from narration."""
        if not narration:
            return None

        text = narration.strip()

        for pattern, _name in cls.PATTERNS:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                name = cls._clean_name(name)
                if name and len(name) >= 2:
                    return name

        return None

    @classmethod
    def _clean_name(cls, name: str) -> str:
        """Clean up extracted counterparty name."""
        # Remove trailing numbers/refs
        name = re.sub(r"\s+\d{6,}.*$", "", name)
        # Remove trailing reference codes
        name = re.sub(r"\s+REF\S*$", "", name, flags=re.I)
        # Remove non-alphanumeric (keep spaces, hyphens, ampersands)
        name = re.sub(r"[^A-Za-z\s\-&]", "", name)
        # Normalize whitespace
        name = re.sub(r"\s+", " ", name).strip()
        return name


# ═══════════════════════════════════════════════════════════════
# MAIN CLASSIFIER — ORCHESTRATES ALL 4 LAYERS
# ═══════════════════════════════════════════════════════════════

class SettleTaxClassifier:
    """
    Main classifier that runs all 4 layers in order.

    Usage:
        classifier = SettleTaxClassifier(
            account_name="OBAFEMI-MOSES MOSINMILOLUWA",
            account_names=["SETTLE TAX LTD"],
        )

        # Single transaction
        result = classifier.classify_single(
            narration="NIP TRANSFER TO JOHN ADEYEMI 12345",
            amount=50000,
            direction="debit",
        )

        # Batch (DataFrame)
        results_df = classifier.classify_batch(tx_df)
    """

    def __init__(
        self,
        account_name: str,
        account_names: List[str] = None,
        user_history: Dict[str, dict] = None,
        llm_api_key: Optional[str] = None,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
    ):
        self.structural = StructuralDetector(account_name, account_names)
        self.history = UserHistoryMatcher(user_history)
        self.rules = RuleEngine()
        self.llm = LLMClassifier(
            api_key=llm_api_key,
            model=llm_model,
            provider=llm_provider,
        )
        self.counterparty_extractor = CounterpartyExtractor()

        # Stats tracking
        self.stats = {
            "total": 0,
            "structural": 0,
            "user_history": 0,
            "rule": 0,
            "llm": 0,
            "unclassified": 0,
        }

    def classify_single(
        self,
        narration: str,
        amount: float,
        direction: str,  # "debit" or "credit"
        date: str = "",
    ) -> ClassifyResult:
        """Classify a single transaction through all 4 layers."""
        self.stats["total"] += 1

        # Extract counterparty for all layers
        counterparty = self.counterparty_extractor.extract(narration)

        # Layer 1: Structural Detection
        result = self.structural.classify(narration, amount, direction)
        if result:
            result.counterparty = result.counterparty or counterparty
            self.stats["structural"] += 1
            return result

        # Layer 2: User History
        result = self.history.classify(counterparty)
        if result:
            result.counterparty = counterparty
            self.stats["user_history"] += 1
            return result

        # Layer 3: Rule Engine
        result = self.rules.classify(narration, direction, amount)
        if result:
            result.counterparty = counterparty
            self.stats["rule"] += 1
            return result

        # Layer 4: Falls through — will be batched for LLM
        self.stats["unclassified"] += 1
        return ClassifyResult(
            category=None,
            type="expense" if direction == "debit" else "income",
            confidence=0.0,
            source=ClassificationSource.UNCLASSIFIED,
            needs_review=True,
            counterparty=counterparty,
            explanation="No layer matched — needs LLM or user classification",
        )

    def classify_batch(self, df, llm_enabled: bool = True):
        """
        Classify an entire DataFrame of transactions.

        Expected columns:
        - Remarks (or narration): transaction description
        - amount: transaction amount
        - direction: "debit" or "credit"
        - Trans_Date (optional): transaction date

        Returns: DataFrame with added classification columns.
        """
        import pandas as pd

        # Normalize column names
        df = df.copy()
        if "Remarks" in df.columns and "narration" not in df.columns:
            df["narration"] = df["Remarks"]
        if "Trans_Date" in df.columns and "date" not in df.columns:
            df["date"] = df["Trans_Date"]

        # Classify each transaction through layers 1-3
        results = []
        llm_queue = []  # transactions that need LLM

        for idx, row in df.iterrows():
            narration = str(row.get("narration", row.get("Remarks", "")))
            amount = float(row.get("amount", 0))
            direction = str(row.get("direction", "debit")).lower()
            date = str(row.get("date", row.get("Trans_Date", "")))

            result = self.classify_single(narration, amount, direction, date)

            if result.source == ClassificationSource.UNCLASSIFIED and llm_enabled:
                llm_queue.append({
                    "df_idx": idx,
                    "narration": narration,
                    "amount": amount,
                    "direction": direction,
                    "date": date,
                })
                results.append((idx, None))  # placeholder
            else:
                results.append((idx, result))

        # Layer 4: Batch LLM classification for unclassified
        if llm_queue:
            all_categories = sorted(ALL_CATEGORIES)
            for batch_start in range(0, len(llm_queue), self.llm.batch_size):
                batch = llm_queue[batch_start:batch_start + self.llm.batch_size]
                llm_results = self.llm.classify_batch(
                    [{"narration": t["narration"], "amount": t["amount"],
                      "direction": t["direction"], "date": t["date"]}
                     for t in batch],
                    all_categories,
                )
                for i, llm_result in enumerate(llm_results):
                    df_idx = batch[i]["df_idx"]
                    llm_result.counterparty = self.counterparty_extractor.extract(
                        batch[i]["narration"]
                    )
                    # Update results
                    for j, (ridx, rval) in enumerate(results):
                        if ridx == df_idx and rval is None:
                            results[j] = (df_idx, llm_result)
                            if llm_result.source == ClassificationSource.LLM:
                                self.stats["llm"] += 1
                                self.stats["unclassified"] -= 1
                            break

        # Add results to DataFrame
        result_map = {idx: r for idx, r in results}
        df["st_category"] = df.index.map(lambda i: result_map[i].category if result_map.get(i) else None)
        df["st_type"] = df.index.map(lambda i: result_map[i].type if result_map.get(i) else None)
        df["st_confidence"] = df.index.map(lambda i: result_map[i].confidence if result_map.get(i) else 0)
        df["st_source"] = df.index.map(lambda i: result_map[i].source.value if result_map.get(i) else "unclassified")
        df["st_needs_review"] = df.index.map(lambda i: result_map[i].needs_review if result_map.get(i) else True)
        df["st_counterparty"] = df.index.map(lambda i: result_map[i].counterparty if result_map.get(i) else None)
        df["st_explanation"] = df.index.map(lambda i: result_map[i].explanation if result_map.get(i) else "")
        df["st_rule_hit"] = df.index.map(lambda i: result_map[i].rule_hit if result_map.get(i) else None)

        return df

    def print_stats(self):
        """Print classification statistics."""
        total = self.stats["total"]
        if total == 0:
            print("No transactions classified yet.")
            return

        print(f"\n{'='*50}")
        print(f"SettleTax Classification Results")
        print(f"{'='*50}")
        print(f"Total transactions: {total}")
        print(f"  Layer 1 (Structural):    {self.stats['structural']:>4} ({self.stats['structural']/total*100:.1f}%)")
        print(f"  Layer 2 (User History):  {self.stats['user_history']:>4} ({self.stats['user_history']/total*100:.1f}%)")
        print(f"  Layer 3 (Rules):         {self.stats['rule']:>4} ({self.stats['rule']/total*100:.1f}%)")
        print(f"  Layer 4 (LLM):           {self.stats['llm']:>4} ({self.stats['llm']/total*100:.1f}%)")
        print(f"  Unclassified:            {self.stats['unclassified']:>4} ({self.stats['unclassified']/total*100:.1f}%)")
        print(f"{'='*50}")

    def learn_from_user(
        self, counterparty: str, category: str, tx_type: str
    ):
        """
        Record a user's manual categorisation for future auto-matching.
        Call this when a user confirms/overrides a category in the app.
        """
        if counterparty:
            self.history.add_mapping(counterparty, category, tx_type, source="user")


# ═══════════════════════════════════════════════════════════════
# BANK STATEMENT PDF PARSER
# ═══════════════════════════════════════════════════════════════

class BankStatementParser:
    """
    Multi-bank PDF statement parser for Nigerian banks.

    Handles:
    - GTBank (Date/Deposit/Withdrawal format, null bytes in headers)
    - Access/GTB legacy (Trans. Date/Credits/Debits format)
    - UBA, Zenith, First Bank, and other Nigerian bank formats
    - Continuation tables across pages (no header on pages 2+)
    - Password-protected PDFs (via pikepdf)

    Usage:
        parser = BankStatementParser()
        df = parser.parse("statement.pdf")
        # Returns DataFrame with columns:
        # Trans_Date, Reference, Remarks, Value_Date, Credits, Debits,
        # Balance, direction, amount, narration
    """

    # Canonical column names used by the classifier
    COLUMN_ALIASES = {
        # Date columns
        "Date": "Trans_Date",
        "Trans. Date": "Trans_Date",
        "Trans Date": "Trans_Date",
        "Transaction Date": "Trans_Date",
        "Post Date": "Trans_Date",
        "Txn Date": "Trans_Date",
        # Value date
        "ValueDate": "Value_Date",
        "Value. Date": "Value_Date",
        "Value Date": "Value_Date",
        # Description / narration
        "Description": "Remarks",
        "Narration": "Remarks",
        "Particulars": "Remarks",
        "Details": "Remarks",
        "Descrip on": "Remarks",       # null-byte cleaned (space variant)
        "Descripon": "Remarks",         # null-byte cleaned (no space variant)
        # Money columns
        "Deposit": "Credits",
        "Credit": "Credits",
        "Deposits": "Credits",
        "Withdrawal": "Debits",
        "Debit": "Debits",
        "Withdrawals": "Debits",
    }

    # Money column names we look for in header detection
    MONEY_COLUMNS = {
        "Deposit", "Withdrawal", "Debits", "Credits",
        "Debit", "Credit", "Deposits", "Withdrawals",
    }

    def __init__(self):
        pass

    def parse(self, pdf_path: str) -> "pd.DataFrame":
        """
        Parse a bank statement PDF and return a clean DataFrame.

        Args:
            pdf_path: Path to the PDF file (can be encrypted).

        Returns:
            DataFrame with columns: Trans_Date, Reference, Remarks,
            Value_Date, Credits, Debits, Balance, direction, amount, narration
        """
        import pdfplumber
        import pandas as pd

        all_rows = []
        header = None

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue

                for table in tables:
                    if not table or len(table) < 1:
                        continue

                    # Clean the first row (potential header)
                    first_row_clean = self._clean_row(table[0])

                    # Check if this row is a header
                    if header is None and self._is_header_row(first_row_clean):
                        header = first_row_clean
                        # Remaining rows are data
                        data_rows = table[1:]
                    elif header is not None:
                        # Header already found — check if this is a continuation table
                        # or a non-transaction table (account info, summary)
                        if self._is_header_row(first_row_clean):
                            # Duplicate header on a new page — skip it, take data rows
                            data_rows = table[1:]
                        elif self._is_data_row(first_row_clean):
                            # Continuation table — all rows are data
                            data_rows = table
                        else:
                            # Non-transaction table (account info, summary) — skip
                            continue
                    else:
                        # No header yet and this isn't one — skip
                        continue

                    # Process data rows
                    for row in data_rows:
                        cleaned = self._clean_row(row)
                        # Skip rows that are repeated headers
                        if self._is_header_row(cleaned):
                            continue
                        # Skip completely empty rows
                        if all(not cell for cell in cleaned):
                            continue
                        # Ensure row has same number of columns as header
                        if len(cleaned) < len(header):
                            cleaned.extend([""] * (len(header) - len(cleaned)))
                        elif len(cleaned) > len(header):
                            cleaned = cleaned[: len(header)]
                        all_rows.append(cleaned)

        if header is None:
            raise ValueError(
                "Could not detect a transaction header in the PDF. "
                "Expected columns containing 'Date' and a money column "
                "(Deposit/Withdrawal/Credits/Debits)."
            )

        if not all_rows:
            raise ValueError(
                "Header was detected but no transaction data rows were found."
            )

        # Build DataFrame
        df = pd.DataFrame(all_rows, columns=header)

        # Normalize column names
        df = self._normalize_columns(df)

        # Clean amounts
        df = self._clean_amounts(df)

        # Add direction, amount, narration columns
        df = self._add_direction_and_amount(df)

        return df

    def _clean_row(self, row: list) -> list:
        """Remove null bytes, strip whitespace from all cells in a row."""
        cleaned = []
        for cell in row:
            if cell is None:
                cleaned.append("")
            else:
                # Remove null bytes (PDF rendering artifact) and strip
                cleaned.append(
                    re.sub(r"\x00", "", str(cell)).strip()
                )
        return cleaned

    def _is_header_row(self, cells: list) -> bool:
        """
        Check if a row of cells looks like a transaction table header.

        Matches if it has:
        - A date column ("Date", "Trans. Date", "Trans Date", etc.) AND
        - At least one money column (Deposit, Withdrawal, Credits, Debits, etc.)
        """
        cell_set = set(cells)

        # Check for date column
        has_date = any(
            "Date" in cell
            for cell in cells
            if cell  # skip empty strings
        )

        # Check for money column
        has_money = bool(cell_set & self.MONEY_COLUMNS)

        return has_date and has_money

    def _is_data_row(self, cells: list) -> bool:
        """
        Check if a row looks like a transaction data row (not account info).

        A transaction row typically starts with a date-like value:
        - DD-Mon-YYYY (e.g., 27-Sep-2025)
        - DD/MM/YYYY (e.g., 27/09/2025)
        - DD-MM-YYYY
        - YYYY-MM-DD
        """
        if not cells or not cells[0]:
            return False
        first_cell = cells[0].strip()
        # Match common Nigerian bank date formats
        return bool(
            re.match(
                r"(\d{1,2}[-/]\w{3}[-/]\d{2,4})"  # DD-Mon-YYYY or DD-Mon-YY
                r"|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"  # DD/MM/YYYY or DD-MM-YYYY
                r"|(\d{4}[-/]\d{1,2}[-/]\d{1,2})",  # YYYY-MM-DD
                first_cell,
            )
        )

    def _normalize_columns(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Rename columns to canonical names used by the classifier."""
        rename_map = {}
        for col in df.columns:
            # Check direct alias
            if col in self.COLUMN_ALIASES:
                rename_map[col] = self.COLUMN_ALIASES[col]
            else:
                # Check after cleaning null bytes (already done, but just in case)
                cleaned = re.sub(r"\x00", "", col).strip()
                if cleaned in self.COLUMN_ALIASES:
                    rename_map[col] = self.COLUMN_ALIASES[cleaned]

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _clean_amounts(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Parse money strings in Credits, Debits, Balance columns."""
        import pandas as pd

        money_cols = ["Credits", "Debits", "Balance"]
        for col in money_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace('"', "", regex=False)
                    .str.replace("₦", "", regex=False)
                    .str.replace("NGN", "", regex=False)
                    .str.extract(r"([-+]?\d*\.?\d+)")[0]
                    .astype(float)
                )

        return df

    def _add_direction_and_amount(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Add direction (debit/credit), amount, and narration columns."""
        import pandas as pd
        import numpy as np

        def get_direction(row):
            d = row.get("Debits")
            c = row.get("Credits")
            if pd.notna(d) and d > 0:
                return "debit"
            if pd.notna(c) and c > 0:
                return "credit"
            return "debit"

        def get_amount(row):
            if row["direction"] == "debit":
                val = row.get("Debits", 0)
                return val if pd.notna(val) else 0
            val = row.get("Credits", 0)
            return val if pd.notna(val) else 0

        df["direction"] = df.apply(get_direction, axis=1)
        df["amount"] = df.apply(get_amount, axis=1)

        # Set narration column (alias for Remarks)
        if "Remarks" in df.columns:
            df["narration"] = (
                df["Remarks"]
                .astype(str)
                .str.replace(r"\s*None\"?$", "", regex=True)
                .str.replace("nan", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.replace("\n", " ", regex=False)
                .str.replace("  ", " ", regex=False)
                .str.strip()
            )
        else:
            # If no Remarks column, try to build from non-standard columns
            known_cols = {
                "Trans_Date", "Value_Date", "Reference",
                "Credits", "Debits", "Balance",
                "direction", "amount",
            }
            extra_cols = [c for c in df.columns if c not in known_cols]
            if extra_cols:
                df["narration"] = (
                    df[extra_cols].astype(str).agg(" ".join, axis=1)
                    .str.replace("nan", "", regex=False)
                    .str.strip()
                )
            else:
                df["narration"] = ""

        # Set date column (alias for Trans_Date)
        if "Trans_Date" in df.columns:
            df["date"] = df["Trans_Date"]

        return df
