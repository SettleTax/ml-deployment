"""
SettleTax Classifier — Google Cloud Functions entry point
=========================================================
Deploy:
    gcloud functions deploy settletax-classifier \
        --gen2 \
        --runtime python311 \
        --trigger-http \
        --allow-unauthenticated \
        --entry-point classify \
        --set-env-vars ANTHROPIC_API_KEY=sk-ant-... \
        --region us-central1

Endpoints (all via the same function URL):
    POST <URL>/classify/single
    POST <URL>/classify/batch
    GET  <URL>/health
"""

import os
import json
import functions_framework
from settletax_classifier import SettleTaxClassifier


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _json_response(data: dict, status: int = 200):
    """Return a (body, status, headers) tuple for Cloud Functions."""
    return (
        json.dumps(data),
        status,
        {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
    )


def _parse_body(request) -> dict:
    try:
        return request.get_json(force=True) or {}
    except Exception:
        return {}


def _build_classifier(body: dict) -> SettleTaxClassifier:
    provider = body.get("llm_provider", "anthropic")

    # API key: request body first, then env var
    api_key = body.get("llm_api_key")
    if not api_key:
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            if provider == "openai"
            else os.environ.get("ANTHROPIC_API_KEY")
        )

    # Disable LLM if caller explicitly opts out
    if not body.get("llm_enabled", True):
        api_key = None

    return SettleTaxClassifier(
        account_name=body.get("account_name", ""),
        account_names=body.get("account_names"),
        user_history=body.get("user_history"),
        llm_api_key=api_key,
        llm_provider=provider,
    )


def _result_to_dict(result) -> dict:
    d = result.to_dict()
    return d


# ─────────────────────────────────────────────
# Route handlers
# ─────────────────────────────────────────────

def _handle_single(body: dict):
    if not body.get("account_name"):
        return _json_response({"error": "account_name is required"}, 400)

    tx = body.get("transaction")
    if not tx:
        return _json_response({"error": "transaction is required"}, 400)

    required_fields = ("narration", "amount", "direction")
    missing = [f for f in required_fields if f not in tx]
    if missing:
        return _json_response({"error": f"Missing fields in transaction: {missing}"}, 400)

    if tx["direction"] not in ("debit", "credit"):
        return _json_response({"error": "direction must be 'debit' or 'credit'"}, 400)

    classifier = _build_classifier(body)
    result = classifier.classify_single(
        narration=str(tx["narration"]),
        amount=float(tx["amount"]),
        direction=str(tx["direction"]),
        date=str(tx.get("date", "")),
    )
    return _json_response({"result": _result_to_dict(result)})


def _handle_batch(body: dict):
    if not body.get("account_name"):
        return _json_response({"error": "account_name is required"}, 400)

    transactions = body.get("transactions")
    if not transactions or not isinstance(transactions, list):
        return _json_response({"error": "transactions must be a non-empty list"}, 400)

    import pandas as pd

    rows = []
    for i, tx in enumerate(transactions):
        if "narration" not in tx or "amount" not in tx or "direction" not in tx:
            return _json_response(
                {"error": f"transactions[{i}] missing narration/amount/direction"}, 400
            )
        rows.append({
            "narration": str(tx["narration"]),
            "amount": float(tx["amount"]),
            "direction": str(tx["direction"]),
            "date": str(tx.get("date", "")),
        })

    classifier = _build_classifier(body)
    df = pd.DataFrame(rows)
    result_df = classifier.classify_batch(df, llm_enabled=body.get("llm_enabled", True))

    results = []
    for _, row in result_df.iterrows():
        results.append({
            "category": row.get("st_category"),
            "type": row.get("st_type") or "expense",
            "confidence": float(row.get("st_confidence") or 0),
            "source": row.get("st_source") or "unclassified",
            "needs_review": bool(row.get("st_needs_review", True)),
            "counterparty": row.get("st_counterparty"),
            "explanation": row.get("st_explanation") or "",
            "rule_hit": row.get("st_rule_hit"),
        })

    return _json_response({"results": results, "stats": classifier.stats})


# ─────────────────────────────────────────────
# Cloud Functions entry point
# ─────────────────────────────────────────────

@functions_framework.http
def classify(request):
    """
    Single Cloud Function that routes by path:
      GET  /health            → health check
      POST /classify/single   → single transaction
      POST /classify/batch    → batch of transactions
    """
    # CORS preflight
    if request.method == "OPTIONS":
        return (
            "",
            204,
            {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "3600",
            },
        )

    path = request.path.rstrip("/")

    if request.method == "GET" and path in ("", "/health"):
        return _json_response({"status": "ok", "service": "settletax-classifier"})

    if request.method != "POST":
        return _json_response({"error": "Method not allowed"}, 405)

    body = _parse_body(request)

    if path.endswith("/classify/single"):
        return _handle_single(body)

    if path.endswith("/classify/batch"):
        return _handle_batch(body)

    return _json_response({"error": f"Unknown path: {path}"}, 404)
