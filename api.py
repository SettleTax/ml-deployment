"""
SettleTax Classifier API
========================
FastAPI wrapper around SettleTaxClassifier.

Endpoints:
    POST /classify/single   — classify one transaction
    POST /classify/batch    — classify a list of transactions

Run locally:
    uvicorn api:app --reload --port 8000

Environment variables:
    ANTHROPIC_API_KEY   — optional, enables LLM fallback layer
    OPENAI_API_KEY      — optional, if you prefer OpenAI as LLM provider
    LLM_PROVIDER        — "anthropic" (default) or "openai"
"""

import os
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from settletax_classifier import SettleTaxClassifier, NarrationCache


# Module-level shared cache — lives for the lifetime of the process.
# All requests to /classify/multi-user share this, so LLM results from
# one batch are reused in the next batch automatically.
_shared_cache = NarrationCache()


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="SettleTax Transaction Classifier",
    description="4-layer Nigerian bank transaction classifier (structural → history → rules → LLM)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class Transaction(BaseModel):
    narration: str = Field(..., description="Bank narration / remarks text")
    amount: float = Field(..., gt=0, description="Transaction amount (positive number)")
    direction: str = Field(..., pattern="^(debit|credit)$", description="'debit' or 'credit'")
    date: Optional[str] = Field(None, description="Transaction date (any format, optional)")


class SingleRequest(BaseModel):
    account_name: str = Field(..., description="Account holder full name (used for self-transfer detection)")
    account_names: Optional[List[str]] = Field(None, description="Additional name aliases (e.g. business name)")
    transaction: Transaction
    user_history: Optional[Dict[str, dict]] = Field(
        None,
        description="Known counterparty→category mappings from past user actions",
    )
    llm_api_key: Optional[str] = Field(None, description="Override API key for LLM layer (else uses env var)")
    llm_provider: Optional[str] = Field("anthropic", description="'anthropic' or 'openai'")
    llm_enabled: Optional[bool] = Field(True, description="Set false to skip LLM layer entirely")


class BatchRequest(BaseModel):
    account_name: str = Field(..., description="Account holder full name")
    account_names: Optional[List[str]] = Field(None)
    transactions: List[Transaction] = Field(..., min_length=1)
    user_history: Optional[Dict[str, dict]] = Field(None)
    llm_api_key: Optional[str] = Field(None)
    llm_provider: Optional[str] = Field("anthropic")
    llm_enabled: Optional[bool] = Field(True)


class UserBatch(BaseModel):
    """One user's data inside a multi-user request."""
    account_name: str = Field(..., description="Account holder full name")
    account_names: Optional[List[str]] = Field(None, description="Additional name aliases")
    transactions: List[Transaction] = Field(..., min_length=1)
    user_history: Optional[Dict[str, dict]] = Field(None)


class MultiUserRequest(BaseModel):
    """
    Classify transactions for multiple users in one optimised call.

    Example:
    [
      {
        "account_name": "JOHN ADEYEMI",
        "transactions": [
          {"narration": "MTN AIRTIME PURCHASE", "amount": 1000, "direction": "debit"},
          {"narration": "SALARY CREDIT", "amount": 350000, "direction": "credit"}
        ]
      },
      {
        "account_name": "AMAKA OKONKWO",
        "transactions": [
          {"narration": "MTN AIRTIME PURCHASE", "amount": 500, "direction": "debit"}
        ]
      }
    ]
    """
    users: List[UserBatch] = Field(..., min_length=1)
    llm_api_key: Optional[str] = Field(None)
    llm_provider: Optional[str] = Field("anthropic")
    llm_enabled: Optional[bool] = Field(True)


class ClassifyResult(BaseModel):
    category: Optional[str]
    type: str
    confidence: float
    source: str
    needs_review: bool
    counterparty: Optional[str]
    explanation: str
    rule_hit: Optional[str]


class SingleResponse(BaseModel):
    result: ClassifyResult


class BatchResponse(BaseModel):
    results: List[ClassifyResult]
    stats: dict


class UserBatchResult(BaseModel):
    account_name: str
    results: List[ClassifyResult]
    stats: dict


class MultiUserResponse(BaseModel):
    users: List[UserBatchResult]
    cache_stats: dict


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _resolve_api_key(request_key: Optional[str], provider: str) -> Optional[str]:
    """Use request-level key first, fall back to environment variable."""
    if request_key:
        return request_key
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return os.getenv("ANTHROPIC_API_KEY")


def _build_classifier(req: SingleRequest | BatchRequest) -> SettleTaxClassifier:
    provider = req.llm_provider or "anthropic"
    api_key = _resolve_api_key(req.llm_api_key, provider)
    return SettleTaxClassifier(
        account_name=req.account_name,
        account_names=req.account_names,
        user_history=req.user_history,
        llm_api_key=api_key if req.llm_enabled else None,
        llm_provider=provider,
    )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify/single", response_model=SingleResponse)
def classify_single(req: SingleRequest):
    """
    Classify a single transaction.

    Example request body:
    ```json
    {
      "account_name": "JOHN ADEYEMI",
      "transaction": {
        "narration": "NIP TRANSFER TO JOHN ADEYEMI 1234567",
        "amount": 50000,
        "direction": "debit"
      }
    }
    ```
    """
    try:
        classifier = _build_classifier(req)
        tx = req.transaction
        raw = classifier.classify_single(
            narration=tx.narration,
            amount=tx.amount,
            direction=tx.direction,
            date=tx.date or "",
        )
        return SingleResponse(result=ClassifyResult(**raw.to_dict()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchResponse)
def classify_batch(req: BatchRequest):
    """
    Classify a list of transactions in one call.

    Transactions that pass layers 1-3 are returned immediately.
    Remaining ones are batched to the LLM layer (if llm_enabled=true).

    Example request body:
    ```json
    {
      "account_name": "JOHN ADEYEMI",
      "transactions": [
        {"narration": "MTN AIRTIME PURCHASE", "amount": 1000, "direction": "debit"},
        {"narration": "SALARY CREDIT MAY 2025", "amount": 350000, "direction": "credit"}
      ]
    }
    ```
    """
    try:
        import pandas as pd

        classifier = _build_classifier(req)

        # Build a DataFrame from the transaction list
        rows = [
            {
                "narration": tx.narration,
                "amount": tx.amount,
                "direction": tx.direction,
                "date": tx.date or "",
            }
            for tx in req.transactions
        ]
        df = pd.DataFrame(rows)

        result_df = classifier.classify_batch(df, llm_enabled=req.llm_enabled)

        results = []
        for _, row in result_df.iterrows():
            results.append(ClassifyResult(
                category=row.get("st_category"),
                type=row.get("st_type") or "expense",
                confidence=float(row.get("st_confidence") or 0),
                source=row.get("st_source") or "unclassified",
                needs_review=bool(row.get("st_needs_review", True)),
                counterparty=row.get("st_counterparty"),
                explanation=row.get("st_explanation") or "",
                rule_hit=row.get("st_rule_hit"),
            ))

        return BatchResponse(results=results, stats=classifier.stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/multi-user", response_model=MultiUserResponse)
def classify_multi_user(req: MultiUserRequest):
    """
    Classify transactions for multiple users in one optimised call.

    Compared to calling /classify/batch once per user:
    - Layers 1-3 run in parallel across all users.
    - LLM-needing transactions are deduplicated across ALL users before
      calling the LLM — same narration from 50 users = 1 LLM call.
    - Results are cached in the server-level NarrationCache so subsequent
      requests benefit immediately.

    Send an array of user objects, each with account_name + transactions:
    ```json
    [
      {
        "account_name": "JOHN ADEYEMI",
        "transactions": [
          {"narration": "MTN AIRTIME PURCHASE", "amount": 1000, "direction": "debit"},
          {"narration": "SALARY CREDIT MAY 2025", "amount": 350000, "direction": "credit"}
        ]
      },
      {
        "account_name": "AMAKA OKONKWO",
        "transactions": [
          {"narration": "DSTV SUBSCRIPTION", "amount": 5000, "direction": "debit"}
        ]
      }
    ]
    ```
    Wrap the array in `{"users": [...]}` when posting.
    """
    try:
        import pandas as pd

        provider = req.llm_provider or "anthropic"
        api_key = _resolve_api_key(req.llm_api_key, provider)

        # Build one classifier per user, all sharing the module-level cache
        classifiers_and_dfs = []
        for user in req.users:
            classifier = SettleTaxClassifier(
                account_name=user.account_name,
                account_names=user.account_names,
                user_history=user.user_history,
                llm_api_key=api_key if req.llm_enabled else None,
                llm_provider=provider,
                shared_cache=_shared_cache,
            )
            rows = [
                {
                    "narration": tx.narration,
                    "amount": tx.amount,
                    "direction": tx.direction,
                    "date": tx.date or "",
                }
                for tx in user.transactions
            ]
            classifiers_and_dfs.append((classifier, pd.DataFrame(rows)))

        # Single optimised call — parallel layers 1-3, deduplicated LLM
        result_dfs = SettleTaxClassifier.classify_batch_multi_user(
            classifiers_and_dfs,
            shared_cache=_shared_cache,
            llm_enabled=req.llm_enabled,
        )

        # Build response
        user_results = []
        for user, result_df, (classifier, _) in zip(
            req.users, result_dfs, classifiers_and_dfs
        ):
            results = []
            for _, row in result_df.iterrows():
                results.append(ClassifyResult(
                    category=row.get("st_category"),
                    type=row.get("st_type") or "expense",
                    confidence=float(row.get("st_confidence") or 0),
                    source=row.get("st_source") or "unclassified",
                    needs_review=bool(row.get("st_needs_review", True)),
                    counterparty=row.get("st_counterparty"),
                    explanation=row.get("st_explanation") or "",
                    rule_hit=row.get("st_rule_hit"),
                ))
            user_results.append(UserBatchResult(
                account_name=user.account_name,
                results=results,
                stats=classifier.stats,
            ))

        return MultiUserResponse(
            users=user_results,
            cache_stats=_shared_cache.stats(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
