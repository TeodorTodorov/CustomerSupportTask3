"""Secure, local-only classifier API. The text is DATA, never an instruction."""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .inference import InvalidQuery, QueryClassifier
from .rules import RuleClassifier
from .security import (
    RequestBodyLimitMiddleware, SecuritySettings, SlidingWindowLimiter,
    is_authorized, validate_query,
)

LOG = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def clean(cls, text: str) -> str:
        return validate_query(text)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    texts: list[str] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def clean(cls, texts: list[str]) -> list[str]:
        return [validate_query(t) for t in texts]


class PredictionResponse(BaseModel):
    category: str
    confidence: float | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    method: str


def create_app(classifier=None, *, security: SecuritySettings | None = None, backend="model",
               model_dir: str | Path = "models/classifier") -> FastAPI:
    # Production settings fail closed when KLETT_API_KEY is missing.
    settings = security or SecuritySettings.from_environment()
    limiter = SlidingWindowLimiter(settings.requests_per_minute)
    inference_slots = threading.BoundedSemaphore(settings.max_concurrent_inference)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if classifier is not None:
            app.state.classifier = classifier
        elif backend == "rules":
            app.state.classifier = RuleClassifier()
        elif backend == "model":
            app.state.classifier = QueryClassifier(Path(model_dir))
        else:
            raise RuntimeError("Unknown backend")
        yield

    app = FastAPI(title="Klett Intent Classifier", version="3.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    trusted_hosts = [host.strip() for host in os.getenv(
        "KLETT_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"
    ).split(",") if host.strip()]
    if not trusted_hosts or "*" in trusted_hosts:
        raise RuntimeError("Set explicit KLETT_ALLOWED_HOSTS; '*' is not permitted")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        # Do not echo untrusted query text or schema-level raw inputs.
        return JSONResponse(status_code=422, content={"detail": "Invalid request body"})

    def auth_and_limit(request: Request, authorization: str | None = Header(default=None)):
        if not is_authorized(authorization, settings.api_key):
            raise HTTPException(status_code=401, detail="Unauthorized",
                                headers={"WWW-Authenticate": "Bearer"})
        # Identity is connected socket IP, not X-Forwarded-For (which is user-controlled).
        addr = request.client.host if request.client else "unknown"
        if not limiter.allow(addr):
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"})

    def run_prediction(call):
        if not inference_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="Inference capacity exceeded",
                                headers={"Retry-After": "2"})
        try:
            return call()
        except InvalidQuery as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            LOG.exception("Inference failure (query body redacted)")
            raise HTTPException(status_code=503, detail="Inference unavailable") from None
        finally:
            inference_slots.release()

    @app.get("/health/live", include_in_schema=False)
    def liveness():
        return {"status": "ok"}

    @app.get("/health/ready", dependencies=[Depends(auth_and_limit)])
    def readiness(request: Request):
        return {"ready": hasattr(request.app.state, "classifier")}

    @app.post("/v1/predict", dependencies=[Depends(auth_and_limit)], response_model=PredictionResponse)
    def predict(payload: QueryRequest, request: Request):
        return run_prediction(lambda: request.app.state.classifier.predict(payload.text).to_dict())

    @app.post("/v1/predict/batch", dependencies=[Depends(auth_and_limit)],
              response_model=list[PredictionResponse])
    def predict_batch(payload: BatchRequest, request: Request):
        if len(payload.texts) > settings.max_batch_size:
            raise HTTPException(status_code=413, detail="Batch size exceeded")
        return run_prediction(lambda: [p.to_dict() for p in
                                      request.app.state.classifier.predict_batch(payload.texts)])

    return app
