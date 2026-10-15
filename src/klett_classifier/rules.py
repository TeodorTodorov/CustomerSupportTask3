"""Deterministic demonstration baseline, NOT a calibrated ML model."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .inference import InvalidQuery
from .labels import Category
from .security import validate_query


@dataclass(frozen=True, slots=True)
class RulePrediction:
    category: str
    confidence: float | None = None
    scores: dict[str, float] | None = None
    method: str = "rules"

    def to_dict(self):
        return {"category": self.category, "confidence": self.confidence,
                "scores": self.scores or {}, "method": self.method}


class RuleClassifier:
    """Transparent fallback for the assignment and for incident-mode triage."""

    ACCESS = re.compile(r"парол|регистрац|акаунт|профил|вл[ея]з|вход|логин|имейл.*(?:смен|промен)|"
                        r"password|sign.?in|log.?in|registr|account|reset.*email|change.*email", re.I)
    FAULT = re.compile(r"не се (?:зарежда|отваря|пуска|вижда|стартира)|не работи|забива|грешка|срива|"
                       r"прекъсва|не мога да (?:отворя|заредя|пусна)|error|crash|freeze|fails? to|"
                       r"won.t load|not loading|does not load|not working|broken|blank screen", re.I)
    CONTENT = re.compile(r"урок|упражнен|учебн|помагал|тест|математик|истори|географ|"
                         r"д[оа]машн|контролн|видеоурок|lesson|exercise|course|textbook|quiz|"
                         r"homework|worksheet|curriculum|educational", re.I)

    def predict(self, text: str) -> RulePrediction:
        text = validate_query(text)
        # Account-related failures belong to the account team, even if technical.
        if self.ACCESS.search(text):
            category = Category.ACCESS
        elif self.FAULT.search(text):
            category = Category.TECHNICAL
        elif self.CONTENT.search(text):
            category = Category.CONTENT
        else:
            category = Category.OTHER
        return RulePrediction(category=category.value)

    def predict_batch(self, texts: list[str]) -> list[RulePrediction]:
        if not isinstance(texts, list):
            raise InvalidQuery("Expected an array of strings")
        return [self.predict(text) for text in texts]
