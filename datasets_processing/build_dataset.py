"""Build English intent-classification training candidates for Klett.

Inputs: Bitext customer support, TriageIQ (train only), optional local English/BG CSV.
Output: English text + Bulgarian target labels compatible with KlettClassifier.

Synthetic items are template-based candidates, NOT validated natural user queries.
Do not evaluate on examples derived from the same templates/sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

TECH = "Технически проблем"
ACCESS = "Достъп или регистрация"
CONTENT = "Учебно съдържание"
OTHER = "Друго"
LABELS = (TECH, ACCESS, CONTENT, OTHER)
LOG = logging.getLogger("klett.dataset")

BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
TRIAGE = "coldstart88/triageiq-dataset"

# Deliberately omit ambiguous intents such as complaint/review and newsletter.
BITEXT_ACCESS = {
    "create_account", "delete_account", "edit_account", "switch_account",
    "recover_password", "registration_problems",
}
BITEXT_OTHER = {
    "check_cancellation_fee", "delivery_options", "check_invoice", "get_invoice",
    "cancel_order", "change_order", "place_order", "check_payment_methods",
    "payment_issue", "check_refund_policy", "get_refund", "track_refund",
    "change_shipping_address", "set_up_shipping_address",
    "contact_customer_service", "contact_human_agent", "delivery_period", "track_order",
}
TRIAGE_LABELS = {
    "technical": TECH, "account": ACCESS, "billing": OTHER,
    "feature_request": OTHER, "other": OTHER,
}

SUBJECTS = [
    "math", "algebra", "geometry", "English", "grammar", "literature",
    "history", "geography", "biology", "chemistry", "physics", "science",
    "computer science", "French", "Spanish", "economics", "civics",
    "reading", "writing", "statistics", "calculus", "music", "art",
]
RESOURCES = [
    "video lesson", "worksheet", "practice test", "interactive quiz", "textbook",
    "homework assignment", "reading passage", "revision guide", "study notes",
    "lesson recording", "exercise set", "practice problems", "flashcards",
    "audio lesson", "answer key", "learning module", "course chapter",
    "mock exam", "lesson plan", "assignment", "worked examples",
]
STAGES = [
    "grade 3", "grade 4", "grade 5", "grade 6", "grade 7",
    "grade 8", "grade 9", "grade 10", "grade 11", "grade 12",
    "primary school", "middle school", "high school",
]
PLATFORMS = ["website", "mobile app", "learning portal", "student dashboard", "course page", "browser"]

CONTENT_PATTERNS = [
    "Where can I find the {resource} for {subject} in {stage}?",
    "Do you have a {resource} for {subject} for {stage}?",
    "I need the {subject} {resource} for {stage}.",
    "Which section contains the {subject} {resource} for {stage}?",
    "Can I access a {resource} for {subject} in {stage}?",
    "I'd like extra {subject} materials for {stage}; where are they?",
    "Is there a {resource} to prepare for {stage} {subject}?",
    "Please show me the {resource} for my {subject} class in {stage}.",
    "Are there more {subject} exercises available for {stage}?",
    "Where are the answers for the {subject} activities in {stage}?",
    "What topics are covered in the {subject} course for {stage}?",
    "Could you point me to a {subject} {resource} suitable for {stage}?",
    "Does the {subject} curriculum in {stage} include a {resource}?",
    "Looking for a {resource} in {subject} for {stage}.",
    "I want to study {subject}; can you recommend a {resource} for {stage}?",
    "Can teachers download the {subject} {resource} for {stage}?",
    "How many {subject} lessons are included for {stage}?",
    "Do you offer printable {subject} {resource} for {stage}?",
    "Could I get a {resource} for reviewing {subject} in {stage}?",
    "Is the {subject} {resource} included in the course for {stage}?",
]
TECH_FAILURES = [
    "doesn't load", "won't open", "keeps buffering", "freezes", "crashes",
    "shows a blank screen", "displays error 500", "displays an error message",
    "fails to start", "stops halfway through", "never finishes loading",
    "doesn't respond when clicked", "is stuck on the loading spinner",
    "shows a network error", "doesn't display correctly", "won't play",
    "closes unexpectedly", "is inaccessible due to a server error",
]
TECH_PATTERNS = [
    "The {resource} for {subject} in {stage} {failure}.",
    "When I open the {subject} {resource}, it {failure}.",
    "My {resource} on the {platform} {failure}.",
    "I cannot use the {subject} {resource} because it {failure}.",
    "The {platform} {failure} when I try the {subject} {resource}.",
    "The {resource} for {stage} {failure}; can you fix it?",
    "The {subject} {resource} {failure}. What can I do?",
    "Every time I select the {resource}, it {failure}.",
    "Help, the {subject} {resource} {failure} on my laptop.",
    "Since the last update, the {resource} {failure}.",
]

ACCESS_OBJECTS = [
    "my student account", "my teacher account", "my parent account",
    "my profile", "my school account", "my login", "the registration form",
    "my verification email", "the password reset page", "my username",
    "my account settings", "my email address", "my authentication code",
]
ACCESS_ACTIONS = [
    "sign in to", "recover", "register", "verify", "update", "reset the password for",
    "activate", "switch", "unlock", "change the email on", "confirm access to",
]
ACCESS_PATTERNS = [
    "How can I {action} {object}?",
    "I need help to {action} {object}.",
    "I can't {action} {object}; what should I do?",
    "Can you explain how to {action} {object}?",
    "Please help me {action} {object} for the learning platform.",
    "What are the steps to {action} {object}?",
    "The system will not let me {action} {object}.",
    "I keep getting rejected when I try to {action} {object}.",
]
ACCESS_SEEDS = [
    "I forgot my password and can't log in.",
    "My password is correct but I still cannot access my account.",
    "The registration confirmation link has expired.",
    "Where do I change the email address associated with my account?",
    "I didn't receive the verification code to create my account.",
    "I am locked out after too many login attempts.",
    "I have to reset my teacher portal password.",
    "How do I register a parent account?",
    "I cannot sign up with my school email.",
    "How do I recover my username?",
    "My account was suspended and I need to regain access.",
    "I can't complete the sign-up process.",
]
OTHER_TOPICS = [
    "subscription cost", "monthly price", "annual price", "discount for schools",
    "invoice", "payment methods", "refund policy", "cancellation policy",
    "billing cycle", "VAT invoice", "company contact number", "office opening hours",
    "customer support email", "school license pricing", "receipt",
    "delivery time for printed textbooks", "bulk purchasing", "payment confirmation",
    "order status", "terms of service", "privacy policy", "trial plan", "gift voucher",
    "renewal fees", "VAT number", "company address", "support hours",
    "accepted currencies", "bank transfer", "order cancellation",
]
OTHER_PATTERNS = [
    "Could you explain your {topic}?",
    "I have a question about the {topic}.",
    "Where can I find information about the {topic}?",
    "Please tell me more about the {topic}.",
    "How does the {topic} work?",
    "Do you have details about the {topic}?",
    "Can I speak to someone regarding the {topic}?",
    "Is there a page describing the {topic}?",
    "I need clarification about the {topic} for our school.",
    "Please send information on the {topic} to me.",
]
OTHER_SEEDS = [
    "Can you provide a VAT invoice for my order?",
    "How much is the annual subscription for a school?",
    "Which payment options do you accept?",
    "When will my printed textbook order be delivered?",
    "Is there a phone number for customer support?",
    "What hours is your support team available?",
    "Can I cancel my subscription and get a refund?",
    "Does your company offer discounts for teachers?",
    "Where are your business offices located?",
    "How can I request a quote for a school-wide license?",
]

# Human-reviewed priority contrasts (paired groups; not intended as benchmark data).
CONTRASTS = [
    ("Where can I find the mathematics video lesson?", CONTENT, "video-math"),
    ("The mathematics video lesson will not load.", TECH, "video-math"),
    ("Is there a grade seven interactive quiz?", CONTENT, "grade7-quiz"),
    ("The grade seven interactive quiz shows a blank page.", TECH, "grade7-quiz"),
    ("Where can I download the chemistry worksheet?", CONTENT, "chem-worksheet"),
    ("Downloading the chemistry worksheet fails with an error.", TECH, "chem-worksheet"),
    ("I need a password reset link for my account.", ACCESS, "password-account"),
    ("The lesson page keeps crashing, even after I log in.", TECH, "password-account"),
    ("Where can I find the online textbook?", CONTENT, "online-book"),
    ("The online textbook freezes after I open it.", TECH, "online-book"),
    ("I need to update the email on my profile.", ACCESS, "email-profile"),
    ("How much does the learning platform subscription cost?", OTHER, "email-profile"),
]


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def record(text: str, label: str, source: str, family: str, kind: str) -> dict:
    return {"text": text.strip(), "label": label, "source": source,
            "family": family, "kind": kind, "language": "en"}


def fetch_hf(repo: str, split: str) -> pd.DataFrame:
    from datasets import load_dataset
    return load_dataset(repo, split=split).to_pandas()


def add_bitext(rows: list[dict], strict: bool) -> None:
    try:
        df = fetch_hf(BITEXT, "train")
        required = {"instruction", "intent"}
        if not required.issubset(df.columns):
            raise ValueError(f"Bitext missing columns: {required - set(df.columns)}")
        for item in df.itertuples(index=False):
            text = str(getattr(item, "instruction", "")).strip()
            intent = str(getattr(item, "intent", "")).strip()
            if not text or "{{" in text or "}}" in text:
                continue  # Placeholder-heavy e-commerce samples need manual adaptation.
            if intent in BITEXT_ACCESS:
                label = ACCESS
            elif intent in BITEXT_OTHER:
                label = OTHER
            else:
                continue
            rows.append(record(text, label, "bitext", f"bitext:{intent}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}", "hf"))
        LOG.info("Bitext ingested; current total %d", len(rows))
    except Exception as exc:
        if strict:
            raise
        LOG.warning("Bitext skipped: %s", exc)


def add_triage(rows: list[dict], strict: bool) -> None:
    try:
        df = fetch_hf(TRIAGE, "train")  # Never use source test split for training.
        if not {"text", "category"}.issubset(df.columns):
            raise ValueError("TriageIQ schema changed")
        for item in df.itertuples(index=False):
            label = TRIAGE_LABELS.get(str(item.category).lower().strip())
            text = str(item.text).strip()
            if label and text:
                rows.append(record(text, label, "triageiq-train",
                                   f"triage:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}", "hf"))
        LOG.info("TriageIQ train ingested; current total %d", len(rows))
    except Exception as exc:
        if strict:
            raise
        LOG.warning("TriageIQ skipped: %s", exc)


def synthetic_rows(seed: int, max_candidates: int) -> list[dict]:
    """Controlled templates, count bounded; reproducible using local seeded RNG."""
    rng = random.Random(seed)
    out: list[dict] = []
    for i, p in enumerate(CONTENT_PATTERNS):
        for subject in SUBJECTS:
            for resource in RESOURCES:
                stage = rng.choice(STAGES)
                text = p.format(subject=subject, resource=resource, stage=stage)
                out.append(record(text, CONTENT, "synthetic-content", f"content:{i}", "template"))
    for i, p in enumerate(TECH_PATTERNS):
        for subject in SUBJECTS:
            for resource in RESOURCES:
                failure = rng.choice(TECH_FAILURES)
                platform = rng.choice(PLATFORMS)
                stage = rng.choice(STAGES)
                out.append(record(p.format(subject=subject, resource=resource,
                                           failure=failure, platform=platform, stage=stage),
                                  TECH, "synthetic-tech", f"tech:{i}", "template"))
    for i, p in enumerate(ACCESS_PATTERNS):
        for action in ACCESS_ACTIONS:
            for obj in ACCESS_OBJECTS:
                if action == "register" and obj not in {"my student account", "my teacher account", "my parent account", "my school account", "the registration form"}:
                    continue
                text = p.format(action=action, object=obj)
                out.append(record(text, ACCESS, "synthetic-access", f"access:{i}", "template"))
    # Contextual variants, ensuring access requests remain about account access.
    access_contexts = [
        "from my laptop", "using the mobile app", "on the school portal",
        "after changing devices", "since yesterday", "with my school email",
        "as a parent", "as a teacher", "as a student", "during registration",
        "without contacting support", "after the recent update",
    ]
    access_context_patterns = [
        "Could you help me {action} {object} {context}?",
        "Is it possible to {action} {object} {context}?",
        "I'm trying to {action} {object} {context}; please advise.",
        "What should I do to {action} {object} {context}?",
    ]
    for i, p in enumerate(access_context_patterns):
        for action in ACCESS_ACTIONS:
            for obj in ACCESS_OBJECTS:
                for context in access_contexts:
                    # Exclude awkward combinations such as 'register my username'.
                    if action == "register" and obj not in {"my student account", "my teacher account", "my parent account", "my school account", "the registration form"}:
                        continue
                    out.append(record(p.format(action=action, object=obj, context=context),
                                      ACCESS, "synthetic-access", f"access-context:{i}", "template"))
    for text in ACCESS_SEEDS:
        out.append(record(text, ACCESS, "seed", "access:seed", "seed"))
    for i, p in enumerate(OTHER_PATTERNS):
        for topic in OTHER_TOPICS:
            text = p.format(topic=topic)
            out.append(record(text, OTHER, "synthetic-other", f"other:{i}", "template"))
    # Administrative questions across organizations and styles; still OTHER.
    other_contexts = [
        "for an individual student", "for parents", "for a public school",
        "for a private school", "for a teacher", "for a school administrator",
        "for a small class", "for a large school", "for a nonprofit organization",
        "for multiple children", "for next semester", "for the next academic year",
        "for international customers", "for online purchases", "for a bulk order",
    ]
    other_context_patterns = [
        "Could you clarify the {topic} {context}?",
        "Who should I contact about the {topic} {context}?",
        "I would like information on the {topic} {context}.",
        "Please explain the {topic} {context}.",
        "Can you send me the details of the {topic} {context}?",
        "Where can I compare the {topic} {context}?",
        "How can I check the {topic} {context}?",
        "What do I need to know about the {topic} {context}?",
    ]
    for i, p in enumerate(other_context_patterns):
        for topic in OTHER_TOPICS:
            for context in other_contexts:
                out.append(record(p.format(topic=topic, context=context),
                                  OTHER, "synthetic-other", f"other-context:{i}", "template"))
    for text in OTHER_SEEDS:
        out.append(record(text, OTHER, "seed", "other:seed", "seed"))
    for text, label, group in CONTRASTS:
        out.append(record(text, label, "contrast", f"contrast:{group}", "contrast"))
    rng.shuffle(out)
    return out[:max_candidates] if max_candidates else out


def add_existing(rows: list[dict], csv_path: Path | None, strict: bool) -> None:
    if csv_path is None:
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Local CSV needs text,label")
    # English data only; machine-translated Bulgarian requires separate audited stage.
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower().eq("en")]
    else:
        # Reject Cyrillic: the final requested training language is English.
        df = df[~df["text"].astype(str).str.contains(r"[А-Яа-я]", regex=True)]
    for item in df.itertuples(index=False):
        text, label = str(item.text).strip(), str(item.label).strip()
        if label in LABELS and text:
            rows.append(record(text, label, "local-english", f"local:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}", "local"))


def select_balanced(rows: list[dict], target_per_label: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No candidate records produced")
    frame["norm"] = frame["text"].map(normalize)
    frame = frame[frame["norm"].str.len().ge(8)].copy()
    frame = frame[~frame["text"].str.contains(r"\{\{|\}\}", regex=True)]
    # If same phrase maps to multiple classes, quarantine all conflicting rows.
    conflicts = frame.groupby("norm")["label"].nunique()
    bad = set(conflicts.index[conflicts.gt(1)])
    quarantined = frame[frame["norm"].isin(bad)].copy()
    frame = frame[~frame["norm"].isin(bad)].copy()
    # Prefer curated/hf over templates when same text occurs more than once.
    priority = {"contrast": 0, "seed": 1, "hf": 2, "local": 3, "template": 4}
    frame["priority"] = frame["kind"].map(priority).fillna(5)
    frame = frame.sort_values("priority").drop_duplicates("norm").copy()
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        group = frame[frame["label"].eq(label)]
        # Prioritize diverse natural examples. Synthetics backfill shortfalls.
        natural = group[group["kind"].ne("template")]
        synthetic = group[group["kind"].eq("template")]
        chosen = natural.sample(n=min(target_per_label, len(natural)), random_state=seed)
        deficit = target_per_label - len(chosen)
        if deficit > 0:
            chosen = pd.concat([chosen, synthetic.sample(n=min(deficit, len(synthetic)), random_state=seed)])
        if len(chosen) < target_per_label:
            LOG.warning("%s: only %d/%d unique records", label, len(chosen), target_per_label)
        selected.append(chosen)
    output = pd.concat(selected, ignore_index=True)
    output = output.sample(frac=1, random_state=seed).reset_index(drop=True)
    return output, quarantined


def main() -> None:
    p = argparse.ArgumentParser(description="Build English training candidates with Bulgarian business labels")
    p.add_argument("--output", type=Path, default=Path("data/english_train_candidates.csv"))
    p.add_argument("--target-per-class", type=int, default=3500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline", action="store_true", help="Templates only: no HF download")
    p.add_argument("--strict", action="store_true", help="Fail if HF downloads fail")
    p.add_argument("--local-english", type=Path, default=None)
    p.add_argument("--max-synthetic-candidates", type=int, default=0)
    args = p.parse_args()
    if args.target_per_class < 1:
        p.error("--target-per-class must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = []
    if not args.offline:
        add_bitext(rows, args.strict)
        add_triage(rows, args.strict)
    add_existing(rows, args.local_english, args.strict)
    rows.extend(synthetic_rows(args.seed, args.max_synthetic_candidates))
    selected, conflicts = select_balanced(rows, args.target_per_class, args.seed)
    dest = args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    selected[["text", "label"]].to_csv(dest, index=False, encoding="utf-8-sig")
    selected[["text", "label", "source", "family", "kind", "language"]].to_csv(
        dest.with_name(dest.stem + "_provenance.csv"), index=False, encoding="utf-8-sig")
    if not conflicts.empty:
        conflicts.drop(columns=["priority"], errors="ignore").to_csv(
            dest.with_name(dest.stem + "_conflicts.csv"), index=False, encoding="utf-8-sig")
    report = {
        "target_per_class": args.target_per_class,
        "total": len(selected),
        "classes": {k: int(v) for k, v in selected["label"].value_counts().items()},
        "by_source": {k: int(v) for k, v in selected["source"].value_counts().items()},
        "by_kind": {k: int(v) for k, v in selected["kind"].value_counts().items()},
        "quarantined_conflicting_rows": len(conflicts),
        "language": "en",
        "warning": "Template-generated records are NOT human-validated. Hold out genuine user queries separately.",
    }
    dest.with_name(dest.stem + "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {dest}")


if __name__ == "__main__":
    main()
