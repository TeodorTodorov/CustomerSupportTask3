"""CLI entry point; prints machine-readable JSON for inference and evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import TrainingSettings
from .evaluation import evaluate
from .inference import QueryClassifier
from .rules import RuleClassifier
from .training import train
from .splitting import create_splits
from .leakage import GroupingSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bulgarian support query classifier")
    sub = parser.add_subparsers(dest="command", required=True)

    trainer = sub.add_parser("train", help="Fine-tune an XLM-R classifier")
    trainer.add_argument("--split-dir", type=Path, required=True, help="Audited splits from split-data")
    trainer.add_argument("--output", type=Path, default=Path("models/classifier"))
    trainer.add_argument("--epochs", type=int, default=5)
    trainer.add_argument("--batch-size", type=int, default=8)
    trainer.add_argument("--learning-rate", type=float, default=2e-5)
    splitter = sub.add_parser("split-data", help="Conservative, auditable train/val/test splitting")
    splitter.add_argument("--data", type=Path, required=True)
    splitter.add_argument("--output", type=Path, default=Path("data/splits"))
    splitter.add_argument("--semantic-threshold", type=float, default=0.91)
    splitter.add_argument("--near-threshold", type=float, default=0.88)
    splitter.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    splitter.add_argument("--seed", type=int, default=42)
    splitter.add_argument("--skip-semantic", action="store_true",
                          help="Skip slow semantic ANN (keeps provenance and lexical checks)")
    splitter.add_argument("--provenance-only", action="store_true",
                          help="Skip both similarity checks; keep pair/family/exact grouping ONLY")
    splitter.add_argument("--approved-review", type=Path, help="Human-approved contrast pairs JSON")

    evaluator = sub.add_parser("evaluate", help="Evaluate on independent test.csv")
    evaluator.add_argument("--data", type=Path, required=True)
    evaluator.add_argument("--model", type=Path, default=Path("models/classifier"))
    evaluator.add_argument("--report", type=Path)

    predictor = sub.add_parser("predict", help="Return JSON for a single query")
    predictor.add_argument("--text", required=True)
    predictor.add_argument("--model", type=Path, default=Path("models/classifier"))
    predictor.add_argument("--backend", choices=["model", "rules"], default="model")

    interactive = sub.add_parser("interactive", help="Interactive terminal mode")
    interactive.add_argument("--model", type=Path, default=Path("models/classifier"))
    interactive.add_argument("--backend", choices=["model", "rules"], default="model")

    server = sub.add_parser("serve", help="Start HTTP API")
    server.add_argument("--model", type=Path, default=Path("models/classifier"))
    server.add_argument("--backend", choices=["model", "rules"], default="model")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        if args.command == "split-data":
            if args.provenance_only:
                logging.warning("PROVENANCE-ONLY: lexical/semantic paraphrases may leak across splits")
            elif args.skip_semantic:
                logging.warning("Semantic similarity audit DISABLED: paraphrases may leak across splits")
            opts = {}
            if args.provenance_only:
                opts["near_fn"] = lambda texts, threshold: []
            if args.provenance_only or args.skip_semantic:
                opts["semantic_fn"] = lambda texts, model_name, threshold, k: []
            result = create_splits(args.data, args.output, seed=args.seed,
                                   settings=GroupingSettings(near_threshold=args.near_threshold,
                                                             semantic_threshold=args.semantic_threshold,
                                                             semantic_model=args.semantic_model),
                                   approved_review=args.approved_review,
                                   checks={"lexical": not args.provenance_only,
                                           "semantic": not (args.skip_semantic or args.provenance_only)},
                                   **opts)
            print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
        elif args.command == "train":
            result = train(
                args.split_dir,
                TrainingSettings(
                    output_dir=args.output,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                ),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            print(json.dumps(evaluate(args.model, args.data, args.report), ensure_ascii=False, indent=2))
        elif args.command == "predict":
            classifier = RuleClassifier() if args.backend == "rules" else QueryClassifier(args.model)
            print(json.dumps(classifier.predict(args.text).to_dict(), ensure_ascii=False, indent=2))
        elif args.command == "interactive":
            classifier = RuleClassifier() if args.backend == "rules" else QueryClassifier(args.model)
            while True:
                try:
                    text = input("Запитване (exit за край): ")
                except (KeyboardInterrupt, EOFError):
                    print()
                    break
                if text.strip().lower() in {"exit", "quit"}:
                    break
                try:
                    print(json.dumps(classifier.predict(text).to_dict(), ensure_ascii=False, indent=2))
                except ValueError as exc:
                    print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        elif args.command == "serve":
            import os
            import uvicorn

            from .security import SecuritySettings
            from .api import create_app
            security = SecuritySettings.from_environment()
            if args.host not in {"127.0.0.1", "::1", "localhost"}:
                if os.getenv("KLETT_ALLOW_REMOTE_BIND") != "true" or not os.getenv("KLETT_ALLOWED_HOSTS"):
                    raise ValueError("Remote bind requires KLETT_ALLOW_REMOTE_BIND=true, explicit KLETT_ALLOWED_HOSTS and TLS proxy")
            app = create_app(security=security, backend=args.backend, model_dir=args.model)
            uvicorn.run(app, host=args.host, port=args.port, workers=1, proxy_headers=False)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
