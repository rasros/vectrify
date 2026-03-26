import logging
import os
import sys

from svgizer.cli import parse_args
from svgizer.search.base import StrategyType
from svgizer.svg.runner import run_svg_search
from svgizer.svg.storage import FileStorageAdapter


def determine_provider_and_model(args) -> tuple[str, str]:
    provider = args.provider
    model = args.model
    default_models = {
        "openai": "gpt-5.4",
        "anthropic": "claude-4-6-sonnet",
        "gemini": "gemini-3.1-pro-preview",
    }

    if provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            print("CRITICAL: No API key found.", file=sys.stderr)
            sys.exit(1)
    else:
        env_var = f"{provider.upper()}_API_KEY"
        if not os.getenv(env_var):
            print(f"CRITICAL: {env_var} not set.", file=sys.stderr)
            sys.exit(1)

    if not model:
        model = default_models.get(provider, "gpt-5.4")
    return provider, model


def main():
    args = parse_args()

    provider, model = determine_provider_and_model(args)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("main")
    logger.info("=== SVGizer parameters ===")
    logger.info(f"  provider: {provider} | model: {model}")
    for key, val in sorted(vars(args).items()):
        logger.info(f"  {key}: {val}")
    logger.info("==========================")

    storage = FileStorageAdapter(
        output_svg_path=args.output,
        resume=args.resume,
        openai_image_long_side=args.image_long_side,
    )

    try:
        run_svg_search(
            image_path=args.image,
            storage=storage,
            workers=args.workers,
            image_long_side=args.image_long_side,
            max_wall_seconds=args.max_wall_seconds,
            log_level=args.log_level,
            scorer_type=args.scorer,
            strategy_type=StrategyType(args.strategy),
            goal=args.goal,
            llm_provider=provider,
            llm_model=model,
            reasoning=args.reasoning,
            write_lineage=args.write_lineage,
            epoch_patience=args.epoch_patience or None,
            min_delta=args.min_delta,
            llm_rate=args.llm_rate,
            pool_size=args.pool_size,
            seed_tasks=args.seed_tasks,
            similarity_threshold=args.similarity_threshold,
            min_diversity=args.min_diversity,
            max_epochs=args.max_epochs or None,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting safely...", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
