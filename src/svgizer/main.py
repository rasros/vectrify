import logging
import os
import sys

from svgizer.cli import parse_args
from svgizer.dashboard import Dashboard
from svgizer.search.base import StrategyType
from svgizer.search.stats import SearchStats
from svgizer.svg.runner import run_svg_search
from svgizer.svg.storage import FileStorageAdapter
from svgizer.utils import setup_logger


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

    setup_logger(args.log_level)
    logger = logging.getLogger("main")
    logger.debug("=== SVGizer parameters ===")
    logger.debug(f"  provider: {provider} | model: {model}")
    for key, val in sorted(vars(args).items()):
        logger.debug(f"  {key}: {val}")
    logger.debug("==========================")

    stats = SearchStats(
        strategy_name=args.strategy,
        model_name=model,
        epoch_patience=args.epoch_patience or 0,
    )

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
            epoch_min_delta=args.epoch_min_delta,
            llm_rate=args.llm_rate,
            pool_size=args.pool_size,
            seeds=args.seeds,
            epoch_diversity=args.epoch_diversity,
            max_epochs=None if args.max_epochs < 0 else max(1, args.max_epochs),
            epoch_pool_size=args.epoch_seeds or None,
            vision_model=args.vision_model,
            stats=stats,
            dashboard=Dashboard(stats),
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting safely...", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
