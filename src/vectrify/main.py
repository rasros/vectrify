import logging
import os
import sys
import traceback
from pathlib import Path

from vectrify.cli import parse_args
from vectrify.dashboard import Dashboard
from vectrify.formats import get_plugin
from vectrify.llm.models import DEFAULT_MODELS, PROVIDERS, api_key_env
from vectrify.search.base import StrategyType
from vectrify.search.stats import SearchStats
from vectrify.utils import setup_logger
from vectrify.vector.runner import run_vector_search
from vectrify.vector.storage import FileStorageAdapter


def determine_provider_and_model(args) -> tuple[str, str]:
    provider = args.provider
    model = args.model

    if provider == "auto":
        provider = next(
            (p for p in PROVIDERS if os.getenv(api_key_env(p))),
            None,
        )
        if provider is None:
            env_vars = ", ".join(api_key_env(p) for p in PROVIDERS)
            print(
                f"Error: no LLM API key found. Set one of {env_vars} "
                "in your environment.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        env_var = api_key_env(provider)
        if not os.getenv(env_var):
            print(
                f"Error: --provider {provider} was selected but {env_var} is "
                f"not set. Export {env_var} or pick a provider whose key is set.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not model:
        model = DEFAULT_MODELS[provider]
    return provider, model


def _fail(message: str, debug: bool) -> None:
    """Print a fatal error and exit; show the full traceback when debug is set.

    Must be called from within an ``except`` block so the active exception is
    available to ``traceback.print_exc()``.
    """
    print(message, file=sys.stderr)
    if debug:
        traceback.print_exc()
    else:
        print("(run with --debug for the full traceback)", file=sys.stderr)
    sys.exit(1)


def format_extension_warning(
    output_path: str, fmt: str, expected_ext: str
) -> str | None:
    """Return a warning if the output extension doesn't match --format, else None."""
    actual = Path(output_path).suffix.lower()
    if actual == expected_ext:
        return None
    return (
        f"Output path '{output_path}' has extension '{actual or '(none)'}' but "
        f"--format {fmt} produces '{expected_ext}' files; writing it anyway."
    )


def main():
    args = parse_args()
    provider, model = determine_provider_and_model(args)

    setup_logger(args.log_level)
    logger = logging.getLogger("main")
    logger.debug("=== Vectrify parameters ===")
    logger.debug(f"  provider: {provider} | model: {model}")
    for key, val in sorted(vars(args).items()):
        logger.debug(f"  {key}: {val}")
    logger.debug("==========================")

    plugin = get_plugin(args.format)

    mismatch = format_extension_warning(args.output, args.format, plugin.file_extension)
    if mismatch:
        logger.warning(mismatch)

    stats = SearchStats(
        strategy_name=args.strategy,
        model_name=model,
        epoch_patience=args.epoch_patience or 0,
    )

    storage = FileStorageAdapter(
        output_path=args.output,
        file_extension=plugin.file_extension,
        resume=args.resume,
        resume_top=args.resume_top,
        save_raster=args.save_raster,
        save_heatmap=args.save_heatmap,
        write_lineage=args.write_lineage,
    )

    use_dashboard = args.dashboard and sys.stdout.isatty()
    if args.dashboard and not use_dashboard:
        logger.info("stdout is not a terminal; disabling the live dashboard.")
    dashboard = Dashboard(stats) if use_dashboard else None

    try:
        run_vector_search(
            image_path=args.image,
            storage=storage,
            workers=args.workers,
            resolution=args.resolution,
            resolution_llm=args.resolution_llm,
            max_wall_seconds=args.max_wall_seconds,
            log_level=args.log_level,
            scorer_type=args.scorer,
            strategy_type=StrategyType(args.strategy),
            goal=args.goal,
            llm_provider=provider,
            llm_model=model,
            reasoning=args.reasoning,
            format_plugin=plugin,
            write_lineage=args.write_lineage,
            save_raster=args.save_raster,
            epoch_patience=args.epoch_patience or None,
            epoch_min_delta=args.epoch_min_delta,
            llm_rate=args.llm_rate,
            pool_size=args.pool_size,
            seeds=args.seeds,
            beams=args.beams,
            cull_keep=args.cull_keep,
            epoch_diversity=args.epoch_diversity,
            tournament_size=args.tournament_size,
            epoch_variance=args.epoch_variance or None,
            max_epochs=args.max_epochs,
            epoch_pool_size=args.epoch_seeds or None,
            epoch_steps=args.epoch_steps or None,
            max_llm_calls=args.max_llm_calls or None,
            max_total_tasks=args.max_total_tasks,
            vision_model=args.vision_model,
            stats=stats,
            dashboard=dashboard,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting safely...", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError:
        _fail(f"Error: input image not found: {args.image}", args.debug)
    except Exception as e:
        _fail(f"Error: {e}", args.debug)


if __name__ == "__main__":
    main()
