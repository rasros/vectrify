import os
import sys

from .cli import parse_args
from .search import run_search
from .storage import FileStorageAdapter


def main():
    args = parse_args()

    # Safety check for the primary dependency
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        raise SystemExit(1)

    # Initialize the storage adapter (Strategy-agnostic)
    storage = FileStorageAdapter(
        output_svg_path=args.output,
        write_lineage=args.write_lineage,
        resume=args.resume,
    )

    try:
        run_search(
            image_path=args.image,
            storage=storage,
            seed_svg_path=args.seed_svg,
            max_accepts=args.max_accepts,
            workers=args.workers,
            base_model_temperature=args.model_temp,
            openai_image_long_side=args.openai_image_long_side,
            max_wall_seconds=args.max_wall_seconds,
            log_level=args.log_level,
            scorer_type=args.scorer,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting safely...", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()