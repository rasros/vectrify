import os
import sys

from .cli import parse_args
from .search import run_search


def main():
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        raise SystemExit(1)

    run_search(
        image_path=args.image,
        output_svg_path=args.output,
        max_accepts=args.max_accepts,
        proposals_per_step=args.proposals_per_step,
        base_model_temperature=args.model_temp,
        workers=args.workers,
        log_level=args.log_level,
        top_k=args.top_k,
        write_top_k_each=args.write_top_k_each,
        max_total_tasks=args.max_total_tasks,
        max_wall_seconds=args.max_wall_seconds,
        resume=args.resume,
        anneal_t0=args.anneal_t0,
        anneal_alpha=args.anneal_alpha,
        anneal_min_t=args.anneal_min_t,
        propose_from_best_prob=args.propose_from_best_prob,
    )


if __name__ == "__main__":
    main()
