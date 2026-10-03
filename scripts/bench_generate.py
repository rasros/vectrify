"""Generate the benches' cartoon rasters with the OpenAI image API.

The images are cel-style drawings made for the trace and shadow benches
(bench_trace, bench_shadows). Their prompts, model, size, quality and set
(tuning or held-out) are in scripts/bench_data/generated.json; the images
themselves are not in the repository but saved to
~/.cache/vectrify-bench/generated. Images already there are kept, so a
run only makes the missing ones; `--force` makes them again. The model is
not deterministic, so a regenerated image differs from the one the
baselines were run on.

The openai package is not one of Vectrify's dependencies:

    uv run --no-project --with openai python scripts/bench_generate.py
    uv run --no-project --with openai python scripts/bench_generate.py \\
        --env ~/path/to/.env --only gen-fox-forest
    uv run --no-project --with openai python scripts/bench_generate.py --models

The key is OPENAI_API_KEY from the environment, or from the .env file
`--env` names; it is only put in this process's environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

DATA = Path(__file__).resolve().parent / "bench_data" / "generated.json"
FOLDER = Path.home() / ".cache" / "vectrify-bench" / "generated"


def load_env(path: Path) -> None:
    """Put OPENAI_API_KEY from the .env file *path* in this process's
    environment, unless it is already set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    for line in path.expanduser().read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = value.strip().strip("'\"")
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", type=Path, help="A .env file with OPENAI_API_KEY")
    parser.add_argument("--only", nargs="+", help="Make only these images (stems)")
    parser.add_argument("--force", action="store_true", help="Remake existing ones")
    parser.add_argument(
        "--max-calls", type=int, default=18, help="Stop after this many API calls"
    )
    parser.add_argument(
        "--models", action="store_true", help="List the image models offered"
    )
    args = parser.parse_args()
    if args.env:
        load_env(args.env)
    from openai import OpenAI  # pyrefly: ignore[missing-import]

    client = OpenAI()
    if args.models:
        for model in sorted(m.id for m in client.models.list()):
            if "image" in model or "dall" in model:
                print(model)
        return
    data = json.loads(DATA.read_text())
    FOLDER.mkdir(parents=True, exist_ok=True)
    calls = 0
    for item in data["images"]:
        stem = Path(item["file"]).stem
        path = FOLDER / item["file"]
        if args.only and stem not in args.only:
            continue
        if path.exists() and not args.force:
            continue
        if calls >= args.max_calls:
            print(f"stopping: {args.max_calls} calls made")
            break
        calls += 1
        prompt = f"{item['prompt']} {data['style']}"
        try:
            result = client.images.generate(
                model=item.get("model", data["model"]),
                prompt=prompt,
                size=item.get("size", data["size"]),
                quality=data["quality"],
                n=1,
            )
        except Exception as error:  # report it and go on
            print(f"{stem}: failed: {type(error).__name__}: {error}")
            continue
        image = (result.data or [None])[0]
        if image is None or not image.b64_json:
            print(f"{stem}: no image returned")
            continue
        path.write_bytes(base64.b64decode(image.b64_json))
        print(f"{stem}: saved {path}", flush=True)
    print(f"{calls} API calls")


if __name__ == "__main__":
    main()
