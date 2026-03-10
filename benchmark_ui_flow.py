import argparse
from pathlib import Path
from time import perf_counter

from chess_fen.service import _load_predictor, analyze_image


def _default_image_paths(root_dir):
    patterns = (
        "*180406*.png",
        "*194748*.png",
        "*160332*.png",
    )
    paths = []
    for pattern in patterns:
        matches = sorted(root_dir.glob(pattern))
        if matches:
            paths.append(matches[0])
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the warm predictor on the local chess screenshots."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional image paths. Defaults to the three local reference screenshots.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    image_paths = [Path(image_path) for image_path in args.images] or _default_image_paths(root_dir)
    if not image_paths:
        raise SystemExit("No benchmark images found.")

    predictor = _load_predictor()
    print("predictor_loaded=true")

    for image_path in image_paths:
        start = perf_counter()
        result = analyze_image(str(image_path), predictor=predictor, use_filters=True)
        elapsed = perf_counter() - start
        print(f"image={image_path.name}")
        print(f"elapsed_s={elapsed:.3f}")
        print(f"fen={result['fen']}")
        print(f"reliable={result['is_reliable']}")
        print(f"quality={result['quality_score']}")
        print(f"confidence={result['avg_confidence']:.6f}")
        print(f"stability={result['stability_count']}")
        print(f"phase={result.get('phase_name') or '-'}")
        print("---")


if __name__ == "__main__":
    main()
