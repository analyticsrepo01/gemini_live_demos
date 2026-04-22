#!/usr/bin/env python3
"""
Nano Banana — Gemini 3.1 Flash Image Preview generator
Generates physics diagrams and illustrations for IE Irodov problems.

Usage:
  python nano_banana.py "prompt text" [--out filename.png] [--ratio 16:9] [--dark] [--n 1] [--problem 1.1]
"""

import os
import sys
import argparse
import re
from pathlib import Path
from io import BytesIO

# ── Vertex AI / GenAI setup ──────────────────────────────────────────────────
os.environ.setdefault("GOOGLE_CLOUD_PROJECT",  "my-project-0004-346516")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

MODEL = "gemini-3.1-flash-image-preview"

# Aspect ratio map: user-friendly → API value
RATIO_MAP = {
    "1:1":  "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3":  "4:3",
    "3:4":  "3:4",
    "3:2":  "3:2",
    "2:3":  "2:3",
    "4:5":  "4:5",
    "5:4":  "5:4",
    "21:9": "21:9",
}


def build_prompt(description: str, problem_id: str = "", dark: bool = False) -> str:
    """Enhance a bare description into a high-quality technical diagram prompt."""
    style = (
        "dark background (#0d1117), glowing colored lines, neon accent colors, "
        "professional tech infographic style"
        if dark else
        "clean white background, crisp lines, professional tech blog / developer documentation style, "
        "high contrast, clear typography"
    )
    return (
        f"Technical diagram — {description}. "
        f"Style: {style}. "
        f"Include clear labels, directional arrows, and annotations where relevant. "
        f"Professional illustration quality, no photorealistic elements, vector art style. "
        f"All text labels in English."
    )


def slug(text: str) -> str:
    """Turn a prompt into a safe filename slug."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return s[:40].strip("_")


def generate_image(
    prompt: str,
    out_path: str,
    ratio: str = "16:9",
    n: int = 1,
    problem_id: str = "",
    dark: bool = False,
) -> list[str]:
    """
    Generate image(s) using Gemini 3.1 Flash Image Preview.
    Returns list of saved file paths.
    """
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig, Modality
    except ImportError:
        print("Installing google-genai...")
        os.system("pip install google-genai -q")
        from google import genai
        from google.genai.types import GenerateContentConfig, Modality

    client = genai.Client()
    full_prompt = build_prompt(prompt, problem_id, dark)

    print(f"  [Nano Banana] Model : {MODEL}")
    print(f"  [Nano Banana] Ratio : {ratio}")
    print(f"  [Nano Banana] Count : {n}")
    print(f"  [Nano Banana] Prompt: {full_prompt[:120]}...")

    saved_paths = []

    for i in range(n):
        print(f"  [Nano Banana] Generating image {i+1}/{n}...")
        response = client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
            config=GenerateContentConfig(
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            ),
        )

        # Extract first image part from response (model may also return reasoning text)
        image_saved = False
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data and not image_saved:
                img_bytes = part.inline_data.data
                mime      = part.inline_data.mime_type or "image/png"
                ext       = mime.split("/")[-1].replace("jpeg", "jpg")

                # Determine output path
                if n == 1:
                    save_path = out_path
                else:
                    base = Path(out_path)
                    save_path = str(base.parent / f"{base.stem}_{i+1}{base.suffix}")

                if not Path(save_path).suffix:
                    save_path += f".{ext}"

                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                Path(save_path).write_bytes(img_bytes)
                saved_paths.append(save_path)
                image_saved = True
                print(f"  [Nano Banana] Saved : {save_path}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Nano Banana — Gemini 3.1 Flash Image Generator")
    parser.add_argument("prompt", nargs="?", default="", help="Image description")
    parser.add_argument("--out",     default="",    help="Output file path (default: auto)")
    parser.add_argument("--ratio",   default="16:9", choices=list(RATIO_MAP.keys()), help="Aspect ratio")
    parser.add_argument("--dark",    action="store_true", help="Dark background style")
    parser.add_argument("--n",       type=int, default=1,  help="Number of images (1-4)")
    parser.add_argument("--problem", default="",    help="Irodov problem ID e.g. 1.1")
    args = parser.parse_args()

    prompt = args.prompt.strip()
    if not prompt:
        parser.print_help()
        sys.exit(1)

    # Auto-generate output path
    out_path = args.out
    if not out_path:
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        out_path = str(plots_dir / f"{slug(prompt)}.png")

    ratio = RATIO_MAP.get(args.ratio, "16:9")
    n     = max(1, min(4, args.n))

    print(f"\n{'─'*55}")
    print(f"  Nano Banana  🍌  Gemini 3.1 Flash Image Preview")
    print(f"{'─'*55}")

    saved = generate_image(
        prompt=prompt,
        out_path=out_path,
        ratio=ratio,
        n=n,
        problem_id=args.problem,
        dark=args.dark,
    )

    print(f"\n{'─'*55}")
    print(f"  Done! {len(saved)} image(s) saved:")
    for p in saved:
        print(f"  → {p}")
    print(f"{'─'*55}\n")

    return saved


# ── Library API (for use from other scripts) ─────────────────────────────────

def generate_for_problem(problem_id: str, description: str, dark: bool = True) -> str:
    """
    Convenience function: generate one diagram for an Irodov problem.
    Returns the saved file path.
    """
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = str(plots_dir / f"nb_p{problem_id.replace('.','_')}_{slug(description)}.png")

    saved = generate_image(
        prompt=description,
        out_path=out_path,
        ratio="16:9",
        n=1,
        problem_id=problem_id,
        dark=dark,
    )
    return saved[0] if saved else ""


if __name__ == "__main__":
    main()
