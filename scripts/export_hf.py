"""Add Transformers loaders to an existing package without changing its weights."""

import argparse
import shutil
from pathlib import Path

from chandassu.hub import export_hf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hub_dir", type=Path, help="Stage the shared Hub code and root model card")
    args = parser.parse_args(argv)
    output = export_hf(args.model_dir, args.output_dir)
    project = Path(__file__).resolve().parents[1]
    for name in ("LICENSE", "NOTICE", "CITATION.cff"):
        shutil.copyfile(project / name, output / name)
    shutil.copytree(project / "LICENSES", output / "LICENSES", dirs_exist_ok=True)
    if args.hub_dir:
        args.hub_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "src/chandassu/hf_model.py", args.hub_dir / "hf_model.py")
        shutil.copyfile(project / "docs/huggingface/README.md", args.hub_dir / "README.md")
        for name in ("LICENSE", "NOTICE", "CITATION.cff"):
            shutil.copyfile(project / name, args.hub_dir / name)
        shutil.copytree(project / "LICENSES", args.hub_dir / "LICENSES", dirs_exist_ok=True)
    print(f"Transformers package: {output}")


if __name__ == "__main__":
    main()
