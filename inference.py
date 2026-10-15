"""Serve a local or immutable Hugging Face model package."""

import argparse
import json
from pathlib import Path

import uvicorn
from huggingface_hub import hf_hub_download, snapshot_download

from chandassu.inference.server import create_app
from chandassu.training.runtime import select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model_dir", default=None)
    source.add_argument("--model_dirs", nargs="+", help="Distinct seed packages from one model family")
    source.add_argument("--model_id", help="Hugging Face repository ID")
    parser.add_argument("--revision", help="Immutable 40-character Hub commit ID")
    parser.add_argument("--subfolder", help="Optional Hub package, e.g. cnn-eight-class, cnn or byt5")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument(
        "--strategy", choices=["single", "ensemble"], default="single", help="Initial UI/API prediction strategy"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("cpu_threads must be positive")
    if args.model_id:
        if not args.revision or len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
            parser.error("Hub models require an immutable lowercase hexadecimal --revision")
        subfolder = args.subfolder.strip("/") if args.subfolder else ""
        if subfolder and (Path(subfolder).name != subfolder or subfolder in {".", ".."}):
            parser.error("--subfolder must be a single model package directory name")
        manifest_file = hf_hub_download(
            args.model_id, "manifest.json", revision=args.revision, subfolder=subfolder or None
        )
        manifest = json.loads(Path(manifest_file).read_text())
        names = list(manifest["files"]) + ["manifest.json"]
        if any(Path(name).name != name for name in names):
            parser.error("Hub manifest must list files inside its model package")
        prefix = subfolder + "/" if subfolder else ""
        snapshot = snapshot_download(
            args.model_id, revision=args.revision, allow_patterns=[prefix + name for name in names]
        )
        folder = str(Path(snapshot) / subfolder) if subfolder else snapshot
    else:
        if args.subfolder:
            parser.error("--subfolder requires --model_id")
        folder = args.model_dirs or args.model_dir or "models/cnn"
    app = create_app(folder, select_device(args.device), args.cpu_threads, args.strategy)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
