"""Package only the received final weights and the existing evaluation data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {"seed_42_model/" + name: args.model / name for name in (
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")}
    files.update({"data/" + name: args.data / name for name in (
        "evaluation_inputs.csv", "evaluation_labels.csv", "phase2_inputs.csv", "selection_manifest.csv")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in files.items():
            payload = path.read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 30, 0, 0, 0))
            archive.writestr(entry, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    metadata = {"archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "expected_files": hashes, "parts": []}
    with args.output.open("rb") as handle:
        index = 1
        while payload := handle.read(90 * 1024 * 1024):
            part = args.output.with_name(args.output.name + f".part{index}")
            part.write_bytes(payload)
            metadata["parts"].append({"name": part.name, "sha256": hashlib.sha256(payload).hexdigest(),
                                      "size": len(payload), "file_id": None})
            index += 1
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)
    print(args.output.stat().st_size, metadata["archive_sha256"])


if __name__ == "__main__":
    main()
