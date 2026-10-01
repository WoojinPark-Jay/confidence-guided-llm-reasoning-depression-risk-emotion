"""Download a private, hash-pinned model/data bundle in Colab."""

import hashlib
import shutil
import zipfile
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(root, expected_files):
    return all((root / name).is_file() and file_sha256(root / name) == digest
               for name, digest in expected_files.items())


def extract_verified_assets(archive_path, root, archive_sha256, expected_files):
    if file_sha256(archive_path) != archive_sha256:
        raise ValueError("Asset download SHA256 mismatch; no files extracted.")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected_files):
            raise ValueError("Unexpected or duplicate files in the asset ZIP.")
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe asset ZIP path.")
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            with archive.open(name) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if file_sha256(temporary) != expected_files[name]:
                raise ValueError(f"Asset file hash mismatch: {name}")
            temporary.replace(target)
    if not verify_assets(root, expected_files):
        raise ValueError("Asset verification failed after extraction.")


def prepare_assets(drive_root, parts, archive_sha256, expected_files):
    root = drive_root / "confidence_guided_llm_reasoning/assets/final_seed42_existing9000_20260930"
    root.mkdir(parents=True, exist_ok=True)
    if verify_assets(root, expected_files):
        print("Verified saved final model and 9000-post data. No download needed.")
        return root / "data", root / "seed_42_model"
    archive_path = root / "final_seed42_existing9000_assets.zip"
    if not archive_path.is_file() or file_sha256(archive_path) != archive_sha256:
        from google.colab import auth
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        print("Authorize jay.seizethemoment@gmail.com to read the private asset file.")
        auth.authenticate_user()
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
        with AuthorizedSession(credentials) as session:
            for index, part in enumerate(parts, start=1):
                target = root / f"asset.part{index}"
                if target.is_file() and file_sha256(target) == part["sha256"]:
                    continue
                partial = target.with_suffix(target.suffix + ".partial")
                url = f"https://www.googleapis.com/drive/v3/files/{part['file_id']}"
                with session.get(url, params={"alt": "media"}, stream=True, timeout=180) as response:
                    if response.status_code in (401, 403, 404):
                        raise RuntimeError(
                            "Private asset access failed. Authenticate with jay.seizethemoment@gmail.com; "
                            "do not change the model or download an old checkpoint."
                        )
                    response.raise_for_status()
                    from tqdm.auto import tqdm
                    with partial.open("wb") as handle, tqdm(total=part["size"], unit="B", unit_scale=True,
                                                            desc=f"Asset {index}/{len(parts)}") as progress:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                handle.write(block)
                                progress.update(len(block))
                if file_sha256(partial) != part["sha256"]:
                    raise ValueError(f"Part {index} is incomplete or changed. Run this cell again.")
                partial.replace(target)
        partial = archive_path.with_suffix(".partial")
        with partial.open("wb") as output:
            for index in range(1, len(parts) + 1):
                with (root / f"asset.part{index}").open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        if file_sha256(partial) != archive_sha256:
            raise ValueError("Download is incomplete or changed. Run this cell again.")
        partial.replace(archive_path)
    extract_verified_assets(archive_path, root, archive_sha256, expected_files)
    print("Verified final model and original 9000-post data; ready for inference.")
    return root / "data", root / "seed_42_model"
