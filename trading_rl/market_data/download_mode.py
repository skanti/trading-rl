"""Select initial, incremental, or explicitly rebuilt NPZ downloads."""

from pathlib import Path


def use_incremental_download(
    output: Path, *, update: bool = False, rebuild: bool = False
) -> bool:
    if update and rebuild:
        raise ValueError("--update and --rebuild are mutually exclusive")
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    manifest = output.with_suffix(".json")
    if output.exists() != manifest.exists():
        raise ValueError(
            f"incomplete dataset: {output} and {manifest} must both exist or both be absent"
        )
    if update and not output.exists():
        raise ValueError("--update requires the existing NPZ and JSON manifest")
    return output.exists() and not rebuild
