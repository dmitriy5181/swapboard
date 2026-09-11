from pathlib import Path

import pytest

from swapboard.api.store import ModelStore
from swapboard.common.models import ModelFile, ModelSource


def gguf(root: Path, relative_path: str, size: int = 8) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def source(*relative_paths: str, name: str = "demo") -> ModelSource:
    return ModelSource(
        name=name,
        files=tuple(
            ModelFile(
                relative_path=relative_path,
                repo_id="/".join(relative_path.split("/")[:2]),
                filename=relative_path.split("/")[-1],
            )
            for relative_path in relative_paths
        ),
    )


def test_size_of_sums_every_file_of_a_model(tmp_path: Path) -> None:
    gguf(tmp_path, "acme/demo-GGUF/demo.gguf", size=10)
    gguf(tmp_path, "acme/demo-GGUF/mmproj.gguf", size=4)

    size = ModelStore(tmp_path).size_of(
        source("acme/demo-GGUF/demo.gguf", "acme/demo-GGUF/mmproj.gguf")
    )

    assert size == 14


def test_size_of_a_missing_model_is_zero(tmp_path: Path) -> None:
    assert ModelStore(tmp_path).size_of(source("acme/demo-GGUF/demo.gguf")) == 0


def test_repository_directory_excludes_a_nested_filename(tmp_path: Path) -> None:
    model_file = source("acme/demo-GGUF/MTP/draft.gguf").primary_file

    directory = ModelStore(tmp_path).repository_directory(model_file)

    assert directory == tmp_path / "acme/demo-GGUF"


def test_remove_deletes_the_files_and_the_emptied_directory(tmp_path: Path) -> None:
    gguf(tmp_path, "acme/demo-GGUF/demo.gguf")
    gguf(tmp_path, "acme/demo-GGUF/mmproj.gguf")

    removed = ModelStore(tmp_path).remove(
        source("acme/demo-GGUF/demo.gguf", "acme/demo-GGUF/mmproj.gguf"), []
    )

    assert removed is True
    assert not (tmp_path / "acme/demo-GGUF").exists()


def test_remove_clears_the_download_cache_left_beside_the_files(tmp_path: Path) -> None:
    """`hf_hub_download` leaves a `.cache` tree that would outlive the model."""
    gguf(tmp_path, "acme/demo-GGUF/demo.gguf")
    cache = tmp_path / "acme/demo-GGUF/.cache/huggingface/demo.metadata"
    cache.parent.mkdir(parents=True)
    cache.write_text("meta", encoding="utf-8")

    ModelStore(tmp_path).remove(source("acme/demo-GGUF/demo.gguf"), [])

    assert not (tmp_path / "acme/demo-GGUF").exists()


def test_remove_keeps_a_directory_another_model_still_uses(tmp_path: Path) -> None:
    gguf(tmp_path, "acme/demo-GGUF/small.gguf")
    gguf(tmp_path, "acme/demo-GGUF/large.gguf")

    ModelStore(tmp_path).remove(source("acme/demo-GGUF/small.gguf"), [])

    assert not (tmp_path / "acme/demo-GGUF/small.gguf").exists()
    assert (tmp_path / "acme/demo-GGUF/large.gguf").exists()


def test_remove_reports_nothing_removed_when_absent(tmp_path: Path) -> None:
    assert ModelStore(tmp_path).remove(source("acme/demo-GGUF/demo.gguf"), []) is False


def test_stray_lists_files_no_configured_model_claims(tmp_path: Path) -> None:
    gguf(tmp_path, "acme/keep-GGUF/keep.gguf")
    gguf(tmp_path, "acme/gone-GGUF/gone.gguf", size=16)

    stray = ModelStore(tmp_path).stray(["acme/keep-GGUF/keep.gguf"])

    assert len(stray) == 1
    assert stray[0].relative_path == "acme/gone-GGUF"
    assert stray[0].files == ("gone.gguf",)
    assert stray[0].size_bytes == 16


def test_stray_groups_every_file_of_one_directory_together(tmp_path: Path) -> None:
    gguf(tmp_path, "acme/gone-GGUF/gone.gguf", size=16)
    gguf(tmp_path, "acme/gone-GGUF/mmproj.gguf", size=4)

    stray = ModelStore(tmp_path).stray([])

    assert len(stray) == 1
    assert stray[0].files == ("gone.gguf", "mmproj.gguf")
    assert stray[0].size_bytes == 20


def test_stray_ignores_the_hugging_face_cache(tmp_path: Path) -> None:
    """Cache entries are bookkeeping, not models a user chose to keep."""
    cached = tmp_path / "acme/demo-GGUF/.cache/huggingface/download/demo.gguf"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x")

    assert ModelStore(tmp_path).stray([]) == []


def test_stray_ignores_files_that_are_not_gguf(tmp_path: Path) -> None:
    readme = tmp_path / "acme/demo-GGUF/README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("notes", encoding="utf-8")

    assert ModelStore(tmp_path).stray([]) == []


def test_stray_stands_a_loose_file_on_its_own(tmp_path: Path) -> None:
    """Grouping it by its parent would make removal mean the whole store."""
    gguf(tmp_path, "loose.gguf")

    stray = ModelStore(tmp_path).stray([])

    assert stray[0].relative_path == "loose.gguf"


def test_stray_of_a_missing_models_directory_is_empty(tmp_path: Path) -> None:
    assert ModelStore(tmp_path / "absent").stray([]) == []


def test_stray_leaves_a_model_whose_path_only_the_config_spells(
    tmp_path: Path,
) -> None:
    """A file swapboard cannot download is still a file llama-swap is running."""
    gguf(tmp_path, "solo.gguf")

    assert ModelStore(tmp_path).stray([str(tmp_path / "solo.gguf")]) == []


def test_remove_stray_deletes_a_whole_stray_directory(tmp_path: Path) -> None:
    gguf(tmp_path, "acme/gone-GGUF/gone.gguf")

    assert ModelStore(tmp_path).remove_stray("acme/gone-GGUF", []) is True
    assert not (tmp_path / "acme/gone-GGUF").exists()


def test_remove_stray_spares_a_configured_file_beside_it(tmp_path: Path) -> None:
    """The entry names the directory, but only the unclaimed file is stray."""
    gguf(tmp_path, "acme/demo-GGUF/keep.gguf")
    gguf(tmp_path, "acme/demo-GGUF/gone.gguf")

    removed = ModelStore(tmp_path).remove_stray(
        "acme/demo-GGUF", ["acme/demo-GGUF/keep.gguf"]
    )

    assert removed is True
    assert (tmp_path / "acme/demo-GGUF/keep.gguf").exists()
    assert not (tmp_path / "acme/demo-GGUF/gone.gguf").exists()


def test_remove_stray_refuses_a_path_holding_only_configured_files(
    tmp_path: Path,
) -> None:
    """Nothing there was ever reported as stray, so nothing there may go."""
    gguf(tmp_path, "acme/demo-GGUF/demo.gguf")

    removed = ModelStore(tmp_path).remove_stray(
        "acme/demo-GGUF", ["acme/demo-GGUF/demo.gguf"]
    )

    assert removed is False
    assert (tmp_path / "acme/demo-GGUF/demo.gguf").exists()


def test_remove_stray_reports_an_unknown_target(tmp_path: Path) -> None:
    assert ModelStore(tmp_path).remove_stray("acme/absent", []) is False


@pytest.mark.parametrize("relative_path", ["../outside", "acme/../../outside", "/etc"])
def test_remove_stray_removes_nothing_outside_the_models_directory(
    tmp_path: Path, relative_path: str
) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)

    assert ModelStore(tmp_path).remove_stray(relative_path, []) is False
    assert outside.exists()


def test_remove_stray_refuses_to_delete_the_store_itself(tmp_path: Path) -> None:
    gguf(tmp_path, "acme/demo-GGUF/demo.gguf")

    assert ModelStore(tmp_path).remove_stray(".", []) is False
    assert (tmp_path / "acme/demo-GGUF/demo.gguf").exists()


def test_remove_spares_a_file_another_model_still_claims(tmp_path: Path) -> None:
    """Two entries can serve the same weights with different settings."""
    gguf(tmp_path, "acme/demo-GGUF/shared.gguf")

    removed = ModelStore(tmp_path).remove(
        source("acme/demo-GGUF/shared.gguf"), ["acme/demo-GGUF/shared.gguf"]
    )

    assert removed is False
    assert (tmp_path / "acme/demo-GGUF/shared.gguf").exists()
