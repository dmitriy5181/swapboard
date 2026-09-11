from pathlib import Path

import pytest

from swapboard.api.config import load_config, model_paths_by_name, model_sources


def test_parse_extracts_all_models_with_hf_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "models:\n"
        '  "embed":\n'
        "    cmd: |\n"
        "      llama-server --port ${PORT}\n"
        "      -m /models/acme/embed-GGUF/embed-Q8_0.gguf\n"
        '  "reranker":\n'
        "    cmd: |\n"
        "      llama-server --port ${PORT}\n"
        "      --model=/models/acme/reranker-GGUF/reranker-Q8_0.gguf\n",
        encoding="utf-8",
    )

    sources = {
        source.name: source for source in model_sources(load_config(config_path))
    }

    assert set(sources) == {"embed", "reranker"}

    embed = sources["embed"].primary_file
    assert embed.repo_id == "acme/embed-GGUF"
    assert embed.filename == "embed-Q8_0.gguf"
    assert embed.relative_path == "acme/embed-GGUF/embed-Q8_0.gguf"

    reranker = sources["reranker"].primary_file
    assert reranker.repo_id == "acme/reranker-GGUF"
    assert reranker.filename == "reranker-Q8_0.gguf"


def test_parse_skips_models_without_resolvable_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "models:\n"
        '  "valid":\n'
        "    cmd: |\n"
        "      llama-server -m /models/acme/demo-GGUF/demo-Q8_0.gguf\n"
        '  "no-model-flag":\n'
        "    cmd: |\n"
        "      llama-server --port ${PORT}\n"
        '  "shallow-path":\n'
        "    cmd: |\n"
        "      llama-server -m demo.gguf\n"
        '  "no-cmd":\n'
        "    proxy: http://localhost:9000\n",
        encoding="utf-8",
    )

    sources = model_sources(load_config(config_path))

    assert {source.name for source in sources} == {"valid"}


def test_parse_skips_paths_whose_repository_would_be_the_root(tmp_path: Path) -> None:
    """`/opt/x.gguf` has three parts, the first being `/`, and must not resolve.

    Accepting it would name the repository `/` and produce an absolute relative
    path, placing the model outside the models directory.
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'models:\n  "rooted":\n    cmd: llama-server -m /opt/demo.gguf\n',
        encoding="utf-8",
    )

    assert model_sources(load_config(config_path)) == []


def test_parse_skips_paths_that_climb_out_of_the_models_directory(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'models:\n  "climber":\n    cmd: llama-server -m ../../etc/demo.gguf\n',
        encoding="utf-8",
    )

    assert model_sources(load_config(config_path)) == []


def test_parse_resolves_macro_prefixed_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "macros:\n"
        '  models_dir: "${env.MODELS_DIR}"\n'
        '  llama_server: "${env.LLAMA_SERVER_BIN}"\n'
        "models:\n"
        '  "demo":\n'
        "    cmd: |\n"
        "      ${llama_server} --port ${PORT}\n"
        "      -m ${models_dir}/acme/demo-GGUF/demo-Q8_0.gguf\n",
        encoding="utf-8",
    )

    sources = model_sources(load_config(config_path))

    assert len(sources) == 1
    source = sources[0]
    assert source.name == "demo"
    assert source.primary_file.repo_id == "acme/demo-GGUF"
    assert source.primary_file.filename == "demo-Q8_0.gguf"
    assert source.primary_file.relative_path == "acme/demo-GGUF/demo-Q8_0.gguf"


@pytest.mark.parametrize(
    "projector_argument",
    [
        "--mmproj /models/acme/demo-GGUF/mmproj-F16.gguf",
        "--mmproj=/models/acme/demo-GGUF/mmproj-F16.gguf",
    ],
)
def test_parse_extracts_multimodal_projector(
    tmp_path: Path, projector_argument: str
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "models:\n"
        '  "demo":\n'
        "    cmd: |\n"
        "      llama-server -m /models/acme/demo-GGUF/demo-Q4_K_M.gguf\n"
        f"      {projector_argument}\n",
        encoding="utf-8",
    )

    source = model_sources(load_config(config_path))[0]

    assert [model_file.filename for model_file in source.files] == [
        "demo-Q4_K_M.gguf",
        "mmproj-F16.gguf",
    ]
    assert all(model_file.repo_id == "acme/demo-GGUF" for model_file in source.files)


def test_parse_skips_model_with_unresolvable_projector_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "models:\n"
        '  "demo":\n'
        "    cmd: |\n"
        "      llama-server -m /models/acme/demo-GGUF/demo-Q4_K_M.gguf\n"
        "      --mmproj mmproj-F16.gguf\n",
        encoding="utf-8",
    )

    assert model_sources(load_config(config_path)) == []


def test_parse_returns_empty_for_config_without_models(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("healthCheckTimeout: 300\n", encoding="utf-8")

    assert model_sources(load_config(config_path)) == []


def test_model_paths_by_name_reports_a_path_no_source_could_be_derived_from(
    tmp_path: Path,
) -> None:
    """swapboard cannot download it, but it must know the file is spoken for."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "models:\n  solo:\n    cmd: llama-server -m /srv/solo.gguf\n",
        encoding="utf-8",
    )
    document = load_config(config_path)

    assert model_sources(document) == []
    assert model_paths_by_name(document) == {"solo": ["/srv/solo.gguf"]}


def test_model_paths_by_name_includes_the_multimodal_projector(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "models:\n"
        "  vision:\n"
        "    cmd: llama-server -m /models/acme/v-GGUF/v.gguf"
        " --mmproj /models/acme/v-GGUF/proj.gguf\n",
        encoding="utf-8",
    )

    assert model_paths_by_name(load_config(config_path)) == {
        "vision": ["/models/acme/v-GGUF/v.gguf", "/models/acme/v-GGUF/proj.gguf"]
    }
