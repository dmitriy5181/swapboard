# swapboard

Model management API and dashboard for [llama-swap](https://github.com/mostlygeek/llama-swap).

swapboard reads your llama-swap configuration, works out which Hugging Face
repository each model comes from, reports whether its files are present on
disk, and downloads the missing ones on request. It ships two parts:

- a **JSON API** (FastAPI) that other services can drive, and
- a **web dashboard** (Flask + HTMX) for doing it by hand.

On macOS it can also deploy itself, running llama-swap and llama.cpp from
pinned, checksummed upstream builds kept in a private directory, so neither
binary lands on your `PATH` and your own llama.cpp build is left alone.

## Install

```sh
pip install swapboard              # API + client
pip install 'swapboard[ui]'        # + web dashboard
pip install 'swapboard[all]'       # + macOS deployment and the local runner
```

Requires Python 3.14 or newer.

## Quick start

```sh
cp .env.example .env
swapboard-dev --config ./llama-swap.example.yml
```

`swapboard-dev` runs llama-swap, the API and the dashboard together with
interleaved logs, and prints the URL to open. On macOS it fetches the pinned
runtimes on first use; elsewhere it falls back to a `llama-swap` on your
`PATH`.

## Configuration

swapboard derives a model's Hugging Face source from the last three components
of its `-m` path, so models must be laid out as `<org>/<repo>/<filename>`. A
model with an `--mmproj` projector is only reported present once both files
exist.

Refer to llama-server through a macro rather than a bare name, so the binary
swapboard manages is the one that gets used:

```yaml
macros:
  models_dir: "${env.MODELS_DIR}"
  llama_server: "${env.LLAMA_SERVER_BIN}"

models:
  "embeddinggemma-300M":
    cmd: |
      ${llama_server} --port ${PORT}
      -m ${models_dir}/ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf
      --embeddings
```

swapboard sets `MODELS_DIR` and `LLAMA_SERVER_BIN` when it launches llama-swap.
See [`llama-swap.example.yml`](llama-swap.example.yml) for a fuller example.

Configuration is read from `SWAPBOARD_*` environment variables, covering the
config file to read, where models are stored, a Hugging Face token for private
or gated repositories, and the addresses the services bind to.
[`.env.example`](.env.example) names every variable, with values suited to
running from a checkout. Left unset, the paths resolve to the installation the
running swapboard belongs to, and `--help` reports the ports.

### Python client

```python
from swapboard import SwapboardClient

client = SwapboardClient("http://127.0.0.1:8771")
status = client.get_status()          # never raises; degrades to "unavailable"
for model in status.models:
    print(model.name, model.present, model.download_state)
```

## Running the services

```sh
swapboard-api --help
swapboard-ui --help
```

Or point any ASGI/WSGI server at `swapboard.api.main:app` and
`swapboard.ui.factory:create_app()`.

## macOS deployment

`swapboard-deploy` installs everything under a single prefix and runs it from
launchd. The prefix is derived from the virtualenv the command runs in, so
there is normally nothing to configure:

```sh
uv venv ~/.swapboard/venv --python 3.14
uv pip install --python ~/.swapboard/venv/bin/python 'swapboard[all]'
~/.swapboard/venv/bin/swapboard-deploy deploy --config ./llama-swap.yml
```

Runtimes, models, config and logs all live under that prefix, and the services
run as launchd agents. `swapboard-deploy uninstall` removes the agents and
runtimes, leaving models and config in place. Apple Silicon and Intel are both
supported.

The pinned runtimes are verified against recorded SHA-256 digests before
extraction, and can be managed on their own:

```sh
swapboard-runtimes install    # or: status, path llama-server, remove
```

Linux is supported for the API and dashboard; for llama-swap itself, use the
upstream container image.

## Development

```sh
uv sync --all-extras --group dev
uv run ruff check src tests
uv run ty check src tests
uv run pytest --cov
```

## License

MIT
