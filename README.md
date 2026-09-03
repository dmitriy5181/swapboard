# swapboard

Model management API and dashboard for [llama-swap](https://github.com/mostlygeek/llama-swap).

swapboard reads your llama-swap configuration, works out which Hugging Face
repository each model comes from, reports whether its GGUF files are present on
disk, and downloads the missing ones on request. It ships two parts:

- a **JSON API** (FastAPI) that other services can drive, and
- a **web dashboard** (Flask + HTMX) for doing it by hand.

On macOS it can also deploy itself: llama-swap and llama.cpp are downloaded from
pinned, checksummed upstream releases into a private directory and run as
launchd agents, so neither binary ever lands on your `PATH` and your own
llama.cpp build is left alone.

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

`swapboard-dev` downloads the pinned llama-swap and llama.cpp builds on first
run (macOS; elsewhere it falls back to a `llama-swap` on your `PATH`), then runs
llama-swap, the API and the dashboard together with interleaved logs.

| Service | Default port |
| --- | --- |
| Dashboard | 8770 |
| API | 8771 |
| llama-swap | 8772 |

Open <http://127.0.0.1:8770>.

## Configuration

swapboard derives a model's Hugging Face source from the last three components
of its `-m` path, so models must be laid out as `<org>/<repo>/<filename>`. A
model with an `--mmproj` projector is only reported present once both files
exist.

Refer to llama-server through the `${llama_server}` macro rather than a bare
name, so the private binary is used:

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

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `SWAPBOARD_LLAMA_SWAP_CONFIG_PATH` | `/etc/llama-swap/config/config.yaml` | Config to read models from |
| `SWAPBOARD_LLAMA_SWAP_PORT` | `8080` | Port llama-swap serves on, reported by `/info` |
| `SWAPBOARD_MODELS_PATH` | `/models` | Where GGUF files are downloaded |
| `SWAPBOARD_HF_TOKEN` | — | Token for private or gated repositories |
| `SWAPBOARD_UI_API_URL` | `http://127.0.0.1:8771` | Where the dashboard reaches the API |
| `SWAPBOARD_UI_HOST` | `127.0.0.1` | Dashboard bind address |
| `SWAPBOARD_UI_PORT` | `8770` | Dashboard port |

## API

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/info` | Port llama-swap is serving on |
| `GET` | `/models` | Status of every configured model |
| `GET` | `/models/{name}` | Status of one model |
| `POST` | `/models/{name}/download` | Starts a background download |

Downloads run in a background thread; poll `/models/{name}` for progress. A
model already downloading will not be started twice, and a failed download can
be retried without refetching files that already arrived.

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
swapboard-api --host 127.0.0.1 --port 8771
swapboard-ui  --host 127.0.0.1 --port 8770
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

That yields:

```
~/.swapboard/
├── venv/
├── runtimes/llama-cpp/       # pinned llama.cpp, never on PATH
├── runtimes/llama-swap/      # pinned llama-swap, never on PATH
├── models/
├── config/llama-swap.yml
└── log/{llama-swap,api,ui}.log
```

and three launchd agents: `com.swapboard.llama-swap`, `com.swapboard.api` and
`com.swapboard.ui`. Use `--no-ui` to skip the dashboard, and
`swapboard-deploy uninstall` to remove the agents and runtimes (models and
config are left in place).

Runtimes are verified against pinned SHA-256 digests before extraction and can
be managed on their own:

```sh
swapboard-runtimes install    # or: status, path llama-server, remove
```

Apple Silicon and Intel are both supported. Linux is supported for the API and
dashboard; for llama-swap itself, use the upstream container image.

## Development

```sh
uv sync --all-extras --group dev
uv run ruff check src tests
uv run ty check src tests
uv run pytest --cov
```

## License

MIT
