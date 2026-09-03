# Guidelines

## General

- This project is using uv for project management, the python interpreter should be started via `uv run python ...`

## Code

- Never describe arguments and return values in Python docstrings.
- Avoid module docstrings unless they capture non-obvious design constraints or invariants.
- Do not maintain backward compatibility when implementing or refactoring.
- When writing new code or refactoring existing code, follow Clean Code principles and load the `clean-code` agentic skill before making changes.
- Do not use plain JavaScript unless strictly necessary. Prefer Bootstrap-native components (modals, collapses, tabs, dropdowns, etc.) and HTMX for server-driven interactions; only fall back to Alpine.js when Bootstrap and HTMX cannot cover the required behaviour.
- Every new implementation must include at least one meaningful happy-path test that exercises observable behavior and would catch a realistic regression. Do not add tests solely to satisfy this requirement or tests that merely restate the implementation.

## Checks

- If Python files changed, run `uv run pytest`, `uv run ty check`, and `uv run ruff check`.
- If frontend JavaScript changed, run `npm test` (requires `npm install` once).
- If any Docker Compose file changed, run `docker compose config` with the required placeholder env variables set.
- If any file in `.github/workflows/` changed, run `actionlint`.
