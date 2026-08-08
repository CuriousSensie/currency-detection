# Development guide

## Commands

- `make install`: create the Python environment and install web dependencies.
- `make lint`: Python lint/type checks plus TypeScript lint/type checks.
- `make test`: Python and component tests.
- `make verify`: lint, tests, and production web build.
- `make api` / `make web`: local development servers.

Training dependencies are intentionally optional. API tests inject a classifier double through the same readiness boundary used by the default application, so contract behavior can be checked without a release ONNX artifact.

## Module rules

- Keep HTTP concepts out of `pkrvision`.
- Keep training-framework imports lazy so CPU serving and tests remain lean.
- Store no executable pickle artifacts at the API trust boundary.
- Add versioned contracts rather than silently changing `/v1` behavior.
- Add a canonical result field before presenting it in the UI or README.
- Document public functions and non-obvious scientific tradeoffs.

## Pull-request evidence

Include affected protocol/config files, commands run, test output, and whether any result or model claim changed. A model change needs artifact checksums and regenerated evaluation; a visual-only change must not change factual evidence.
