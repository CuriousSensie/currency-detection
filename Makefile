.PHONY: install lint test api web verify

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev]'
	cd web && pnpm install

lint:
	.venv/bin/ruff check ml api tests
	.venv/bin/mypy ml/src api
	cd web && pnpm lint && pnpm typecheck

test:
	.venv/bin/pytest
	cd web && pnpm test

api:
	.venv/bin/uvicorn api.app.main:app --reload --port 8000

web:
	cd web && pnpm dev

verify: lint test
	cd web && pnpm build
