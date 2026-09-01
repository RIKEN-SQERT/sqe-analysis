.PHONY: docs docs-clean docs-live

docs:
	uv run --group docs sphinx-build -M html docs docs/_build -j "auto"

docs-clean:
	uv run --group docs sphinx-build -M clean docs docs/_build

docs-live:
	uv run --group docs --with sphinx-autobuild sphinx-autobuild --watch src docs docs/_build/html -j "auto"
