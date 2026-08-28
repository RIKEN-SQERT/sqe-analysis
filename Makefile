.PHONY: docs docs-clean docs-live

docs:
	uv run --group docs sphinx-build -M html docs docs/_build

docs-clean:
	uv run --group docs sphinx-build -M clean docs docs/_build
	rm -rf docs/api/_autosummary

docs-live:
	uv run --group docs --with sphinx-autobuild sphinx-autobuild --watch src docs docs/_build/html
