# nebula3d — common dev tasks.
#
# The web UI is a single React app (web/) built into two gitignored targets:
#   make ui        -> src/nebula3d/server/static  (served by native `nebula3d-web`)
#   make ui-pages  -> web/dist                 (GitHub Pages / Pyodide build)
# Both are build artifacts: rerun the matching target after changing web/src,
# or the running UI will be stale (native `nebula3d-web` warns at startup if so).

NPM ?= npm
WEB := web

.DEFAULT_GOAL := help

.PHONY: help ui ui-pages web-install check

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

ui: ## Rebuild the native (API-mode) SPA served by `nebula3d-web`
	cd $(WEB) && $(NPM) run build

ui-pages: ## Rebuild the GitHub Pages / Pyodide bundle (web/dist)
	cd $(WEB) && $(NPM) run build:pages

web-install: ## Install frontend dependencies (npm install in web/)
	cd $(WEB) && $(NPM) install

check: ## Run the backend test + lint + type suite
	./scripts/check.sh
