# Aloy local dev — one place to run the whole stack.
#
# Run these from Git Bash (make is a Unix tool; recipes are bash), e.g.:
#   cd products/aloy && make dev
#
# The stack is three processes:
#   api    — FastAPI on :$(API_PORT)         (uv run aloy-backend)
#   worker — durable-run + cron loop          (uv run aloy-backend-worker)
#   web    — Vite dev server on :$(WEB_PORT)  (npm run dev)
# The web app talks to the API at localhost:$(API_PORT) by default, so no
# URL wiring is needed for local dev.

SHELL := bash
.ONESHELL:
.PHONY: help install dev api worker web stop check

API_HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_PORT ?= 5173
BACKEND  := backend
APP      := app

help: ## Show this help
	@echo "Aloy local dev - targets:"
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
	  sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Most common:  make install   (once)   then   make dev"

install: ## Install backend + frontend deps (run once, or after dependency changes)
	@echo "==> backend deps (uv sync)"
	cd $(BACKEND) && uv sync --extra dev
	@echo "==> frontend deps (npm install)"
	cd $(APP) && npm install
	@echo "==> done. Next: make dev"

dev: ## Run API + worker + web together; Ctrl-C stops all (see 'make stop' if a port sticks)
	@echo "Starting Aloy: API http://$(API_HOST):$(API_PORT)  |  web http://localhost:$(WEB_PORT)"
	@echo "Ctrl-C stops everything. If a port stays held afterwards, run: make stop"
	trap 'kill 0' EXIT INT TERM
	( cd $(BACKEND) && HOST=$(API_HOST) PORT=$(API_PORT) uv run aloy-backend ) &
	( cd $(BACKEND) && uv run aloy-backend-worker ) &
	( cd $(APP) && npm run dev -- --port $(WEB_PORT) ) &
	wait

api: ## Run only the API server (foreground; own terminal)
	cd $(BACKEND) && HOST=$(API_HOST) PORT=$(API_PORT) uv run aloy-backend

worker: ## Run only the durable worker (foreground; own terminal)
	cd $(BACKEND) && uv run aloy-backend-worker

web: ## Run only the web app (foreground; own terminal)
	cd $(APP) && npm run dev -- --port $(WEB_PORT)

stop: ## Kill the API/web port holders AND any stray worker (Windows orphan cleanup)
	@echo "Freeing ports $(API_PORT) and $(WEB_PORT), and stopping any stray worker..."
	@powershell -NoProfile -Command \
	  "Get-NetTCPConnection -LocalPort $(API_PORT),$(WEB_PORT) -State Listen -ErrorAction SilentlyContinue | \
	   Select-Object -ExpandProperty OwningProcess -Unique | \
	   ForEach-Object { Write-Host ('  killing port PID ' + \$$_); Stop-Process -Id \$$_ -Force -ErrorAction SilentlyContinue }; \
	   Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | \
	   Where-Object { \$$_.Name -in @('python.exe','uv.exe') -and \$$_.CommandLine -like '*aloy-backend-worker*' } | \
	   ForEach-Object { Write-Host ('  killing worker PID ' + \$$_.ProcessId); Stop-Process -Id \$$_.ProcessId -Force -ErrorAction SilentlyContinue }; \
	   Write-Host '  done.'"

check: ## Verify the API is up (one listener on the port + health route)
	@powershell -NoProfile -Command \
	  "\$$c = try { Invoke-RestMethod http://$(API_HOST):$(API_PORT)/v1/health -TimeoutSec 3 } catch { \$$null }; \
	   if (\$$c) { Write-Host '  API healthy:' (\$$c | ConvertTo-Json -Compress) } else { Write-Host '  API not responding on :$(API_PORT)' }"
