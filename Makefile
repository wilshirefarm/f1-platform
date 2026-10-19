.PHONY: install lint fmt test up down replay train simulate

RACE ?= 2024-monza
SPEED ?= 60

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

up:
	docker compose up -d

down:
	docker compose down

replay:
	@echo "replay RACE=$(RACE) SPEED=$(SPEED): not implemented (Stage 2)"

train:
	@echo "train: not implemented (Stage 3)"

simulate:
	@echo "simulate: not implemented (Stage 3b)"
