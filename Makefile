.PHONY: check setup help

# Default target
help:
	@echo "Available commands:"
	@echo "  make check  - Run fast checks (typecheck, lint, tests, build)"
	@echo "  make setup - Initialize project (copy .env.example to .env)"
	@echo "  make help  - Show this help message"

# Run fast checks
check:
	./scripts/check.sh

# Initialize project environment
setup:
	@echo "Setting up project..."
	@if [ ! -f .env ]; then \
		cp .env.example .env && \
		echo "✓ Created .env from .env.example"; \
	else \
		echo "⚠ .env already exists, skipping"; \
	fi
