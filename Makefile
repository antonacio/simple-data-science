setup:
	@echo "Setting up the project..."
	pip install --upgrade uv
	make create-venv
	make activate-venv
	make sync-requirements

uninstall:
	@echo "Uninstalling the project..."
	make deactivate-venv
	make delete-venv

create-venv:
	@echo "Creating virtual environment..."
	uv venv --python 3.12

activate-venv:
	@echo "Activating virtual environment..."
	. .venv/bin/activate

deactivate-venv:
	@echo "Deactivating virtual environment..."
	deactivate

delete-venv:
	@echo "Deleting virtual environment..."
	rm -rf .venv

sync-requirements:
	@echo "Syncing requirements..."
	uv sync

checks:
	@echo "Running checks..."
	pre-commit run -a
