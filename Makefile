setup:
	@echo "Setting up the project..."
	pip install --upgrade uv
	make create-venv
	make activate-venv
	make sync-requirements
	make install-pre-commit

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

install-pre-commit:
	@echo "Installing pre-commit..."
	pre-commit install

checks:
	@echo "Running checks..."
	pre-commit run -a

unzip-datasets:
	@echo "Unzipping datasets..."
	unzip -j data/expenses.csv.zip -d data/
	unzip -j data/fetal_health.csv.zip -d data/
