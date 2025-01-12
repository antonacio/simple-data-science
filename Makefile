install-pre-commit:
	pip install pre-commit && \
	pre-commit install

lint:
	pre-commit run -a

test:
	python -m pytest
