checks:
	@echo "Running checks..."
	pre-commit run -a

unzip-datasets:
	@echo "Unzipping datasets..."
	unzip -j data/expenses.csv.zip -d data/
	unzip -j data/fetal_health.csv.zip -d data/

convert-notebooks-to-html:
	rm -rf examples/*.html
	@for nb in src/*.ipynb; do \
		echo "Converting $$nb to HTML..."; \
		jupyter nbconvert --to html --execute "$$nb" --output-dir=examples/ --ExtractOutputPreprocessor.enabled=False; \
	done
