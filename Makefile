install:
	pip install -r requirements.txt

test:
	pytest -q

train:
	python -m semantic_canary.cli train --rows 100000

benchmark:
	python -m semantic_canary.cli benchmark --rows 100000

api:
	uvicorn semantic_canary.api:app --reload
