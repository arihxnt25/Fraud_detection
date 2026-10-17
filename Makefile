.PHONY: install demo train api dashboard test lint docker

install:
	pip install -r requirements-dev.txt

demo:            ## train on synthetic data (no download needed)
	python scripts/run_pipeline.py --synthetic

train:           ## train on IEEE-CIS (10% stratified sample; set SAMPLE=1.0 for full)
	python scripts/run_pipeline.py --sample $${SAMPLE:-0.1}

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q

lint:
	ruff check src api dashboard scripts tests

docker:
	docker compose up --build
