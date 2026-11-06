export PATH := $(CURDIR)/.venv/bin:$(PATH)
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

.PHONY: all test inference dashboard serve run clean

all: test

test:
	pytest -q --strict-markers

inference:
	python3 -c "from params.space import ParameterSpace; from inference.simulator import simulate_batch; space = ParameterSpace(); print('Simulator active. Dim:', space.dim)"

dashboard:
	python3 ui/virtual_patient.py

serve:
	python3 ui/server.py 8050

run:
	python3 main.py

clean:
	rm -rf .pytest_cache/ __pycache__/ */__pycache__/ .coverage data/sim/*
