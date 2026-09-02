.PHONY: test benchmark research-lab research-stress oracle-audit aws-package aws-contract reproduce secret-gate

PYTHON ?= python3
PYTHONPATH := src

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m carno_router benchmark \
		--data tests/fixtures/synthetic_battles.csv \
		--pricing configs/normalized_costs.json \
		--output artifacts/synthetic_reference \
		--seed 1729 --bootstrap 500 --selection-bootstrap 500 --max-quality-drop 0.02

research-lab:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/build_research_autonomy_lab.py \
		--output artifacts/research_autonomy_v1

research-stress:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/build_research_stress_lab.py \
		--output artifacts/research_stress_v1 --replicates 100 --root-seed 2831729

oracle-audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/build_reference_oracle_audit.py \
		--stress artifacts/research_stress_v1 --output artifacts/research_stress_oracle_v1 \
		--attacks-per-task 200 --root-seed 7071067

aws-package:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/build_aws_lambda_package.py \
		--output build/aws/carno_the283_lambda.zip

aws-contract: aws-package
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/validate_aws_shard_contract.py \
		--package build/aws/carno_the283_lambda.zip \
		--output artifacts/aws_lambda_shard_contract_validation.json \
		--core-output artifacts/aws_lambda_shard_contract_core.json

reproduce:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) tools/reproducibility_gate.py

secret-gate:
	$(PYTHON) tools/secret_gate.py .
