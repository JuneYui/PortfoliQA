# PortfoliQA  Guide

This document provides instructions for setting up the required environment and running the end-to-end pipeline for the PortfoliQA framework.

## 1. Setup and Installation

It is recommended to use `conda` to manage the environment.

1.  **Create a new conda environment:**
    ```bash
    conda create --name portfoliqa python=3.10
    ```

2.  **Activate the environment:**
    ```bash
    conda activate portfoliqa
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Running

The following commands demonstrate how to run each stage. Please replace the placeholder file paths with the actual paths to your data files.

### Step 1: Semantic Parsing (Planer Agent)

This script uses to decompose questions into `Logical Fragments`.

* **Inputs:**
    * `-i`: Path to the `.pth` file containing the retrieved subgraphs from SubgraphRAG.
    * `-c`: Path to the file containing the topic entities for each question.
* **Output:** A `.jsonl` file containing the generated logical fragments.

```bash
python semantic_parser.py -i path/to/subgraph_data.pth -c path/to/topic_entities.jsonl
```

### Step 2: Query Plan Construction

This script takes the `Logical Fragments` and uses the deterministic construction tool to generate an optimized `Query Plan`.

* **Inputs:**
    * `-p`: Path to the `logical_fragments.jsonl` file generated in the previous step.
    * `-g`: Path to the SubgraphRAG `.pth` file.
* **Output:** A `.jsonl` file containing the executable query plans.

```bash
python query_constructor.py -p path/to/logical_fragments.jsonl -g path/to/subgraph_data.pth
```

### Step 3: Plan Execution (Aligner Agents)

This script executes the `Query Plan` using the Aligner Agent swarm. The agents search for evidence paths and assemble the findings into `Evidence Portfolios`.

* **Inputs:**
    * `-q`: Path to the `query_plans.jsonl` file generated in the previous step.
    * `-g`: Path to the SubgraphRAG `.pth` file.
* **Output:** A `.jsonl` file containing the structured evidence portfolios.

```bash
python constraint_verifier.py -q path/to/query_plans.jsonl -g path/to/subgraph_data.pth
```

### Step 4: Final Reasoning (LLM Reasoner)

This final script passes the `Evidence Portfolios` to the LLM Reasoner for the final evaluation and ranking of answers.

* **Inputs:**
    * `-p`: Path to the `evidence_portfolios.jsonl` file generated in the previous step.
* **Output:** A `.jsonl` file containing the final ranked answers and justifications.

```bash
python reasoner.py -p path/to/evidence_portfolios.jsonl
```
