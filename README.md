# AI-Assistant for Power Grid Operation

This repository implements a multi-agent system to optimize the power grid dispatching operations. The system integrates several agents for topology management, reconnection, DCOPF optimization to get storage units, curtailment and redispatching and Imitation Learning based on Graph Transformer Model.

## Table of Contents
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Evaluation](#evaluation)

## Installation

Follow these steps to set up the environment and install required dependencies:

1. Create the conda environment:

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the environment:

    ```bash
    conda activate dispatching
    ```

3. Install the additional required package:

    ```bash
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    ```

## Repository Structure

```plaintext
├── Agent
│   ├── AgentReconnection.py        # Agent that reconnects disrupted lines in case of an attack.
│   ├── AgentRecoverTopo.py         # Agent to recover the original topology when grid is safe.
│   ├── AgentImitationTopk.py       # Imitation model to predict top-k actions as input for AgentTopology.
│   ├── AgentTopology.py            # Agent to search topology actions using greedy or full simulation.
│   ├── DispatcherAgent.py          # Dispatcher agent optimizing grid operations via DCOPF optimization.
│   ├── PowerGridAgent.py           # Main agent responsible for making decisions and controlling the grid which is combined all above modules
├── Imitation_model                 # Directory containing the imitation learning models.
│   ├── model                       # Pre-trained model for predicting top-k actions.
│   ├── src                         # Script for training the imitation model.
│   └── config                      # Config for the imitation models.
├── Env_test_challenge_L2RPN_IDF_2023
│   └── (dataset files for testing and evaluation)
├── config.json                     # Configuration file for agent parameters.
├── run_agent.py                    # Main script to run the agents.
├── demo.py                         # # Script for running a demo using Streamlit.
```

## Key Python Files

The repository contains the following key Python files, each serving a specific purpose in the multi-agent system:

| File Name                      | Description                                                                                     |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| `AgentReconnection.py`          | Agent responsible for reconnecting disrupted lines in case of an attack.                        |
| `AgentRecoverTopo.py`           | Agent to recover the original topology when the grid is safe.                                   |
| `AgentImitationTopk.py`         | Imitation model to predict top-k actions as input for the AgentTopology.                        |
| `AgentTopology.py`              | Agent to search topology actions using either greedy or full simulation methods.                 |
| `DispatcherAgent.py`            | Dispatcher agent optimizing grid operations via DCOPF optimization.                             |
| `run_agent.py`                  | Main script to run the agents and simulate decision-making in the power grid.                   |
| `test_eval.py`                  | Script for testing specific scenarios and evaluating performance on predefined datasets.        |
| `eval_score.py`                  | Script for evaluation score for dataset of challenge L2RPN2023 on [CodaBench Competition](https://www.codabench.org/competitions/1891/#/participate-tab).|
| `demo.py`                       | Script for running a demo using Streamlit to showcase the agents in action.                     |


## Configuration

The agent configurations are stored in the `config.json` file. Below are the key parameters:

| Parameter                    | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| algo                          | Specifies the algorithm, options include greedy search or full simulation of top-k actions.   |
| dispatching                   | Determines if the dispatcher module should be used (true or false).          |
| search_method                 | Options: multi_agent_dependent, multi_agent_independent, or single_agent.    |
| imitation                     | Set to true to use the Imitation model for top-k action predictions.         |
| topk                          | The number of top-k actions predicted by the Imitation model (e.g., 15).     |
| multiple_parades              | The number of best choices (e.g., 5) after reranking from simulation of top-k actions by grid2op.          |
| min_rho                       | Specifies the threshold for actions that reduce rho below the default value. |
| imitation_n1_config_path                       | config file path of Imitation learning model for N-1 case. |
| imitation_overload_config_path                       | config file path of Imitation learning model for Overload case. |
| action_space_path_N1                       | File path containing reduced topology action for N-1 case. |
| action_space_path_Overload                       | File path containing reduced topology action for Overload case.|

## Usage

### Testing

- To run the agent and simulate decision-making in the power grid, execute the `run_agent.py` script:

```bash
python run_agent.py
```


## Evaluation score for agent on test dataset of envronement L2RPN 2023  

- Evaluate the Model: Evaluate the model's performance and score on the provided dataset [Env_test_challenge_L2RPN_IDF_2023](https://www.dropbox.com/scl/fi/c498m7u262gnwx7vhq76l/Env_test_challenge_L2RPN_IDF_2023.zip?rlkey=1vdicuqqvszajo2hdqk2w5sap&st=xyou2z1o&dl=0):

```bash
python eval_score.py -env ./Env_test_challenge_L2RPN_IDF_2023/  -config config.json
```