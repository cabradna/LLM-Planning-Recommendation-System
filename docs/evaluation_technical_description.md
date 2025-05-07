# Dyna-Q Job Recommender Model: Evaluation Process

**Note on Scope:** This document describes the evaluation process for the Dyna-Q job recommender model. While it references the general evaluation utilities found in `src/evaluate.py` and `src/utils/evaluator.py`, it places particular emphasis on how these utilities are employed within the experimental setup of the `neural_dyna_q_notebook.py` notebook. In this notebook context, the primary agents for comparison (`agent_cosine` and `agent_hybrid`) are trained directly within the session, rather than being loaded from pre-existing model paths (a capability also supported by `src/evaluate.py` for more general use).

## 1. Introduction

This document provides a detailed technical description of the evaluation process employed for the Dyna-Q based job recommendation system. The primary goal of this process is to rigorously assess the performance of different agent configurations, particularly to quantify the impact of Large Language Model (LLM) based pretraining on the agent\'s effectiveness in a simulated job recommendation environment. This document aims to be self-contained, enabling a reader familiar with reinforcement learning (RL) concepts to understand and potentially replicate the evaluation procedure.

## 2. Evaluation Objectives

The core objectives of the evaluation process are:

1.  **Assess Overall Agent Performance**: To measure the general effectiveness of a trained Dyna-Q agent in recommending jobs to simulated applicants.
2.  **Compare Different Training Strategies**: Specifically, within the `neural_dyna_q_notebook.py` context, to systematically compare the performance of a Dyna-Q agent trained solely with cosine similarity-based rewards (`agent_cosine`) against a Dyna-Q agent trained using a hybrid LLM-enhanced strategy (`agent_hybrid`).
3.  **Evaluate Cold-Start Performance**: To understand how well agents (trained under different strategies) perform in initial interactions when they have limited experience with specific applicants or the environment, which is a common challenge in recommender systems.
4.  **Analyze Performance Over Time**: To observe how an agent\'s performance metrics (e.g., reward, apply rate) evolve as it accumulates more (simulated) experience. This helps in understanding learning stability and convergence.
5.  **Ensure Reproducibility**: To establish a standardized evaluation protocol that allows for consistent and comparable results across different experiments and model iterations.

## 3. Evaluation Environment

In the context of the `neural_dyna_q_notebook.py` experiment, the evaluation environment is instantiated directly within the notebook (primarily in Block 5, and potentially modified for LLM integration in Block 7). This notebook-instantiated environment is then passed to the `Evaluator` methods.

While `src/evaluate.py` provides a command-line script for running evaluations and includes its own logic for environment instantiation based on arguments (e.g., `--use_llm_simulator`), the notebook evaluation uses the environment instance it creates locally.

*   **Environment Instantiation (Conceptual, as done in `neural_dyna_q_notebook.py`, Block 5):**
    The notebook selects an environment class based on the `reward_strategy` variable.
    ```python
    # From: neural_dyna_q_notebook.py (Block 5, simplified)
    # tensor_cache is initialized earlier
    # reward_strategy is defined (e.g., "hybrid", "cosine", "llm")

    if reward_strategy == "cosine":
        env = JobRecommendationEnv(
            tensor_cache=tensor_cache,
            # ... other params ...
        )
    elif reward_strategy == "llm":
        env = LLMSimulatorEnv(
            tensor_cache=tensor_cache,
            # ... other params ...
        )
    elif reward_strategy == "hybrid": 
        env = HybridEnv(
            tensor_cache=tensor_cache,
            # ... other params ...
            cosine_weight=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
        )
    # ... further setup for LLM in HybridEnv or LLMSimulatorEnv (Block 7)
    # if STRATEGY_CONFIG["llm"]["enabled"] and (reward_strategy == "llm" or reward_strategy == "hybrid"):
    #     env = env.setup_llm(llm_model, tokenizer, device="auto")
    ```
    *   `JobRecommendationEnv`: A standard environment simulating interactions.
    *   `LLMSimulatorEnv`: An environment that might incorporate an LLM to simulate responses.
    *   `HybridEnv`: An environment that combines aspects of cosine similarity and LLM-based rewards.
    *   All these environments are expected to be initialized with a `tensor_cache` in the notebook setup.

The specific details of states ($s$), actions ($a$), rewards ($r$), and transition dynamics ($s, a \rightarrow s', r$) are defined within these respective environment classes (located in `src/environments/job_env.py`). The evaluation utilities in `src/utils/evaluator.py` (e.g., `Evaluator.compare_agents`) interact with the provided `env` object through methods like `env.reset()`, `env.get_valid_actions()`, `env.get_action_vector()`, and `env.step()`.

## 4. Key Performance Metrics

The following metrics are systematically collected and calculated, primarily within the `evaluate_agent` method of the `Evaluator` class in `src/utils/evaluator.py`.

1.  **Episode Rewards List**: A collection of the total reward accumulated in each evaluation episode.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        # Within the episode loop:
        # episode_reward += reward 
        # ...
        # After episode loop:
        episode_rewards.append(episode_reward)
        ```

2.  **Average Episode Reward (`avg_reward`)**: The mean of the total rewards obtained across all evaluation episodes.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        avg_reward = np.mean(episode_rewards)
        ```

3.  **Standard Deviation of Episode Reward (`std_reward`)**: The standard deviation of the total rewards, indicating performance consistency.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        std_reward = np.std(episode_rewards)
        ```

4.  **Minimum Episode Reward (`min_reward`)**: The minimum total reward achieved in any single episode.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        # "episode_rewards" is a list of rewards from all episodes
        results = {
            # ...
            "min_reward": np.min(episode_rewards),
            # ...
        }
        ```
        *(Note: Direct calculation `np.min(episode_rewards)` is part of the results dictionary construction).*

5.  **Maximum Episode Reward (`max_reward`)**: The maximum total reward achieved in any single episode.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        # "episode_rewards" is a list of rewards from all episodes
        results = {
            # ...
            "max_reward": np.max(episode_rewards),
            # ...
        }
        ```
        *(Note: Direct calculation `np.max(episode_rewards)` is part of the results dictionary construction).*

6.  **Episode Steps List**: A collection of the number of steps taken in each evaluation episode.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        # Within the episode loop:
        # step_count += 1
        # ...
        # After episode loop:
        episode_steps.append(step_count)
        ```

7.  **Average Steps per Episode (`avg_steps`)**: The mean number of steps taken per episode.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        avg_steps = np.mean(episode_steps)
        ```

8.  **Response Counts (`response_counts`)**: A tally of each type of discrete response string received from the environment's `info` dictionary.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        # response_counts = defaultdict(int)
        # ...
        # Within the step loop, after env.step(action_idx):
        response = info.get("response", "")
        if response:
            response_counts[response] += 1
        ```

9.  **Apply Rate (`apply_rate`)**: The proportion of total steps where the environment's response was specifically `"APPLY"`.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        total_steps = sum(episode_steps)
        apply_rate = response_counts.get("APPLY", 0) / total_steps if total_steps > 0 else 0
        ```

10. **Response Percentages (`response_percentages`)**: The percentage distribution of each recorded response type out of the total steps taken across all episodes.
    *   **Implementation (`src/utils/evaluator.py`):**
        ```python
        # response_percentages = {resp: count / total_steps * 100 for resp, count in response_counts.items()}
        # This is calculated assuming total_steps > 0. The full line in context handles total_steps = 0 for apply_rate,
        # and response_percentages would be based on counts from response_counts.
        # A more direct quote for the percentage calculation part:
        response_percentages = {resp: count / total_steps * 100 for resp, count in response_counts.items()}
        ```
    *   **Note**: The specific strings for responses (other than `"APPLY"`) originate from the environment implementation (e.g., `JobRecommendationEnv.step()` or `LLMSimulatorEnv.step()`) and are not hardcoded in the `Evaluator` beyond the collection mechanism.

## 5. Evaluation Methodologies

### 5.1. Single Agent Evaluation

This is the fundamental procedure for assessing the performance of any given agent. The core logic is implemented in the `evaluate_agent` method of the `Evaluator` class (`src/utils/evaluator.py`).

**Procedure Overview:**

The agent interacts with the environment for a specified number of episodes (`num_episodes`). In each episode, an applicant is chosen, the environment is reset, and the agent takes actions for a maximum number of steps (`max_steps_per_episode`). Key metrics like rewards and responses are collected.

**Implementation Details (`src/utils/evaluator.py` - `Evaluator.evaluate_agent`):**

*   **Initialization of Metrics:**
    ```python
    episode_rewards = []
    episode_steps = []
    response_counts = defaultdict(int)
    ```

*   **Main Evaluation Loop (Iterating through episodes):**
    ```python
    for episode in range(num_episodes):
        applicant_id = np.random.choice(applicant_ids)
        state = env.reset(applicant_id)
        episode_reward = 0
        step_count = 0
        # ... (inner step loop follows)
    ```

*   **Inner Step Loop (Agent-Environment Interaction within an episode):**
    ```python
    for step in range(max_steps_per_episode):
        valid_action_indices = env.get_valid_actions()
        available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
        
        action_idx, _ = agent.select_action(state, available_actions, eval_mode=True)
        
        next_state, reward, done, info = env.step(action_idx)
        
        state = next_state
        episode_reward += reward
        step_count += 1
        
        response = info.get("response", "")
        if response:
            response_counts[response] += 1
        
        if done:
            break
    ```

*   **Recording Episode Metrics:**
    ```python
    episode_rewards.append(episode_reward)
    episode_steps.append(step_count)
    ```

*   **Final Metric Calculation (after all episodes are run - examples):**
    ```python
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_steps)
    total_steps = sum(episode_steps)
    apply_rate = response_counts.get("APPLY", 0) / total_steps if total_steps > 0 else 0
    response_percentages = {resp: count / total_steps * 100 for resp, count in response_counts.items()}
    ```
    The full set of calculated metrics is returned in a dictionary (see Section 4 for a complete list).

### 5.2. Comparative Evaluation: Cosine-Trained vs. Hybrid-Trained Agent (Notebook Context)

This methodology, as applied in `neural_dyna_q_notebook.py`, directly compares the `agent_cosine` (trained in Block 9 of the notebook using cosine similarity rewards) with the `agent_hybrid` (trained in Block 10 using the hybrid LLM strategy). The comparison leverages the `compare_agents` method of the `Evaluator` class (`src/utils/evaluator.py`).

**Procedure Overview (Notebook Context):**

1.  The `agent_cosine` is trained (Block 9 of the notebook).
2.  The `agent_hybrid` is trained independently with fresh network weights (Block 10 of the notebook).
3.  The `Evaluator.compare_agents` method is called, where its `baseline_agent` parameter receives `agent_cosine`, and its `pretrained_agent` parameter receives `agent_hybrid`.
4.  This method then evaluates both agents independently using `self.evaluate_agent` (detailed in Section 5.1).
5.  The results from these individual evaluations are used to calculate improvement percentages for key metrics.

**Implementation Details (`src/utils/evaluator.py` - `Evaluator.compare_agents`):**

*   **Evaluating Individual Agents (called with `agent_cosine` and `agent_hybrid` from the notebook):**
    ```python
    logger.info("Evaluating baseline agent...") # In notebook context, this is agent_cosine
    baseline_results = self.evaluate_agent(
        baseline_agent, env, applicant_ids, num_episodes, max_steps_per_episode
    )
    
    logger.info("Evaluating pretrained agent...") # In notebook context, this is agent_hybrid
    pretrained_results = self.evaluate_agent(
        pretrained_agent, env, applicant_ids, num_episodes, max_steps_per_episode
    )
    ```

*   **Calculating Improvement Percentages (comparing `agent_hybrid` to `agent_cosine`):**
    *   Reward Improvement ($\%\Delta R$):
        $$ \%\Delta R = \frac{\bar{R}_{ep, agent\_hybrid} - \bar{R}_{ep, agent\_cosine}}{|\bar{R}_{ep, agent\_cosine}|} \times 100\% $$
        (Handle division by zero if $\bar{R}_{ep, agent\_cosine} = 0$).
        *   **Code Snippet for Reward Improvement (variable names `pretrained_results` and `baseline_results` map to `agent_hybrid` and `agent_cosine` respectively in this context):**
            ```python
            avg_reward_improvement = (
                (pretrained_results["avg_reward"] - baseline_results["avg_reward"]) / 
                abs(baseline_results["avg_reward"]) * 100 if baseline_results["avg_reward"] != 0 else float('inf')
            )
            ```

    *   Apply Rate Improvement ($\%\Delta AR$):
        $$ \%\Delta AR = \frac{\text{ApplyRate}_{agent\_hybrid} - \text{ApplyRate}_{agent\_cosine}}{\text{ApplyRate}_{agent\_cosine}} \times 100\% $$
        (This formula is applied if $\text{ApplyRate}_{agent\_cosine} > 0$; otherwise, the improvement is considered $\text{float('inf')}$).
        *   **Code Snippet for Apply Rate Improvement (variable names map as above):**
            ```python
            apply_rate_improvement = (
                (pretrained_results["apply_rate"] - baseline_results["apply_rate"]) / 
                baseline_results["apply_rate"] * 100 if baseline_results["apply_rate"] > 0 else float('inf')
            )
            ```

*   **Compiling Comparison Results:**
    ```python
    comparison = {
        "baseline": baseline_results,    # Results for agent_cosine
        "pretrained": pretrained_results,  # Results for agent_hybrid
        "improvements": {
            "avg_reward": avg_reward_improvement,
            "apply_rate": apply_rate_improvement
        }
    }
    ```

*   **Visualization and Saving (Conceptual):**
    The method then calls internal functions for visualization and saving results.
    ```python
    # Create visualizations
    self.visualize_comparison(comparison)
    
    # Save results to file
    self.save_results(comparison, "agent_comparison.json")
    ```

### 5.3. Cold-Start Performance Evaluation

This assesses how agents perform with minimal interaction history. In the context of `neural_dyna_q_notebook.py`, this would involve comparing `agent_cosine` and `agent_hybrid` under these conditions. The generic logic for this type of evaluation is found in the `evaluate_cold_start_performance` function in `src/evaluate.py`.

**Procedure Overview (applying to `agent_cosine` vs. `agent_hybrid` from notebook):**

For a specified number of episodes (`num_episodes`), both agents (`agent_cosine` as `baseline_agent` and `agent_hybrid` as `pretrained_agent` in the function call) are evaluated. In each episode, an applicant is chosen, and each agent interacts with the environment for a fixed, small number of steps (`num_steps`). The cumulative reward for these initial steps is recorded and compared.

**Implementation Details (`src/evaluate.py` - `evaluate_cold_start_performance`):**

*   **Initialization:**
    ```python
    # logger.info(f"Evaluating cold-start performance for {num_episodes} episodes, {num_steps} steps each")
    baseline_episode_rewards = []
    pretrained_episode_rewards = [] # Corresponds to agent_hybrid in notebook context
    ```

*   **Agent Interaction Loop (Example for `baseline_agent`, which would be `agent_cosine`):**
    This loop runs for `num_steps` (e.g., `args.cold_start_steps`).
    ```python
    # Inside the episode loop, for the baseline_agent (agent_cosine):
    env.reset(applicant_id)
    baseline_reward = 0
    for step in range(num_steps):
        valid_action_indices = env.get_valid_actions()
        available_actions = [env.get_action_vector(idx) for idx in valid_action_indices]
        action_idx, _ = baseline_agent.select_action(env.current_state, available_actions, eval_mode=True)
        _, reward, _, _ = env.step(action_idx)
        baseline_reward += reward
    baseline_episode_rewards.append(baseline_reward)
    ```
    A similar loop is executed for the `pretrained_agent` (which would be `agent_hybrid` from the notebook) using the same `applicant_id`.

*   **Calculating Aggregate Metrics (comparing `agent_hybrid` to `agent_cosine`):**
    ```python
    avg_baseline_reward = np.mean(baseline_episode_rewards) # agent_cosine results
    avg_pretrained_reward = np.mean(pretrained_episode_rewards) # agent_hybrid results
    improvement = ((avg_pretrained_reward - avg_baseline_reward) / 
                   abs(avg_baseline_reward) * 100 if avg_baseline_reward != 0 else float('inf'))
    ```
    The `results` dictionary and visualization calls follow, as shown in the previous version of this document.

### 5.4. Performance Over Time Evaluation

This method assesses how a single agent's performance metrics evolve. In the context of `neural_dyna_q_notebook.py`, this evaluation would typically be applied to `agent_hybrid` after its full training in Block 10 to observe its learning stability or performance over extended simulated interactions. The process involves functions from both `src/evaluate.py` and `src/utils/evaluator.py`.

**Procedure Overview:**

1.  **Parameter Setup in `src/evaluate.py` (if called from script):**
    The `main` script can call `evaluate_performance_over_time` (from `src/evaluate.py`), passing an agent (e.g., `agent_hybrid` if adapted for this) and step configurations.
    *   **Code (`src/evaluate.py` - `main` function context if evaluating `agent_hybrid`):
        ```python
        # Assuming agent_hybrid is loaded or available as 'target_eval_agent'
        # if args.performance_over_time:
        #     logger.info("Evaluating performance over time")
        #     evaluate_performance_over_time(
        #         agent=target_eval_agent, # This would be agent_hybrid
        #         env=env,
        #         applicant_ids=sample_applicant_ids,
        #         steps=list(range(0, args.num_episodes, args.eval_interval))
        #     )
        ```

2.  **Delegation to `Evaluator` Class (`src/evaluate.py` - `evaluate_performance_over_time` function):
    This function calls the `Evaluator`'s method.
    *   **Code (`src/evaluate.py` - `evaluate_performance_over_time` function):
        ```python
        # def evaluate_performance_over_time(agent: DynaQAgent, ..., steps: List[int]) -> Dict[str, List[float]]:
        #     ...
        #     results = evaluator.evaluate_performance_over_time(
        #         agent=agent, # agent_hybrid passed here
        #         ...
        #         num_episodes=max(steps),
        #         eval_interval=min(steps)
        #     )
        #     return results
        ```

3.  **Core Logic in `Evaluator.evaluate_performance_over_time` (`src/utils/evaluator.py`):
    The `Evaluator` method performs iterative evaluation on the provided agent (e.g., `agent_hybrid`).
    *   The **Critical Note on `eval_interval` Parameter** (potential `ValueError` if `min(steps)` is 0) still applies as detailed in the previous version of this document.
    *   The **Implementation details** (Initialization, Iterative Loop, Result Compilation) from the previous version of this document for `Evaluator.evaluate_performance_over_time` remain valid, with the understanding that the `agent` parameter it receives would be `agent_hybrid` from the notebook's context.

## 6. Justification for Evaluation Design

The chosen evaluation methodologies are designed to provide a comprehensive understanding of agent behavior, with design choices directly supported by the codebase.

*   **Standard RL Metrics**: Average reward and steps per episode are standard in RL. Their calculation is shown in Section 4, based on `np.mean(episode_rewards)` and `np.mean(episode_steps)` respectively, as implemented in `src/utils/evaluator.py` (`Evaluator.evaluate_agent`).

*   **Task-Specific Metrics**: Apply rate and response percentages are crucial for recommender systems. `apply_rate` is explicitly calculated in `src/utils/evaluator.py` (`Evaluator.evaluate_agent`):
    ```python
    apply_rate = response_counts.get("APPLY", 0) / total_steps if total_steps > 0 else 0
    ```
    Response percentages are also calculated from `response_counts` (see Section 4).

*   **Controlled Comparisons**: Evaluating baseline and pretrained agents under identical conditions is ensured by passing the same `env`, `applicant_ids`, `num_episodes`, and `max_steps_per_episode` to `self.evaluate_agent` for both agents within `Evaluator.compare_agents` (`src/utils/evaluator.py`). Similarly, in `evaluate_cold_start_performance` (`src/evaluate.py`), the same `applicant_id` is used for both agents within each episode loop.

*   **Addressing Practical Challenges (Cold-Start)**: The `evaluate_cold_start_performance` function in `src/evaluate.py` directly implements this by running agents for a limited number of `num_steps` per episode.

*   **Reproducibility**: This is supported by:
    *   Setting random seeds at the beginning of `src/evaluate.py`'s `main` function:
        ```python
        # src/evaluate.py - main()
        set_seed(args.seed)
        ```
        The `set_seed` function itself (in `src/evaluate.py`): 
        ```python
        def set_seed(seed: int) -> None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        ```
    *   Passing the seed to environment constructors:
        ```python
        # src/evaluate.py - main()
        env = JobRecommendationEnv(db_connector=db_connector, random_seed=args.seed)
        # or LLMSimulatorEnv(..., random_seed=args.seed)
        ```
    *   Using a deterministic (greedy) policy for the agent during evaluation:
        ```python
        # src/utils/evaluator.py - Evaluator.evaluate_agent()
        action_idx, _ = agent.select_action(state, available_actions, eval_mode=True)
        ```
        This `eval_mode=True` flag is consistently used in all evaluation scenarios.

*   **Visualization**: The `Evaluator` class initializes and uses a `Visualizer` instance. Calls to plotting functions are made, for example, in `Evaluator.compare_agents`:
    ```python
    # src/utils/evaluator.py - Evaluator.compare_agents()
    self.visualize_comparison(comparison)
    ```
    And in `evaluate_cold_start_performance` (`src/evaluate.py`):
    ```python
    # src/evaluate.py - evaluate_cold_start_performance()
    visualizer.plot_evaluation_results(...)
    ```
    The `Visualizer` class itself (in `src/utils/visualizer.py`, though its internal code is not detailed here) is responsible for generating the plots.

## 7. Key Hyperparameters and Configuration

**Context for `neural_dyna_q_notebook.py`:**
It's important to distinguish between the general capabilities of `src/evaluate.py` (which includes loading models via CLI arguments) and the specific experimental setup in `neural_dyna_q_notebook.py`. In the notebook:
*   `agent_cosine` (baseline) is trained in Block 9.
*   `agent_hybrid` (experimental) is trained in Block 10.
*   These in-memory, trained agent instances are then directly passed to the evaluation functions (e.g., `Evaluator.compare_agents`).
*   Therefore, CLI arguments like `--baseline_path`, `--pretrained_path`, `--baseline_episode`, and `--pretrained_episode` from `src/evaluate.py` are not used by the notebook for defining `agent_cosine` and `agent_hybrid` for their primary comparison. They would be relevant if one were using `src/evaluate.py` to evaluate models saved from a previous notebook run.

The following CLI arguments from `src/evaluate.py` still influence how the evaluation utilities behave when called by the notebook (e.g., number of episodes for evaluation, seed): 

**Command-Line Arguments (`src/evaluate.py` - `if __name__ == "__main__":` block):**

*   **Model Paths:**
    *   `--baseline_path`: Path to the baseline model directory.
        ```python
        parser.add_argument("--baseline_path", type=str, help="Path to baseline model directory")
        ```
        *   Usage in `main()` if not provided: Defaults to a path constructed using `PATH_CONFIG["model_dir"]` and "baseline".
            ```python
            baseline_path = args.baseline_path or os.path.join(
                os.path.dirname(__file__), PATH_CONFIG["model_dir"], "baseline"
            )
            ```
    *   `--pretrained_path`: Path to the pretrained model directory.
        ```python
        parser.add_argument("--pretrained_path", type=str, help="Path to pretrained model directory")
        ```
        *   Usage in `main()` if not provided: Defaults to a path constructed using `PATH_CONFIG["model_dir"]` and "pretrained".
            ```python
            pretrained_path = args.pretrained_path or os.path.join(
                os.path.dirname(__file__), PATH_CONFIG["model_dir"], "pretrained"
            )
            ```
    *   `--baseline_episode`: Episode number for the baseline model.
        ```python
        parser.add_argument("--baseline_episode", type=int, default=TRAINING_CONFIG["num_episodes"], 
                            help="Episode number for baseline model")
        ```
    *   `--pretrained_episode`: Episode number for the pretrained model.
        ```python
        parser.add_argument("--pretrained_episode", type=int, default=TRAINING_CONFIG["num_episodes"], 
                            help="Episode number for pretrained model")
        ```

*   **Evaluation Modes (Boolean Flags):**
    These flags determine which evaluation routines are executed. If none are specified, a default set is enabled.
    ```python
    parser.add_argument("--evaluate_baseline", action="store_true", help="Evaluate baseline agent")
    parser.add_argument("--evaluate_pretrained", action="store_true", help="Evaluate pretrained agent")
    parser.add_argument("--compare", action="store_true", help="Compare baseline and pretrained agents")
    parser.add_argument("--cold_start", action="store_true", help="Evaluate cold-start performance")
    parser.add_argument("--performance_over_time", action="store_true", help="Evaluate performance over time")
    ```
    *   Default enabling logic in `src/evaluate.py`:
        ```python
        if not (args.evaluate_baseline or args.evaluate_pretrained or args.compare or 
                args.cold_start or args.performance_over_time):
            args.evaluate_baseline = True
            args.evaluate_pretrained = True
            args.compare = True
            args.cold_start = True
            # args.performance_over_time is NOT enabled by default in this block
        ```
        *(Note: The `main` function in `src/main.py` which can call the evaluation script has a similar block that enables `args.performance_over_time = True` as well by default if no specific eval option is passed to it for the evaluation subprocess).* 

*   **Evaluation Parameters:**
    *   `--num_episodes`: Number of evaluation episodes for general performance and comparison.
        ```python
        parser.add_argument("--num_episodes", type=int, default=EVAL_CONFIG["num_eval_episodes"], 
                            help="Number of evaluation episodes")
        ```
        This value is passed as `num_episodes` to `Evaluator.evaluate_agent` and `Evaluator.compare_agents`.
    *   `--cold_start_episodes`: Number of episodes for cold-start evaluation.
        ```python
        parser.add_argument("--cold_start_episodes", type=int, default=10, 
                            help="Number of episodes for cold-start evaluation")
        ```
        Used as `num_episodes` in `evaluate_cold_start_performance` (`src/evaluate.py`).
    *   `--cold_start_steps`: Number of steps per episode for cold-start evaluation.
        ```python
        parser.add_argument("--cold_start_steps", type=int, default=10, 
                            help="Number of steps per episode for cold-start evaluation")
        ```
        Used as `num_steps` in `evaluate_cold_start_performance` (`src/evaluate.py`).
    *   `--eval_interval`: Interval for performance over time evaluation checkpoints.
        ```python
        parser.add_argument("--eval_interval", type=int, default=100, 
                            help="Interval for performance over time evaluation")
        ```
        Used in `src/evaluate.py` to generate the `steps` list: `list(range(0, args.num_episodes, args.eval_interval))`.
    *   `--use_llm_simulator`: Boolean flag to use `LLMSimulatorEnv`.
        ```python
        parser.add_argument("--use_llm_simulator", action="store_true", 
                            help="Use LLM simulator for evaluation")
        ```

*   **Other Parameters:**
    *   `--seed`: Random seed for reproducibility.
        ```python
        parser.add_argument("--seed", type=int, default=100, help="Random seed")
        ```
        Used in `set_seed()` and passed to environment constructors in `src/evaluate.py`.
    *   `--device`: Computation device ("cuda" or "cpu").
        ```python
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                            help="Device to run evaluation on")
        ```
        Used for loading models: `device = torch.device(args.device)`.

**Core Evaluator Parameters (`src/utils/evaluator.py` - `Evaluator` class methods):**

*   `max_steps_per_episode` in `Evaluator.evaluate_agent`:
    *   **Definition:**
        ```python
        def evaluate_agent(self, agent: DynaQAgent, env: JobRecommendationEnv, 
                           applicant_ids: List[str], num_episodes: int = EVAL_CONFIG["num_eval_episodes"],
                           max_steps_per_episode: int = 100) -> Dict[str, Any]:
        ```
    *   **Usage:** This parameter defaults to `100` and is used directly by `evaluate_agent`. When `evaluate_agent` is called from `compare_agents` or `evaluate_model_performance` (in `src/evaluate.py`), this default of `100` is used as it's not overridden in those call paths. For `evaluate_performance_over_time`, the `Evaluator`'s `evaluate_agent` is also called with this default of 100 for `max_steps_per_episode` during its mini-evaluation runs.

**Environment and Agent Specifics:**

*   **Applicant IDs (`sample_applicant_ids`):**
    *   **Source (`src/evaluate.py` - `main` function):**
        ```python
        sample_applicant_ids = [f"applicant_{i}" for i in range(100)]
        ```
    *   This list is passed to various evaluation functions.
*   **Agent Evaluation Mode:**
    *   The agent consistently uses a greedy policy during evaluation phases by setting `eval_mode=True` in `select_action` calls.
    *   **Example (`src/utils/evaluator.py` - `Evaluator.evaluate_agent`):
        ```python
        action_idx, _ = agent.select_action(state, available_actions, eval_mode=True)
        ```

## 8. Output

The evaluation process generates several types of output, primarily managed by logging configurations, and explicit saving functions within the `Evaluator` class.

1.  **Log Files**: Detailed logs of the evaluation steps and results are written to `evaluation.log` (and to the console).
    *   **Configuration (`src/evaluate.py`):**
        ```python
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("evaluation.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        ```
    *   **Example Log Messages (`src/utils/evaluator.py` - `Evaluator.evaluate_agent`):**
        ```python
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        # ...
        logger.info(f"Evaluated {episode+1}/{num_episodes} episodes. " \
                    f"Avg reward: {np.mean(episode_rewards):.4f}")
        # ...
        logger.info(f"Evaluation complete. Average reward: {avg_reward:.4f}")
        ```

2.  **JSON Files**: Structured results are saved to disk by the `Evaluator.save_results` method.
    *   **Called from (`src/utils/evaluator.py` - e.g., `Evaluator.compare_agents`):**
        ```python
        self.save_results(comparison, "agent_comparison.json")
        ```
    *   **Implementation (`src/utils/evaluator.py` - `Evaluator.save_results`):**
        ```python
        def save_results(self, results: Dict[str, Any], filename: str) -> None:
            # ... (serializable conversion logic) ...
            serializable_results = convert_to_serializable(results)
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            logger.info(f"Results saved to {filepath}")
        ```
        The `results_dir` is initialized in `Evaluator.__init__`:
        ```python
        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), 
                                                         PATH_CONFIG["results_dir"])
        os.makedirs(self.results_dir, exist_ok=True)
        ```

3.  **Visualization Plots**: Image files (e.g., PNGs) of various plots are generated by the `Visualizer` class methods (e.g., `plot_evaluation_results`, `plot_reward_histogram`, `plot_learning_curves`). These are typically saved within the same `results_dir` by the `Visualizer`.
    *   **Example Call (`src/evaluate.py` - `evaluate_cold_start_performance`):**
        ```python
        visualizer.plot_evaluation_results(
            baseline_rewards=baseline_episode_rewards,
            pretrained_rewards=pretrained_episode_rewards,
            title="Cold-Start Performance Comparison",
            filename="cold_start_comparison.png"
        )
        ```
    *   The `Visualizer` is initialized with this `results_dir`:
        ```python
        # src/utils/evaluator.py - Evaluator.__init__()
        self.visualizer = Visualizer(results_dir=self.results_dir)
        ```

This comprehensive output allows for thorough analysis, reporting, and archiving of evaluation findings directly based on the implemented code paths. 