**Note on Scope:** This document describes the training process for the Dyna-Q based job recommendation agent, primarily focusing on the implementation within `src/training/agent.py`. It also provides specific context on how this agent class is utilized to train two distinct agent instances (`agent_cosine` and `agent_hybrid`) with different strategies within the `neural_dyna_q_notebook.py` experimental setup.

# Dyna-Q Agent Training: Technical Description

This document provides a detailed technical explanation of the training process for the Dyna-Q based job recommendation agent, grounding explanations in direct quotes from the implementation to ensure fidelity and traceability.

## 1. Training Objectives

The primary objective is to train a specialized reinforcement learning agent for each target applicant. The agent's goal is to learn an optimal policy to recommend jobs (actions) to an applicant (state) that maximize cumulative discounted reward. This is stated in the `DynaQAgent` class docstring:

```python
# From: src/training/agent.py (DynaQAgent class docstring)
"""
Implements the Dyna-Q algorithm, which combines:
1. Direct reinforcement learning (Q-learning)
2. Model-based planning (world model learning and simulation)
"""
```

The system focuses on specializing an agent for a single target applicant. In the `neural_dyna_q_notebook.py` setup, two such specialized agents are trained independently for the same target applicant but using different reward strategies and training procedures:
1.  `agent_cosine`: Trained using rewards derived purely from cosine similarity.
2.  `agent_hybrid`: Trained using a hybrid strategy involving LLM-generated feedback and potentially cosine similarity components.

## 2. Agent Architecture

The `DynaQAgent` class (`src/training/agent.py`) defines the architecture. In `neural_dyna_q_notebook.py`, two separate instances of this architecture (each with its own Q-Network, World Model, and Target Q-Network) are created: one for `agent_cosine` and one for `agent_hybrid`.

Each agent instance comprises three main neural network components, initialized within its `DynaQAgent.__init__` method:

### 2.1. Q-Network ($Q(s, a; \theta_Q)$)
-   **Purpose**: Approximates the action-value function $Q(s, a)$.
-   **Initialization Snippet** (from `src/training/agent.py` - `DynaQAgent.__init__`):
    ```python
    # Example for one agent, repeated for agent_cosine and agent_hybrid with their respective networks
    self.q_network = q_network if q_network is not None else QNetwork(
        state_dim=state_dim, # from MODEL_CONFIG
        action_dim=action_dim, # from MODEL_CONFIG
        hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
        activation=MODEL_CONFIG["q_network"]["activation"]
    ).to(self.device)
    ```

### 2.2. World Model ($M(s, a; \theta_M)$)
-   **Purpose**: Learns to predict the immediate reward $R(s,a)$ given a state $s$ and action $a$.
-   **Initialization Snippet** (from `src/training/agent.py` - `DynaQAgent.__init__`, adapted for notebook context):
    ```python
    # Example for one agent
    input_dim_wm = MODEL_CONFIG["q_network"]["state_dim"] + MODEL_CONFIG["q_network"]["action_dim"]
    self.world_model = world_model if world_model is not None else WorldModel(
        input_dim=input_dim_wm, # Calculated as state_dim + action_dim
        hidden_dims=MODEL_CONFIG["world_model"]["hidden_dims"],
        dropout_rate=MODEL_CONFIG["world_model"]["dropout_rate"],
        activation=MODEL_CONFIG["world_model"]["activation"]
    ).to(self.device)
    ```

### 2.3. Target Q-Network ($Q'(s, a; \theta_{Q'})$)
-   **Purpose**: Provides a stable target for Q-Network updates.
-   **Initialization Snippet** (from `src/training/agent.py` - `DynaQAgent.__init__`):
    ```python
    # Example for one agent
    if target_network is not None:
        self.target_network = target_network
    else:
        self.target_network = QNetwork(
            state_dim=MODEL_CONFIG["q_network"]["state_dim"],
            action_dim=MODEL_CONFIG["q_network"]["action_dim"],
            hidden_dims=MODEL_CONFIG["q_network"]["hidden_dims"],
            dropout_rate=MODEL_CONFIG["q_network"]["dropout_rate"],
            activation=MODEL_CONFIG["q_network"]["activation"]
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict()) # Copied from the main q_network
    ```

## 3. Decision-Making Process (Action Selection)

The `DynaQAgent.select_action` method, used by both `agent_cosine` and `agent_hybrid`, selects actions using an $\epsilon$-greedy policy:

```python
# From: src/training/agent.py (DynaQAgent.select_action method)
epsilon_to_use = 0.0 if eval_mode else self.epsilon # self.epsilon is an agent attribute

if random.random() > epsilon_to_use:
    # Exploit: Select action with highest Q-value
    with torch.no_grad():
        # Assuming available_actions is a list of tensors:
        q_values_list = []
        for action_tensor in available_actions:
            action_tensor = action_tensor.to(self.device)
            # Ensure state is on the correct device and appropriately shaped
            current_state_tensor = state.to(self.device).unsqueeze(0) if state.dim() == 1 else state.to(self.device)
            action_tensor_unsqueezed = action_tensor.unsqueeze(0) if action_tensor.dim() == 1 else action_tensor
            
            q_value = self.q_network(current_state_tensor, action_tensor_unsqueezed)
            q_values_list.append(q_value.item())
        
        max_idx = np.argmax(q_values_list)
        selected_action_tensor = available_actions[max_idx]
    return max_idx, selected_action_tensor
else:
    # Explore: Select random action
    idx = random.randrange(len(available_actions))
    return idx, available_actions[idx]
```
Note: The `select_action` method in `src/training/agent.py` also contains a commented-out vectorized computation path. The snippet above reflects the primary loop-based approach used if `available_actions` is a list of tensors, common in the notebook.

The exploration rate `self.epsilon` is decayed over time (this is primarily relevant for `agent_hybrid` during its online training):
```python
# From: src/training/agent.py (DynaQAgent.update_epsilon method)
self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

## 4. Learning Process

The learning process within any `DynaQAgent` instance involves several interconnected components. How these are utilized differs slightly between the training of `agent_cosine` (which uses a static dataset and the `.pretrain()` method) and `agent_hybrid` (which uses online interaction with the environment via the `.train()` and `.train_step()` methods) in `neural_dyna_q_notebook.py`.

### 4.1. Environment Interaction & Experience Replay
Transitions $(s_t, a_t, r_t, s_{t+1})$ are stored in a replay buffer (`self.replay_buffer`).
-   **Storage (`src/training/agent.py` - `DynaQAgent.store_experience`):**
    ```python
    self.replay_buffer.push(state, action, reward, next_state)
    ```
-   **Generation for `agent_hybrid` (Online Learning in `neural_dyna_q_notebook.py` Block 10, via `DynaQAgent.train_step`):
    ```python
    # From: src/training/agent.py (DynaQAgent.train_step method)
    # ... action selection ...
    next_state, reward, done, info = env.step(action_idx_for_env)
    self.store_experience(state, action_vector, reward, next_state)
    ```
-   **Data for `agent_cosine` (Offline Learning in `neural_dyna_q_notebook.py` Block 9):
    `agent_cosine` is trained using the `.pretrain()` method, which is fed a pre-generated dataset (`states_cosine`, `actions_cosine`, `rewards_cosine`, `next_states_cosine`). It does not use its own replay buffer or `store_experience` during this specific training phase.

### 4.2. Direct Reinforcement Learning (Q-Network Update)
Both agents update their respective Q-Networks using the `DynaQAgent.update_q_network` method. The target $y_j$ for a transition $(s_j, a_j, r_j, s_{j+1})$ is computed as $r_j + \gamma Q'(s_{j+1}, a_j; \theta_{Q'})$.
-   **For `agent_hybrid`:** Called from `train_step` with batches from its replay buffer.
-   **For `agent_cosine`:** Called from `pretrain` with batches from the provided static dataset.
```python
# From: src/training/agent.py (DynaQAgent.update_q_network method)
# Compute current Q values
current_q_values = self.q_network(states, actions) # Q(s_j, a_j)

# Compute next Q values (target)
with torch.no_grad():
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
    # Calculate target Q values
    # next_q_values here are r_j + gamma * Q'(s_{j+1}, a_j)
    next_q_values = rewards + self.gamma * self.target_network(next_states, actions)

# Compute loss
loss = self.q_criterion(current_q_values, next_q_values) # MSE Loss

# Update Q-network
self.q_optimizer.zero_grad()
loss.backward()
self.q_optimizer.step()
```

### 4.3. World Model Learning
Both agents update their respective World Models using `DynaQAgent.update_world_model` to predict rewards:
-   **For `agent_hybrid`:** Called from `train_step` with batches from its replay buffer.
-   **For `agent_cosine`:** Called from `pretrain` with batches from the provided static dataset.
```python
# From: src/training/agent.py (DynaQAgent.update_world_model method)
# Predict rewards using the world model
predicted_rewards = self.world_model(states, actions) # M(s_j, a_j)

# Compute loss
loss = self.world_criterion(predicted_rewards, rewards) # MSE Loss with actual r_j

# Update world model
self.world_optimizer.zero_grad()
loss.backward()
self.world_optimizer.step()
```

### 4.4. Planning (Model-Based Learning)
The `DynaQAgent.plan` method allows the agent to perform planning steps using its world model. 
-   **For `agent_hybrid`:** This is called within `train_step` if `self.tensor_cache` is available (which it should be, as it's passed during init) and `len(self.replay_buffer)` is sufficient (i.e., `TRAINING_CONFIG["min_replay_buffer_size"]` is met, though the `plan` method itself checks for `self.batch_size`).
-   **For `agent_cosine`:** The `plan` method is not explicitly called during its training via the `.pretrain()` method in `neural_dyna_q_notebook.py` Block 9. The `.pretrain()` method focuses on updating the Q-network and world model directly from the provided dataset.

Standard planning approach snippet (relevant for `agent_hybrid`):
```python
# From: src/training/agent.py (DynaQAgent.plan method, standard planning part)
# This snippet shows the logic if not using the optimized tensor cache planning path.
# For simplicity, we'll use a batch from the buffer
states, actions, rewards_from_buffer, _ = self.replay_buffer.sample(self.batch_size)

# Predict rewards using world model
states = states.to(self.device)
actions = actions.to(self.device)
predicted_rewards = self.world_model(states, actions) # r_hat_sim = M(s_sim, a_sim)

# Compute targets for Q-learning
with torch.no_grad():
    next_states = states # Assumes static state for reward prediction model: s'_sim = s_sim
    next_q_val_target = self.target_network(next_states, actions) # Q'(s'_sim, a_sim)
    
    # Ensure predicted_rewards has the correct shape (batch_size, 1)
    if predicted_rewards.dim() == 1:
        predicted_rewards = predicted_rewards.unsqueeze(1)
    
targets = predicted_rewards + self.gamma * next_q_val_target # y_hat_sim

current_q_values = self.q_network(states, actions) # Q(s_sim, a_sim)
loss = self.q_criterion(current_q_values, targets) # MSE Loss

self.q_optimizer.zero_grad()
loss.backward()
self.q_optimizer.step()
# total_loss += loss.item() is part of the loop in plan()
```

### 4.5. Target Network Update
For both agents, the Target Q-Network is periodically updated with the weights of their Q-Network:
-   **For `agent_hybrid`:** Called from `train_step` based on `self.target_update_freq`.
-   **For `agent_cosine`:** Called from `pretrain` based on `self.target_update_freq`.
```python
# From: src/training/agent.py (DynaQAgent.update_target_network method)
self.target_network.load_state_dict(self.q_network.state_dict())
```

## 5. Reward Mechanisms & Training Strategies

The `neural_dyna_q_notebook.py` script sets up two distinct training pipelines:

-   **Cosine Similarity Strategy (for `agent_cosine`):** 
    `agent_cosine` is trained exclusively using rewards derived from cosine similarity between applicant and job embeddings. This training occurs in Block 9 of the notebook, using the `agent_cosine.pretrain()` method with a statically generated dataset where rewards are pre-calculated cosine similarities.
    *   **Data Generation Snippet (`neural_dyna_q_notebook.py`, Block 9 context for `agent_cosine`):**
        ```python
        # From: neural_dyna_q_notebook.py (Block 9)
        initial_state_tensor_cosine = tensor_cache.get_applicant_state(target_candidate_id)
        all_rewards_cosine = tensor_cache.calculate_cosine_similarities(initial_state_tensor_cosine)
        if STRATEGY_CONFIG["cosine"]["scale_reward"]:
            all_rewards_cosine = (all_rewards_cosine + 1) / 2 # Scale from [-1, 1] to [0, 1]
        
        cosine_training_data = []
        # indices_to_use_cosine is defined based on num_pretraining_samples
        for cache_job_idx_cosine in tqdm(indices_to_use_cosine, desc="Generating agent_cosine training data"):
            action_tensor_cosine = tensor_cache.get_job_vector_by_index(cache_job_idx_cosine)
            reward_cosine = all_rewards_cosine[cache_job_idx_cosine].item()
            next_state_tensor_cosine = initial_state_tensor_cosine 
            cosine_training_data.append((initial_state_tensor_cosine, action_tensor_cosine, reward_cosine, next_state_tensor_cosine))
        ```
    *   **Training Call (`neural_dyna_q_notebook.py`, Block 9):**
        ```python
        # From: neural_dyna_q_notebook.py (Block 9)
        agent_cosine_training_metrics = agent_cosine.pretrain(
            states=states_cosine,         # torch.stack of cosine_training_data[0]
            actions=actions_cosine,       # torch.stack of cosine_training_data[1]
            rewards=rewards_cosine,         # torch.tensor of cosine_training_data[2]
            next_states=next_states_cosine, # torch.stack of cosine_training_data[3]
            num_epochs=num_pretraining_epochs, 
            batch_size=cosine_training_batch_size
        )
        ```

-   **Hybrid LLM Strategy (for `agent_hybrid`):** 
    `agent_hybrid` is trained using the `HybridEnv` (if `reward_strategy` config is "hybrid"), which combines LLM-simulated feedback with cosine similarity, or purely LLM feedback if configured. This training occurs in Block 10 of the notebook via the `agent_hybrid.train()` method, which involves online interaction with the environment.
    *   **Environment Used (`neural_dyna_q_notebook.py`, Block 5 & 7 context for `agent_hybrid`):
        ```python
        # From: neural_dyna_q_notebook.py (Block 5 & 7)
        # Assuming reward_strategy is "hybrid"
        env = HybridEnv(
            tensor_cache=tensor_cache,
            reward_scheme=STRATEGY_CONFIG["llm"]["response_mapping"],
            cosine_weight=STRATEGY_CONFIG["hybrid"]["initial_cosine_weight"],
            random_seed=ENV_CONFIG["random_seed"]
        )
        # LLM model and tokenizer are set up on this 'env' instance
        # env = env.setup_llm(llm_model, tokenizer, device="auto") 
        ```
    *   **Training Call (`neural_dyna_q_notebook.py`, Block 10):
        ```python
        # From: neural_dyna_q_notebook.py (Block 10)
        training_metrics_hybrid = agent_hybrid.train(
            env=env, # The HybridEnv instance
            num_episodes=num_episodes_hybrid,
            max_steps_per_episode=max_steps_per_episode_hybrid,
            applicant_ids=[target_candidate_id]
        )
        ```
    *   The `cosine_weight` for the hybrid strategy can be annealed, managed by `DynaQAgent.update_cosine_weight()` called within `train_step`:
        ```python
        # From: src/training/agent.py (DynaQAgent.update_cosine_weight method)
        if self.cosine_annealing or self.training_strategy != "hybrid": # Correction: Should apply if annealing AND hybrid
             return # Or: if not (self.cosine_annealing and self.training_strategy == "hybrid"): return

        if self.steps_done < self.cosine_anneal_steps:
            progress = self.steps_done / self.cosine_anneal_steps
            # The original code for self.cosine_weight update was: 
            # self.cosine_weight = self.cosine_weight - (progress * (self.cosine_weight - self.cosine_anneal_end))
            # A more standard linear interpolation from initial (e.g., 1.0) to end (e.g., 0.0) is:
            # initial_weight = 1.0 # Assuming it starts at 1.0 for hybrid
            # self.cosine_weight = initial_weight - progress * (initial_weight - self.cosine_anneal_end)
            # The existing code seems to anneal from the current cosine_weight, which might be fine if cosine_weight is initialized appropriately.
            # For clarity, sticking to the original code structure for the quote:
            self.cosine_weight = self.cosine_weight - (progress * (self.cosine_weight - self.cosine_anneal_end))
            # logger.debug(f"Updated cosine_weight to {self.cosine_weight}")
        ```

## 6. Optimization Techniques

-   **Optimizer**: Adam optimizer is used, initialized in `DynaQAgent.__init__`:
    ```python
    # From: src/training/agent.py (DynaQAgent.__init__)
    self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
    self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.lr)
    ```
-   **Loss Functions**: MSELoss is used for both Q-Network and World Model:
    ```python
    # From: src/training/agent.py (DynaQAgent.__init__)
    self.q_criterion = nn.MSELoss()
    self.world_criterion = nn.MSELoss()
    ```
-   **Batch Size**: Updates are performed on mini-batches, with `self.batch_size` used in `replay_buffer.sample()` and loops.

## 7. Key Hyperparameters
Many hyperparameters are initialized in `DynaQAgent.__init__` from `TRAINING_CONFIG` (defined in `config/config.py`) or `kwargs` passed during agent instantiation. The `neural_dyna_q_notebook.py` script utilizes these configurations for both `agent_cosine` and `agent_hybrid`.

**Common Agent Hyperparameters (from `src/training/agent.py` - `DynaQAgent.__init__`):**
-   `gamma` (Discount factor): `self.gamma = kwargs.get("gamma", TRAINING_CONFIG["gamma"])`
-   `lr` (Learning rate for Q-Network and World Model optimizers): `self.lr = kwargs.get("lr", TRAINING_CONFIG["lr"])`
-   `batch_size` (for replay buffer sampling and pretrain batching): `self.batch_size = kwargs.get("batch_size", TRAINING_CONFIG["batch_size"])`
-   `target_update_freq` (how often to copy Q-Network weights to Target Q-Network): `self.target_update_freq = kwargs.get("target_update_freq", TRAINING_CONFIG["target_update_freq"])`
-   `planning_steps` (number of model-based updates per real interaction for `agent_hybrid`):
    ```python
    self.planning_steps = kwargs.get("planning_steps", TRAINING_CONFIG["planning_steps"])
    ```
-   Epsilon parameters for $\epsilon$-greedy exploration (primarily for `agent_hybrid` during online training):
    ```python
    self.epsilon_start = kwargs.get("epsilon_start", TRAINING_CONFIG["epsilon_start"])
    self.epsilon_end = kwargs.get("epsilon_end", TRAINING_CONFIG["epsilon_end"])
    self.epsilon_decay = kwargs.get("epsilon_decay", TRAINING_CONFIG["epsilon_decay"])
    ```

**Specific Hyperparameters from `neural_dyna_q_notebook.py` context:**

*   **For `agent_cosine` training (Block 9):**
    *   Number of samples for static dataset generation: `num_pretraining_samples = TRAINING_CONFIG["pretraining"]["num_samples"]`
    *   Number of training epochs over the static dataset: `num_pretraining_epochs = TRAINING_CONFIG["pretraining"]["num_epochs"]`
    *   Batch size for this training: `cosine_training_batch_size = TRAINING_CONFIG["batch_size"]` (can be set distinctly if desired).

*   **For `agent_hybrid` training (Block 10):**
    *   Number of training episodes: `num_episodes_hybrid = TRAINING_CONFIG["num_episodes"]`
    *   Max steps per episode: `max_steps_per_episode_hybrid = TRAINING_CONFIG["max_steps_per_episode"]`
    *   Hybrid strategy specific (from `STRATEGY_CONFIG["hybrid"]` in `config/config.py`, used by `HybridEnv` and potentially `DynaQAgent` for `cosine_weight` annealing):
        *   `initial_cosine_weight`
        *   Cosine annealing parameters in `DynaQAgent`: `self.cosine_annealing`, `self.cosine_anneal_steps`, `self.cosine_anneal_end`.

Other crucial parameters (e.g., network architectures for QNetwork and WorldModel) are defined in `MODEL_CONFIG` within `config/config.py` and used during the instantiation of these networks for both `agent_cosine` and `agent_hybrid`.

## 8. Training Loop Structures (from `neural_dyna_q_notebook.py`)

The `neural_dyna_q_notebook.py` script orchestrates two distinct training procedures:

### 8.1. Training `agent_cosine` (Notebook Block 9)
This agent is trained using a pre-generated static dataset where rewards are based on cosine similarity. The `agent_cosine.pretrain()` method is used.

*   **Agent Initialization (`neural_dyna_q_notebook.py`, Block 9):**
    ```python
    # From: neural_dyna_q_notebook.py (Block 9)
    print("Initializing agent_cosine for COSINE strategy training...")
    q_network_cosine = QNetwork(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        # ... other QNetwork params ...
    ).to(device)
    world_model_cosine = WorldModel(
        input_dim=MODEL_CONFIG["world_model"]["input_dim"],
        # ... other WorldModel params ...
    ).to(device)
    agent_cosine = DynaQAgent(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        q_network=q_network_cosine,
        world_model=world_model_cosine,
        training_strategy="cosine",
        device=device,
        target_applicant_id=target_candidate_id,
        tensor_cache=tensor_cache
    )
    agent_cosine.planning_steps = TRAINING_CONFIG['planning_steps'] 
    agent_cosine.batch_size = TRAINING_CONFIG["batch_size"] # or cosine_training_batch_size
    ```

*   **Data Generation for Cosine Rewards (`neural_dyna_q_notebook.py`, Block 9):**
    ```python
    # From: neural_dyna_q_notebook.py (Block 9)
    initial_state_tensor_cosine = tensor_cache.get_applicant_state(target_candidate_id)
    all_rewards_cosine = tensor_cache.calculate_cosine_similarities(initial_state_tensor_cosine)
    if STRATEGY_CONFIG["cosine"]["scale_reward"]:
        all_rewards_cosine = (all_rewards_cosine + 1) / 2
    
    cosine_training_data = []
    # num_samples_to_generate_cosine and indices_to_use_cosine are defined earlier in the notebook block
    for cache_job_idx_cosine in tqdm(indices_to_use_cosine, desc="Generating agent_cosine training data"):
        action_tensor_cosine = tensor_cache.get_job_vector_by_index(cache_job_idx_cosine)
        reward_cosine = all_rewards_cosine[cache_job_idx_cosine].item()
        next_state_tensor_cosine = initial_state_tensor_cosine 
        cosine_training_data.append((initial_state_tensor_cosine, action_tensor_cosine, reward_cosine, next_state_tensor_cosine))
    
    states_cosine = torch.stack([d[0] for d in cosine_training_data]).to(device)
    actions_cosine = torch.stack([d[1] for d in cosine_training_data]).to(device)
    rewards_cosine = torch.tensor([d[2] for d in cosine_training_data], dtype=torch.float32, device=device)
    next_states_cosine = torch.stack([d[3] for d in cosine_training_data]).to(device)
    ```

*   **Training Call (`neural_dyna_q_notebook.py`, Block 9, using `agent_cosine.pretrain` from `src/training/agent.py`):**
    ```python
    # From: neural_dyna_q_notebook.py (Block 9)
    agent_cosine_training_metrics = agent_cosine.pretrain(
        states=states_cosine,
        actions=actions_cosine,
        rewards=rewards_cosine,
        next_states=next_states_cosine,
        num_epochs=num_pretraining_epochs, # Variable from notebook
        batch_size=cosine_training_batch_size # Variable from notebook
    )
    ```
    The `agent_cosine.pretrain` method internally loops through epochs and batches, calling `update_q_network` and `update_world_model` (see Section 4 for details on these methods).

### 8.2. Training `agent_hybrid` (Notebook Block 10)
This agent is trained online through interaction with the environment (e.g., `HybridEnv`), using the `agent_hybrid.train()` method, which in turn calls `agent_hybrid.train_step()` for each episode.

*   **Agent Initialization (within multi-experiment loop in `neural_dyna_q_notebook.py`, Block 10):**
    ```python
    # From: neural_dyna_q_notebook.py (Block 10, inside experiment loop)
    # print(f"Initializing new agent_hybrid instance for experiment {exp_idx+1}...")
    q_network_hybrid_exp = QNetwork(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        # ... other QNetwork params ...
    ).to(device)
    world_model_hybrid_exp = WorldModel(
        input_dim=MODEL_CONFIG["world_model"]["input_dim"],
        # ... other WorldModel params ...
    ).to(device)
    agent_hybrid = DynaQAgent(
        state_dim=MODEL_CONFIG["q_network"]["state_dim"],
        action_dim=MODEL_CONFIG["q_network"]["action_dim"],
        q_network=q_network_hybrid_exp,
        world_model=world_model_hybrid_exp,
        training_strategy=reward_strategy, # This is 'hybrid' in the notebook context
        device=device,
        target_applicant_id=target_candidate_id,
        tensor_cache=tensor_cache
    )
    agent_hybrid.planning_steps = TRAINING_CONFIG['planning_steps']
    agent_hybrid.batch_size = TRAINING_CONFIG['batch_size']
    ```

*   **Overall Training Orchestration (`agent_hybrid.train` from `src/training/agent.py`, called in Notebook Block 10):**
    ```python
    # From: src/training/agent.py (DynaQAgent.train method)
    # target_applicant_id is set based on applicant_ids[0]
    # self.target_applicant_id = target_applicant_id
    with tqdm(total=num_episodes, desc="Training Dyna-Q Agent") as pbar:
        for episode in range(num_episodes):
            # ... (Strategy switching logic for hybrid, if applicable, though notebook sets it at init) ...
            episode_metrics = self.train_step(env, target_applicant_id, max_steps_per_episode)
            
            # History update, pbar update, logging occur here
            # ...
            # Periodic evaluation call
            if eval_frequency > 0 and episode % eval_frequency == 0:
                eval_reward = self.evaluate(env, 5, max_steps_per_episode, [target_applicant_id])
            # Periodic model saving
            if model_dir and save_frequency > 0 and episode > 0 and episode % save_frequency == 0:
                self.save_models(model_dir, episode)
    # Final evaluation and model saving follow
    ```

*   **Per-Episode Training Logic (`agent_hybrid.train_step` from `src/training/agent.py`):**
    ```python
    # From: src/training/agent.py (DynaQAgent.train_step method, simplified)
    # metrics = { ... } initialised
    state = env.reset(applicant_id) # applicant_id is target_applicant_id for agent_hybrid
    # if self.training_strategy == "hybrid" and isinstance(env, HybridEnv):
    #     env.set_cosine_weight(self.cosine_weight)

    for step_num in range(max_steps): # max_steps is max_steps_per_episode_hybrid
        valid_action_indices = env.get_valid_actions()
        if not valid_action_indices:
            break # No valid actions
        available_actions_list = [env.get_action_vector(idx) for idx in valid_action_indices]
        if not available_actions_list:
            break # Could not get action vectors
        
        # Action selection (eval_mode=False by default in train_step context)
        selected_idx_in_sample, chosen_action_vector = self.select_action(state, available_actions_list)
        
        # Environment step using the index relative to the current sample of actions
        next_state, reward, done, info = env.step(selected_idx_in_sample)
        
        self.store_experience(state, chosen_action_vector, reward, next_state)
        # metrics['episode_reward'] += reward (or similar accumulation)
        
        if len(self.replay_buffer) >= TRAINING_CONFIG["min_replay_buffer_size"]:
            # Sample from replay buffer
            s_batch, a_batch, r_batch, ns_batch = self.replay_buffer.sample(self.batch_size)
            
            q_loss = self.update_q_network(s_batch, a_batch, r_batch, ns_batch)
            # metrics['q_network_loss'] += q_loss
            
            world_loss = self.update_world_model(s_batch, a_batch, r_batch)
            # metrics['world_model_loss'] += world_loss
            
            if self.tensor_cache and self.planning_steps > 0: # Check planning_steps too
                planning_metrics = self.plan(env) # env is passed
                # metrics['planning_loss'] += planning_metrics['planning_loss']
            
            self.steps_done += 1
            if self.steps_done % self.target_update_freq == 0:
                self.update_target_network()
            
            self.update_epsilon()
            if self.training_strategy == "hybrid": # Only update cosine for hybrid
                self.update_cosine_weight()
            
        state = next_state
        if done:
            break
    # Return averaged metrics for the episode
    ```

This revised structure provides a clear distinction between the training procedures for `agent_cosine` and `agent_hybrid` as orchestrated by `neural_dyna_q_notebook.py`.