# Neural Network Model Proposition for Dyna-Q Job Recommender with MongoDB Integration and Comparative Training Strategies

## Overview and Motivation: Tackling the Cold-Start Problem via Diverse Training Signals

Recommender systems using Reinforcement Learning (RL) often struggle with the **cold-start problem**: providing relevant recommendations for new users or upon initial deployment due to a lack of interaction data. This initial phase of random or poorly informed recommendations can deter users.

This project explores using a **Dyna-Q** framework, powered by neural networks and sourcing data from a MongoDB Atlas database (`rl_jobsdb`), to address this. Inspired by recent work using Large Language Models (LLMs) to bootstrap RL agents, our central idea is to **pre-train or train** the agent using simulated reward signals before or during online learning. This aims to equip the agent with an effective initial policy, improving day-one performance.

Crucially, we will investigate and compare **three distinct strategies** for generating the reward signals used to train the agent's Q-network and World Model:

1.  **Cosine Similarity:** Using a simple, direct measure of semantic relevance between applicant and job embeddings.
2.  **LLM Feedback Only:** Relying entirely on a sophisticated LLM user simulator to generate nuanced behavioral feedback.
3.  **Hybrid Approach:** Pre-training with cosine similarity rewards and then fine-tuning with LLM feedback rewards.

By comparing these approaches, we aim to understand the trade-offs between different reward simulation techniques in the context of pre-training RL agents for job recommendation and mitigating the cold-start problem. This document details the MDP formulation, the neural network architecture, data integration, the learning process, the specifics of each training strategy, and the evaluation plan.

## MDP Formulation: Modeling the Job Recommendation Process

We model the task as a Markov Decision Process (MDP):

* **Agent:** The recommendation system.
* **Environment:** The job search context (applicant profile from MongoDB, jobs from MongoDB, feedback mechanism defined by the training strategy).
* **State ($s_t$):** Applicant's context at time $t$, derived from MongoDB.
    * `v_applicant_state, t` = `aggregate_embeddings(` fetched from `candidates_embeddings` for applicant ID `t` `)`. (Likely concatenation of pre-computed hard/soft skill embeddings).
    * *Phase 2:* Will include dynamic history (SASRec).
* **Action Space ($\mathcal{A}$):** Set of all job postings (e.g., in `all_jobs`).
* **Action Selection ($a_t$):**
    1.  Sample candidate jobs $J_{candidates}$ (size $n$) from $\mathcal{A}$.
    2.  Fetch corresponding job vectors $v_{job}$ from `job_embeddings` collection via `pymongo`.
    3.  Estimate $Q(s_t, v_{job})$ for each candidate using the Q-network.
    4.  Select action $a_t$ (job `_id`) using an $\epsilon$-greedy policy based on Q-values.
    5.  Recommend top-$k$ jobs based on Q-values.
* **Reward ($R_{t+1}$):** **Critically, the source of this scalar feedback signal depends on the chosen training strategy (detailed below).**
* **State Transition ($P(s_{t+1}|s_t, a_t)$):** In the current phase with static states, $s_{t+1} = s_t$. Transitions are implicitly handled by the reward generation mechanism (Cosine/LLM Sim) or learned by the World Model.
* **Discount Factor ($\gamma$):** Value between 0 and 1 for future reward weighting.

## Data Integration with MongoDB (`rl_jobsdb`)

Neural network inputs (state and action vectors) are fetched from the `rl_jobsdb` database on MongoDB Atlas using `pymongo`.

* **Connection:** Establish connection to Atlas cluster, select `rl_jobsdb`.
* **State Vector Retrieval (`v_applicant_state`):** Query `candidates_embeddings` by applicant ID. Retrieve and aggregate (e.g., concatenate) embedding arrays into a PyTorch tensor.
* **Action Vector Retrieval (`v_job`):** Query `job_embeddings` by `original_job_id` for candidate jobs. Retrieve and potentially aggregate/combine fields like `job_title_embeddings`, `tech_skills_vectors` into PyTorch tensors.
* **Data Loading Pipeline:** Use PyTorch `Dataset` and `DataLoader` to manage efficient batch fetching and preprocessing using `pymongo`.

## Neural Network Architectures

Two main networks are proposed:

### 1. Q-Network Architecture ($Q(s, a; \theta)$)

* **Purpose:** Estimate the action-value function $Q^*(s, a)$.
    $$
    Q^*(s, a) = \mathbb{E}\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | s_t = s, a_t = a, \pi^* \right]
    $$
* **Input:** Concatenated tensor $[v_{applicant\_state}, v_{job}]$ (shape `(batch_size, state_dim + action_dim)`), with vectors sourced from MongoDB.
* **Architecture (MLP):**
    * Input Layer: Size `state_dim + action_dim`.
    * Hidden Layers: Sequence of `torch.nn.Linear` with `torch.nn.ReLU` (or similar) activations and `torch.nn.Dropout`. Tunable hyperparameters (layer count, neuron count).
    * Output Layer: `torch.nn.Linear(H_last, 1)` (single scalar output, no activation).
* **Output:** Estimated Q-values (shape `(batch_size, 1)`).
* **Target Network ($Q(s, a; \theta^{-})$):** Identical structure, weights $\theta^{-}$ updated periodically by copying from the main network's weights $\theta$.

### 2. World Model Architecture ($\hat{P}(R | s, a)$)

* **Purpose:** Predict the immediate reward $\hat{R}$ for a given state-action pair, used in Dyna-Q planning steps. (State prediction $\hat{s}'$ is trivial currently as $s_{t+1}=s_t$).
* **Input:** Concatenated tensor $[v_{applicant\_state}, v_{job}]$ (shape `(batch_size, state_dim + action_dim)`).
* **Architecture (MLP):**
    * Input Layer: Size `state_dim + action_dim`.
    * Hidden Layers: Similar MLP structure (`Linear` -> `ReLU` -> `Dropout`). Can be separate from or share layers with the Q-Network.
    * Output Layer: `torch.nn.Linear(H_last_model, 1)` predicting the scalar reward $\hat{R}$.
* **Output:** Predicted rewards (shape `(batch_size, 1)`).
* **Training:** Supervised learning on *real* experiences $(s_t, a_t, R_{t+1})$ from $\mathcal{D}_{real}$, minimizing MSE loss: $(\hat{R} - R_{t+1})^2$. **The *target* $R_{t+1}$ used here depends on the active training strategy (Cosine or LLM Feedback).**

## Learning Process: Loss Calculation and Training

The Q-Network learns by minimizing the MSE between its predictions and the Bellman target.

**1. Target Q-Value Calculation ($y_t$):**
For an experience $(s_t, a_t, R_{t+1}, s_{t+1})$ (real or simulated):
$$
y_t = R_{t+1} + \gamma \max_{a' \in \mathcal{A}_{candidates, t+1}} Q(s_{t+1}, v_{job}'; \theta^{-})
$$
* $R_{t+1}$: The **immediate reward** obtained from the specific training strategy (Cosine Sim, LLM Feedback, or model prediction $\hat{R}$ during planning).
* The $\max$ term uses the **target Q-network ($\theta^{-}$) ** for stability to estimate the best value achievable from the next state $s_{t+1}$.

**2. Loss Function (MSE):**
$$
\text{Loss}(\theta) = \mathbb{E}_{(s_t, a_t, R_{t+1}, s_{t+1}) \sim \mathcal{D}} \left[ \left( Q(s_t, v_{job, t}; \theta) - y_t \right)^2 \right]
$$
* Minimizing this loss makes the **main Q-Network's ($\theta$)** output consistent with the observed/simulated rewards and estimated future values.

**3. Optimization:** Use backpropagation and an optimizer (e.g., Adam) to update $\theta$.

## Training Strategies and Reward Simulation

We will implement and compare three distinct approaches for generating the crucial reward signal ($R_{t+1}$) used in both Direct RL updates and for training the World Model.

### Strategy 1: Cosine Similarity Rewards

* **Concept:** Assumes that relevance is directly proportional to the semantic similarity between the applicant and the job in the embedding space.
* **Reward Calculation ($R_{t+1}$):** The reward for recommending job $a_t$ (vector $v_{job, t}$) in state $s_t$ (vector $v_{applicant\_state, t}$) is computed directly as the cosine similarity:
    $$
    R_{t+1} = \frac{v_{applicant\_state, t} \cdot v_{job, t}}{\|v_{applicant\_state, t}\| \|v_{job, t}\|}
    $$
    * Vectors $v_{applicant\_state, t}$ and $v_{job, t}$ are fetched from MongoDB (`candidates_embeddings`, `job_embeddings`) and potentially aggregated (e.g., concatenating hard/soft skill vectors for the applicant, concatenating title/skill vectors for the job).
* **Training Impact:**
    * Provides a computationally inexpensive, deterministic reward signal based purely on embedding alignment.
    * The Q-network learns to recommend jobs whose aggregated embeddings are geometrically close to the applicant's aggregated embedding.
    * The World Model learns to predict this cosine similarity value.
    * This serves as a baseline representing pure semantic matching within the RL framework.

### Strategy 2: LLM Feedback Only Rewards

* **Concept:** Leverages an LLM as a sophisticated user simulator to generate behavioral feedback, aiming for more nuanced and realistic interaction patterns than simple similarity.
* **Reward Calculation ($R_{t+1}$):**
    1.  The recommended job $a_t$ (along with state $s_t$ and context) is presented to the LLM simulator.
    2.  The LLM generates a simulated user action (e.g., 'APPLY', 'SAVE', 'IGNORE').
    3.  This simulated action is mapped to a predefined scalar reward value:
        * Example: $R_{t+1} = 1.0$ if action is 'APPLY', $0.5$ if 'SAVE', $-0.1$ if 'IGNORE'.
* **Training Impact:**
    * The reward signal is based on the LLM's complex internal reasoning and simulation of user behavior.
    * The Q-network learns a policy that optimizes for outcomes deemed positive by the LLM simulator.
    * The World Model learns to predict the reward signal generated by the LLM.
    * Potential for capturing complex preferences but also susceptible to LLM biases, hallucinations, and computational cost.

### Strategy 3: Hybrid (Cosine Pre-training + LLM Fine-tuning)

* **Concept:** A two-phase approach ("scaffolding") attempting to combine the benefits of both strategies: fast initial learning based on semantic relevance, followed by refinement using richer behavioral signals.
* **Reward Calculation ($R_{t+1}$) & Training Phases:**
    * **Phase 1 (Pre-training):**
        * The agent (Q-network $\theta$, World Model $\hat{P}$) is trained using rewards calculated via **Cosine Similarity** as in Strategy 1.
        * This phase quickly establishes a baseline policy based on semantic alignment.
    * **Phase 2 (Fine-tuning):**
        * Training *continues* from the weights learned in Phase 1.
        * The reward signal source is **switched**. $R_{t+1}$ is now generated based on **LLM Feedback** as in Strategy 2.
        * The Q-network and World Model are further updated using these LLM-based rewards.
* **Training Impact:**
    * Aims for rapid initial convergence using the simpler cosine reward, then refines the policy with more complex LLM feedback.
    * Hypothesizes better overall sample efficiency and potentially a more robust final policy compared to using only one strategy.

## Integration with Dyna-Q

The chosen training strategy directly impacts the Dyna-Q loop:

1.  **Direct RL:** The reward $R_{t+1}$ used in the Bellman target $y_t$ comes from the active strategy (Cosine Similarity, LLM Feedback, or the current phase of the Hybrid approach).
2.  **Model Learning:** The World Model $\hat{P}$ is trained to predict the reward $R_{t+1}$ generated by the active strategy.
3.  **Planning:** The World Model $\hat{P}$ generates simulated rewards $\hat{R}$ (which approximate the rewards of the active strategy). These simulated rewards $\hat{R}$ are used in the Bellman target $y_{sim}$ for planning updates.

## Evaluation Strategy: Comparing the Training Approaches

The evaluation will compare the effectiveness of the three training strategies, focusing on how well they address the cold-start problem.

* **Methodology:**
    * Train three separate Dyna-Q agents, one for each strategy (Cosine Only, LLM Only, Hybrid).
    * Compare them against a fourth **Baseline Agent** (Dyna-Q trained from scratch with random initialization, perhaps using a simple reward like +1 for any interaction initially or needing online feedback).
    * Evaluate all agents in a separate, simulated online environment where rewards are generated consistently (e.g., using a held-out reward model or maybe even the LLM simulator itself, applied uniformly *during evaluation only*). Data for states/actions will be fetched from MongoDB.
* **Metrics:** Analyze initial performance (first few episodes/steps) using metrics like Cumulative Reward, Average Reward per Step, and Time-to-Threshold. Long-term convergence will also be compared.
* **Goal:** Determine which strategy (or combination) yields the best initial performance ("warmest" start) and potentially the best overall performance, providing insights into the value of cosine similarity vs. LLM feedback for pre-training/training RL recommenders.

This comprehensive structure outlines the neural model, its data sources, and critically, the different ways rewards will be simulated and compared during training, setting the stage for evaluating these distinct approaches.