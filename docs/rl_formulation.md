# Refined Approach: LLM-Driven Dyna-Q for Enhanced Job Recommendation

## Overview and Motivation: Tackling the Cold-Start Problem

Recommender systems, especially those learning complex user preferences over time using Reinforcement Learning (RL), often face a significant challenge: the **cold-start problem**. When a new user (applicant) interacts with the system, or when the system is first deployed, it lacks historical data to make informed recommendations. Traditional RL agents, starting with randomly initialized policies, require numerous interactions – often involving suboptimal or irrelevant recommendations – before they learn effective strategies. This initial period of poor performance can lead to user frustration and abandonment, hindering the system's ability to gather the very data it needs to improve.

Inspired by recent advancements leveraging Large Language Models (LLMs) in RL and recommendation, this project explores a novel approach to mitigate this cold-start issue in job recommendation. Our core idea is to **pre-train or train** the RL agent's value function (specifically, the Q-network within a Dyna-Q framework) using preferences and reward signals derived from different sources. We hypothesize that leveraging semantic similarity or sophisticated LLM simulations can provide valuable initial signal about job-applicant compatibility, leading to better initial policies.

Specifically, we will investigate and compare **three distinct strategies** for generating the reward signals used to train the agent:

1.  **Cosine Similarity:** Using a direct measure of semantic relevance.
2.  **LLM Feedback Only:** Relying on an LLM user simulator for behavioral feedback.
3.  **Hybrid Approach:** Combining cosine similarity pre-training with LLM feedback fine-tuning.

By comparing these approaches via a rigorous evaluation strategy, we aim to understand the trade-offs and effectiveness of different reward simulation techniques for mitigating the cold-start problem in RL-based job recommenders. The Dyna-Q architecture is chosen to enhance learning efficiency by combining direct interaction experiences with planning steps based on a learned model of the environment dynamics. Data for applicant states and job actions will be sourced from a MongoDB Atlas database (`rl_jobsdb`).

This document outlines the proposed methodology, detailing the MDP formulation, the neural network components, data integration, the learning process, the specifics of each training strategy, and the evaluation plan.

## MDP Formulation: Modeling the Job Recommendation Process

We formally model the job recommendation task as a Markov Decision Process (MDP), defined by the following components:

* **Agent:** The recommendation system itself, responsible for learning and executing the recommendation policy.
* **Environment:** The dynamic context of the job search. This includes the candidate's profile (sourced from MongoDB), the constantly changing pool of available jobs (sourced from MongoDB), and the mechanism that provides feedback on recommendations, which **differs based on the training strategy employed**.
* **State ($s_t$):** A snapshot of the applicant's relevant context at a specific time step $t$. In the current phase, the state focuses on the applicant's qualifications and is represented by a vector `v_applicant_state, t` derived from MongoDB:
    * `v_applicant_state, t` = `concat(`$v_{hard\_skills}, v_{soft\_skills}$`)`
        * $v_{hard\_skills}$: Embedding derived from the applicant's hard skills (retrieved from `candidates_embeddings`).
        * $v_{soft\_skills}$: Embedding derived from the applicant's soft skills (retrieved from `candidates_embeddings`).
    * *Future Work (Phase 2):* The state will be augmented with dynamic interaction history (e.g., sequence of viewed/applied jobs) potentially processed by a model like SASRec.
* **Action Space ($\mathcal{A}$):** The potentially vast set of all job postings available at time $t$ (represented in MongoDB, e.g., `all_jobs`).
* **Action Selection ($a_t$):** Since $\mathcal{A}$ is too large to evaluate exhaustively, we use a practical approach:
    1.  **Candidate Sampling:** At time $t$, sample a manageable subset $J_{candidates} \subset \mathcal{A}$ of size $n$. This might involve querying MongoDB based on heuristics.
    2.  **Action Vector Retrieval:** Fetch the vector representation $v_{job}$ for each job $j \in J_{candidates}$ from the `job_embeddings` collection in MongoDB.
    3.  **Q-Value Estimation:** For each job $j \in J_{candidates}$, use the neural Q-network to estimate its value in the current state: $Q(s_t, v_{job})$.
    4.  **Policy Execution:** Select the action $a_t$ (the specific job `_id` corresponding to $v_{job,t}$) from $J_{candidates}$ based on the estimated Q-values, using an exploration strategy like $\epsilon$-greedy ($1-\epsilon$ probability of choosing the best action, $\epsilon$ probability of choosing randomly from candidates).
    5.  **Recommendation (Output):** Present the top-$k$ jobs from $J_{candidates}$, ranked by their Q-values, to the user/simulator.
* **Reward ($R_{t+1}$):** A numerical feedback signal received after the agent takes action $a_t$. **Crucially, how this reward is generated depends directly on the training strategy being used**, as detailed in the "Training Strategies and Reward Simulation" section below. It serves as the primary learning signal indicating the immediate success or failure of the recommendation $a_t$.
* **State Transition ($P(s_{t+1}|s_t, a_t)$):** Describes the probability of moving to the next state $s_{t+1}$.
    * **Current Phase:** Given the static nature of the state definition ($s_t$ only contains applicant profile embeddings), the state transition is trivial: $s_{t+1} = s_t$. The environment's dynamics are captured primarily within the reward signal.
    * **Learned Model:** The World Model in Dyna-Q learns to predict the reward $\hat{R}$ and implicitly learns this trivial state transition $\hat{s}' = s_t$.
* **Discount Factor ($\gamma$):** A hyperparameter ($0 \le \gamma \le 1$) determining the present value of future rewards in the Q-function calculation.

## Data Integration with MongoDB (`rl_jobsdb`)

The neural networks rely on vector representations fetched from your MongoDB Atlas database (`rl_jobsdb`) using the `pymongo` library.

* **Connection:** The system establishes a connection to your MongoDB Atlas cluster and selects the `rl_jobsdb` database.
* **State Vector Retrieval:** To construct $v_{applicant\_state}$ for an applicant, the system queries `candidates_embeddings`, retrieves pre-computed embedding arrays, and aggregates them (e.g., concatenation) into a PyTorch tensor.
* **Action Vector Retrieval:** For candidate jobs $J_{candidates}$, the system queries `job_embeddings` using `original_job_id`s to fetch relevant embedding fields (e.g., `job_title_embeddings`, `tech_skills_vectors`) and combines them into the $v_{job}$ PyTorch tensor.
* **Data Loading Pipeline:** This MongoDB interaction is managed within a data loading pipeline (e.g., PyTorch `Dataset` / `DataLoader`) for efficient batch preparation.

## The Neural Network's Role: Approximating the Q-Function

At the heart of our RL agent is a neural network acting as a **function approximator** for the **optimal action-value function (Q-function)**, $Q^*(s, a)$. This function predicts the expected total discounted future reward achievable from state $s$ by taking action $a$ and following the optimal policy thereafter:
$$
Q^*(s, a) = \mathbb{E}\left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | s_t = s, a_t = a, \pi^* \right]
$$
Our network, parameterized by $\theta$, learns $Q(s, a; \theta) \approx Q^*(s, a)$.

* **Input:** Concatenated state and action vectors $[v_{applicant\_state}, v_{job}]$, sourced from MongoDB.
* **Output:** A single scalar estimated Q-value.
* **Learning Goal:** Adjust $\theta$ to make the network's output accurately predict the expected return, learning the value of specific jobs for specific applicants.

## Neural Network Architectures

### 1. Q-Network Architecture ($Q(s, a; \theta)$)

* **Purpose:** Estimate Q-values.
* **Input:** Tensor `(batch_size, state_dim + action_dim)` from MongoDB embeddings.
* **Architecture (MLP):** Standard Multi-Layer Perceptron.
    * Input Layer: Size `state_dim + action_dim`.
    * Hidden Layers: Sequence of `torch.nn.Linear` with `torch.nn.ReLU` (or similar) activations and `torch.nn.Dropout` for regularization. The number and size of layers are tunable hyperparameters.
    * Output Layer: `torch.nn.Linear(H_last, 1)` (single scalar, no activation).
* **Output:** Estimated Q-values `(batch_size, 1)`.
* **Target Network ($Q(s, a; \theta^{-})$):** Identical architecture; weights $\theta^{-}$ copied periodically from $\theta$ for stable target calculation.

### 2. World Model Architecture ($\hat{P}(R | s, a)$)

* **Purpose:** Predict immediate reward $\hat{R}$ for Dyna-Q planning. (State prediction $\hat{s}'$ is trivial in this phase).
* **Input:** Concatenated tensor `(batch_size, state_dim + action_dim)`.
* **Architecture (MLP):** Similar MLP structure. Can be separate or share layers with Q-Network.
    * Input Layer: Size `state_dim + action_dim`.
    * Hidden Layers: `Linear` -> `ReLU` -> `Dropout` sequence.
    * Output Layer: `torch.nn.Linear(H_last_model, 1)` predicting scalar reward $\hat{R}$.
* **Output:** Predicted rewards `(batch_size, 1)`.
* **Training:** Supervised learning on *real* experiences $(s_t, a_t, R_{t+1})$, minimizing MSE loss $(\hat{R} - R_{t+1})^2$. The target $R_{t+1}$ depends on the active training strategy.

## Learning the Q-Function: Loss Calculation and Training

The Q-network learns by minimizing the discrepancy (loss) between its predictions and a target value derived from the Bellman equation.

**1. The Bellman Equation and the Target Q-Value ($y_t$):**

The Bellman equation provides the foundation: $Q^*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a')]$. We use this to compute a target $y_t$ for each experience tuple $(s_t, a_t, R_{t+1}, s_{t+1})$:
$$
y_t = R_{t+1} + \gamma \max_{a' \in \mathcal{A}_{candidates, t+1}} Q(s_{t+1}, v_{job}'; \theta^{-})
$$

* $R_{t+1}$: The immediate reward obtained according to the **active training strategy** (Cosine, LLM, or Hybrid phase).
* $\gamma$: The discount factor.
* $\max_{a'} Q(s_{t+1}, v_{job}'; \theta^{-})$: The **stable estimate** (using the target network $\theta^{-}$) of the maximum value achievable from the next state $s_{t+1}$ by choosing the best candidate action $a'$.

The target $y_t$ represents a bootstrapped estimate of the true value of $(s_t, a_t)$, combining observed reward with estimated optimal future value.

**2. The Loss Function: Mean Squared Error (MSE):**

We minimize the squared difference between the Q-network's prediction and the target:
$$
\text{Loss}(\theta) = \mathbb{E}_{(s_t, a_t, R_{t+1}, s_{t+1}) \sim \mathcal{D}} \left[ \left( Q(s_t, v_{job, t}; \theta) - y_t \right)^2 \right]
$$

* **Interpretation:** This Bellman error squared penalizes inaccuracies in the Q-network's current value estimates relative to the observed rewards and subsequent estimated values.
* **Minimization:** Gradient descent adjusts $\theta$ to reduce this error, making $Q(s, a; \theta)$ converge towards $Q^*(s, a)$.
* **Replay Buffer ($\mathcal{D}$):** Experiences are stored in a buffer $\mathcal{D}$ and sampled in mini-batches to compute the expected loss, improving stability.

**3. Optimization:**

Use **backpropagation** to compute gradients $\nabla_\theta \text{Loss}(\theta)$ and update $\theta$ with an optimizer like Adam or RMSprop.

## Training Strategies and Reward Simulation

A core part of this project is comparing how different methods of simulating the reward signal $R_{t+1}$ impact the agent's learning and initial performance.

### Strategy 1: Cosine Similarity Rewards

* **Concept:** Reward is based on the direct semantic similarity between applicant and job embeddings.
* **Reward Calculation ($R_{t+1}$):**
    $$
    R_{t+1} = \frac{v_{applicant\_state, t} \cdot v_{job, t}}{\|v_{applicant\_state, t}\| \|v_{job, t}\|}
    $$
    Vectors $v_{applicant\_state, t}$ (from `candidates_embeddings`) and $v_{job, t}$ (from `job_embeddings`) are fetched via `pymongo`. The specific fields within the embeddings used for the calculation (e.g., concatenating hard/soft skill vectors) need to be defined.
* **Training Impact:** Provides a fast, cheap reward signal driving the agent to recommend semantically similar jobs. The World Model learns to predict this similarity score.

### Strategy 2: LLM Feedback Only Rewards

* **Concept:** Uses an LLM as a user simulator to generate richer, behavior-based feedback.
* **Reward Calculation ($R_{t+1}$):**
    1.  Present state $s_t$ and action $a_t$ (job details) to the LLM simulator.
    2.  LLM generates a simulated action (e.g., 'APPLY', 'SAVE', 'IGNORE').
    3.  Map this simulated action to a scalar reward (e.g., 'APPLY' -> +1.0, 'SAVE' -> +0.5, 'IGNORE' -> -0.1).
* **Training Impact:** Trains the agent based on potentially complex, simulated user preferences. The World Model learns to predict the LLM's reward output. More computationally expensive and relies heavily on simulator fidelity.

### Strategy 3: Hybrid (Cosine Pre-training + LLM Fine-tuning)

* **Concept:** A two-stage "scaffolding" approach.
* **Phase 1 (Pre-training):** Train the Q-network and World Model using **Cosine Similarity** rewards (Strategy 1). Establishes a semantically grounded initial policy.
* **Phase 2 (Fine-tuning):** Continue training the networks initialized from Phase 1, but **switch the reward source** to **LLM Feedback** (Strategy 2). Refines the policy using simulated behavioral signals.
* **Training Impact:** Aims to leverage the speed of cosine similarity for initial learning and the richness of LLM feedback for later refinement.

## Integration with Dyna-Q: Combining Real and Simulated Experience

Dyna-Q utilizes the selected training strategy within its loop:

1.  **Pre-training Phase (Optional, esp. for Hybrid):** Train $Q(s, a; \theta)$ and $\hat{P}(R | s, a)$ using either Cosine or LLM rewards (depending on the strategy) before starting the main loop. For Hybrid, this *is* Phase 1 using Cosine rewards.
2.  **Online Phase (Dyna-Q Loop):**
    * **a) Direct RL:**
        * Agent selects action $a_t$ using $Q(s, a; \theta)$.
        * Environment provides *real* reward $R_{t+1}$ based on the **currently active strategy** (Cosine, LLM, or Hybrid Phase 2's LLM reward) and next state $s_{t+1}$.
        * Store $(s_t, a_t, R_{t+1}, s_{t+1})$ in $\mathcal{D}_{real}$.
        * Perform Q-learning update on $\theta$ using a batch from $\mathcal{D}_{real}$ and the target $y_t$ (which uses $R_{t+1}$ from the active strategy).
    * **b) Model Learning:**
        * Train World Model $\hat{P}$ on batches from $\mathcal{D}_{real}$ to predict the reward $R_{t+1}$ (from the active strategy).
    * **c) Planning:**
        * Sample past $(s, a)$ from $\mathcal{D}_{real}$.
        * Use World Model $\hat{P}$ to predict simulated reward $\hat{R}$.
        * Perform Q-learning update on $\theta$ using simulated experience $(s, a, \hat{R}, s)$ and target $y_{sim} = \hat{R} + \gamma \max_{a'} Q(s, a'; \theta^{-})$.

## Evaluation Strategy: Comparing Training Strategies

The evaluation will rigorously compare the three training strategies and a baseline.

* **Core Hypothesis:** Pre-training or training with simulated rewards (Cosine or LLM-based) improves initial ("cold-start") performance compared to random initialization. The Hybrid approach may offer the best balance.
* **Evaluation Methodology:**
    * Train agents using each of the three strategies and a baseline (random start).
    * Evaluate all agents in a separate, consistent simulated online environment (fetching data from MongoDB).
    * Compare performance using metrics focused on **initial learning phase** (e.g., cumulative reward in first N episodes/steps, average reward, steps-to-threshold) and long-term convergence.
* **Expected Outcome:** Quantify the benefits of each strategy in mitigating cold start, providing insights into whether semantic similarity, complex LLM simulation, or a hybrid combination is most effective for initializing the RL job recommender.