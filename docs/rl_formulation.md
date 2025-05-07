# LLM-Driven Dyna-Q for Enhanced Job Recommendation: Reinforcement Learning Formulation

## 1. Overview and Motivation

This document details the Reinforcement Learning (RL) formulation for a job recommendation system. The primary challenge addressed is the **cold-start problem**, where traditional RL agents struggle due to a lack of initial interaction data. Our approach employs the Dyna-Q architecture, which integrates direct reinforcement learning with model-based planning, to enhance learning efficiency.

A key aspect of this work is the investigation of different reward generation strategies to guide the agent's learning, especially in the initial phases:
1.  **Cosine Similarity Rewards**: Leveraging semantic similarity between applicant and job profiles.
2.  **LLM-Simulated Rewards**: Using a Large Language Model (LLM) to simulate user feedback and generate richer reward signals.
3.  **Hybrid Strategy**: A two-phase approach potentially combining an initial phase of training with semantic similarity rewards, followed by fine-tuning with LLM-simulated rewards.

The goal is to develop an agent that learns an effective policy for recommending jobs that maximize an applicant's cumulative satisfaction or success, represented by a discounted sum of rewards.

## 2. Markov Decision Process (MDP) Formulation

The job recommendation task is modeled as a Markov Decision Process (MDP), defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

*   **Agent**: The recommendation system itself, tasked with learning and executing the job recommendation policy.

*   **Environment**: The context of an applicant's job search. This includes the applicant's profile, the available jobs, and the mechanism providing feedback on recommendations. The nature of this feedback, and thus the reward signal, varies based on the active training strategy.

*   **State Space ($\mathcal{S}$)**: A state $s_t \in \mathcal{S}$ at time step $t$ represents the relevant context of the applicant. It is defined as a vector embedding $v_{\text{applicant}, t}$ derived from the applicant's profile (e.g., concatenating embeddings of hard and soft skills).
    $s_t = v_{\text{applicant}, t}$
    In the current formulation, the applicant's profile features are considered static for the duration of an episode, meaning the state primarily serves to contextualize actions.

*   **Action Space ($\mathcal{A}$)**: The set of all available jobs at time $t$. An action $a_t \in \mathcal{A}$ corresponds to recommending a specific job to the applicant. Similar to the state, an action is represented by a vector embedding $v_{\text{job}, t}$ derived from the job's features.
    $a_t = v_{\text{job}, t}$

*   **Policy ($\pi(a|s)$)**: The agent's policy dictates the probability of selecting action $a$ in state $s$. During training and execution, actions are selected from a candidate subset of jobs $J_{\text{candidates}} \subset \mathcal{A}$. An $\epsilon$-greedy strategy is employed for action selection:
    *   With probability $1-\epsilon$, the agent selects the action (job) $a_t = \arg\max_{a \in J_{\text{candidates}}} Q(s_t, a; \theta)$, exploiting its current knowledge.
    *   With probability $\epsilon$, the agent selects a random action from $J_{\text{candidates}}$, exploring the action space.
    The exploration rate $\epsilon$ may be annealed over time.

*   **Reward Function ($R(s_t, a_t)$)**: The reward $R_{t+1}$ is a scalar feedback signal received after taking action $a_t$ in state $s_t$. Its generation is critically dependent on the active training strategy:
    *   **Cosine Similarity Reward**: The reward is a direct measure of semantic similarity between the applicant state embedding and the recommended job action embedding:
        \[ R_{t+1}^{\text{cosine}} = \frac{s_t \cdot a_t}{\|s_t\| \|a_t\|} \]
        This reward may be scaled (e.g., to $[0, 1]$).
    *   **LLM-Simulated Reward**: The reward is derived from an LLM acting as a user simulator. The LLM receives $s_t$ and $a_t$ (or their textual representations) and generates simulated user behavior (e.g., 'apply', 'ignore'). This behavior is then mapped to a scalar reward value:
        \[ R_{t+1}^{\text{LLM}} = f_{\text{LLM}}(s_t, a_t) \]
        where $f_{\text{LLM}}$ represents the LLM simulation and mapping process.
    *   **Hybrid Reward Strategy**: This involves distinct training phases. In an initial phase, rewards might be $R_{t+1}^{\text{cosine}}$. In a subsequent fine-tuning phase, rewards switch to $R_{t+1}^{\text{LLM}}$, or a weighted combination.

*   **State Transition Dynamics ($P(s_{t+1}|s_t, a_t)$)**: Given the current formulation where the applicant's state $s_t$ is static within an episode (representing fixed applicant characteristics), the state transition is deterministic and trivial:
    \[ s_{t+1} = s_t \]
    The environment's dynamics are primarily captured through the reward signal, which reflects the immediate outcome of the recommendation $a_t$.

*   **Discount Factor ($\gamma$)**: A scalar $0 \le \gamma \le 1$ that determines the present value of future rewards. A value closer to 0 prioritizes immediate rewards, while a value closer to 1 gives more weight to long-term rewards.

*   **Objective**: The agent's objective is to learn a policy $\pi^*$ that maximizes the expected discounted cumulative reward (return) from each state $s$:
    \[ G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \]

## 3. Value Function Approximation

To find the optimal policy, we estimate the optimal action-value function $Q^*(s,a)$, which is the expected return starting from state $s$, taking action $a$, and thereafter following the optimal policy $\pi^*$:
$$Q^*(s,a) = \mathbb{E}_{\pi^*} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | s_t=s, a_t=a \right]$$
This function is approximated using a neural network, the Q-Network, parameterized by weights $\theta$. So, $Q(s,a; \theta) \approx Q^*(s,a)$. The Q-network takes the concatenated state and action vectors $[s, a]$ as input and outputs a single scalar Q-value.

## 4. Learning the Action-Value Function

The Q-Network's parameters $\theta$ are learned using a variant of Q-learning, incorporating experience replay and a target network.

*   **Experience Replay**: Transitions $(s_t, a_t, R_{t+1}, s_{t+1})$ are stored in a replay buffer $\mathcal{D}$. During training, mini-batches are sampled from this buffer to update the network, breaking correlations and improving learning stability.

*   **Temporal Difference (TD) Learning**: The Q-Network is updated using TD learning. For a transition $(s_j, a_j, R_{j+1}, s_{j+1})$ sampled from $\mathcal{D}$:
    *   The **current Q-value** is $Q(s_j, a_j; \theta)$.
    *   The **target Q-value** ($y_j$) is constructed based on the observed reward $R_{j+1}$ and the estimated value of the next state using the target network $Q(s,a; \theta^{-})$ (see below). Crucially, reflecting the actual implementation, the target is calculated as:
        \[ y_j = R_{j+1} + \gamma Q(s_{j+1}, a_j; \theta^{-}) \]
        This target uses the action $a_j$ taken in state $s_j$ to evaluate the Q-value in the next state $s_{j+1}$. This differs from the standard Q-learning target which involves a $\max_{a'}$ operation over actions in $s_{j+1}$. This specific form implies that the value of taking action $a_j$ is assessed based on its immediate reward and the discounted future value assuming action $a_j$ (or an action with similar characteristics if $a_j$ itself is used as a proxy for the policy's choice in $s_{j+1}$) would be relevant for $s_{j+1}$. Given $s_{j+1} = s_j$, this effectively means the target is $R_{j+1} + \gamma Q(s_j, a_j; \theta^{-})$.

*   **Loss Function**: The Q-Network parameters $\theta$ are updated by minimizing the Mean Squared Error (MSE) between the current Q-value and the target Q-value:
    \[ L(\theta) = \mathbb{E}_{(s_j,a_j,R_{j+1},s_{j+1}) \sim \mathcal{D}} \left[ (y_j - Q(s_j, a_j; \theta))^2 \right] \]

*   **Optimization**: The loss is minimized using gradient descent-based optimization algorithms (e.g., Adam).

*   **Target Network**: A separate target network $Q(s,a; \theta^{-})$ with parameters $\theta^{-}$ is used to compute the target values $y_j$. The weights $\theta^{-}$ are periodically copied from the online Q-Network's weights $\theta$ (i.e., $\theta^{-} \leftarrow \theta$). This helps stabilize learning by keeping the target values fixed for a period.

## 5. Dyna-Q Framework Integration

The Dyna-Q architecture enhances sample efficiency by integrating direct RL, world model learning, and planning.

*   **Direct Reinforcement Learning**: The agent learns directly from real interactions with the environment. Experiences $(s_t, a_t, R_{t+1}, s_{t+1})$ are collected, where $R_{t+1}$ is determined by the active training strategy. These experiences are used to update the Q-Network $Q(s,a;\theta)$ as described in Section 4.

*   **World Model Learning**: The agent learns a model of the environment's dynamics. Since state transitions are trivial ($s_{t+1}=s_t$), the world model $M(s,a;\phi)$ (parameterized by $\phi$) focuses on predicting the immediate reward:
    \[ M(s,a;\phi) \approx R(s,a) \]
    The world model is trained in a supervised manner using real experiences $(s_t, a_t, R_{t+1})$ by minimizing the MSE between the predicted reward $\hat{R}_t = M(s_t, a_t; \phi)$ and the actual reward $R_{t+1}$:
    \[ L(\phi) = \mathbb{E}_{(s_t,a_t,R_{t+1}) \sim \mathcal{D}} \left[ (R_{t+1} - M(s_t, a_t; \phi))^2 \right] \]

*   **Planning**: The learned world model is used to generate simulated experiences for additional Q-Network updates. The planning process involves:
    1.  Sampling a previously experienced state-action pair $(s,a)$ from the replay buffer $\mathcal{D}$.
    2.  Using the world model to predict the reward for this pair: $\hat{R} = M(s,a;\phi)$.
    3.  Since $s' = s$, the simulated experience is $(s,a,\hat{R},s)$.
    4.  Updating the Q-Network $Q(s,a;\theta)$ using this simulated experience. The target for this update is:
        \[ \hat{y} = \hat{R} + \gamma Q(s, a; \theta^{-}) \]
        This update rule is consistent with the one used for direct RL, using the Q-value of the same state-action pair $(s,a)$ for the next state's value, as the state is static.
    This planning step is repeated for a number of iterations after each real interaction.

*   **Overall Dyna-Q Process**:
    1.  Interact with the environment: $s_t \xrightarrow{a_t} R_{t+1}, s_{t+1}$. Store $(s_t, a_t, R_{t+1}, s_{t+1})$ in $\mathcal{D}$.
    2.  Direct RL Update: Update $Q(s,a;\theta)$ using a batch from $\mathcal{D}$.
    3.  Model Learning Update: Update $M(s,a;\phi)$ using a batch from $\mathcal{D}$.
    4.  Planning: Perform $N$ planning steps. Each step involves sampling $(s,a)$ from $\mathcal{D}$, generating $\hat{R}$ via $M(s,a;\phi)$, and updating $Q(s,a;\theta)$ using $(s,a,\hat{R},s)$.

## 6. Training Strategies in the Dyna-Q Context

The choice of reward strategy (Cosine, LLM, Hybrid) directly influences the $R_{t+1}$ values used in both the direct RL updates and for training the world model $M(s,a;\phi)$.

*   **Cosine Similarity Strategy**: The agent (`agent_cosine`) can be trained using a dataset where rewards are pre-calculated cosine similarities. This training updates both the Q-Network and the World Model based on these semantic rewards. The World Model thus learns to predict cosine similarity.
*   **Hybrid LLM Strategy**: The agent (`agent_hybrid`) is trained online.
    *   **Direct RL**: The $R_{t+1}$ comes from the `HybridEnv` (which may combine LLM feedback and cosine similarity, possibly with annealing weights for the cosine component).
    *   **Model Learning**: The World Model learns to predict these (potentially complex or hybrid) rewards from the `HybridEnv`.
    *   **Planning**: The Q-Network is updated using rewards simulated by this learned World Model.

This formulation provides a foundation for understanding how the agent learns to make job recommendations by interacting with its environment (real or simulated) and refining its value estimates and world model over time.