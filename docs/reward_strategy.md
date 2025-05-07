# Reward Strategies in the Dyna-Q Job Recommender

## 1. Introduction

The reward signal is a cornerstone of Reinforcement Learning (RL), guiding the agent's learning process by providing feedback on its actions. In our Dyna-Q based job recommendation system, the definition and generation of rewards are critical for training an agent that can effectively match applicants with suitable job opportunities. This document details the different reward strategies implemented, explaining their theoretical basis and how they are realized within the environment.

The environment, particularly `JobRecommendationEnv` and its subclasses (`LLMSimulatorEnv`, `HybridEnv`), is responsible for calculating rewards based on the active strategy. Configuration for these strategies is primarily managed in `config/config.py` under `STRATEGY_CONFIG`.

## 2. Core Reward Principle

The fundamental goal is to provide a scalar reward $R_{t+1}$ after the agent takes an action $a_t$ (recommending a job) in state $s_t$ (representing the applicant). This reward indicates the immediate desirability or appropriateness of that job recommendation for that applicant.

## 3. Reward Strategies

Three primary reward strategies are implemented: Cosine Similarity, LLM-Simulated Feedback, and a Hybrid approach.

### 3.1. Cosine Similarity Reward Strategy

*   **Concept**: This strategy provides a reward based on the direct semantic similarity between the applicant's profile embedding and the recommended job's embedding. A higher cosine similarity suggests a better semantic match.
*   **Implementation**:
    *   The `JobRecommendationEnv.calculate_cosine_reward(action_idx)` method is responsible for this calculation.
    *   Given the current applicant state vector $s_t$ and the selected job action vector $a_t$ (obtained via `get_action_vector(action_idx)`), the cosine similarity is computed:
        $$R_{\text{cosine}} = \frac{s_t \cdot a_t}{\|s_t\| \|a_t\|}$$
    *   **Dimensionality Handling**: The applicant state vector (e.g., 768 dimensions from hard and soft skills) and job action vector (e.g., 1536 dimensions including tech skills, soft skills, title, experience) may have different dimensions. As per `JobRecommendationEnv.calculate_cosine_reward`, to ensure a meaningful comparison, the job vector is typically truncated to match the dimension of the applicant state vector. For instance, the first 768 dimensions of the job vector (representing its tech and soft skills) are compared against the 768 dimensions of the applicant state.
    *   **Scaling**: The raw cosine similarity score, which ranges from -1 to 1, can optionally be scaled to a [0, 1] range. This is controlled by `STRATEGY_CONFIG["cosine"]["scale_reward"]` (boolean) in `config/config.py`. If true, the reward becomes:
        $$R_{\text{scaled\_cosine}} = \frac{R_{\text{cosine}} + 1}{2}$$
*   **Purpose**: Provides a computationally inexpensive and readily available reward signal based on semantic content. It's useful for initial model training or as a baseline.
*   **Environment Class**: Primarily used by `JobRecommendationEnv` when `reward_strategy` is "cosine".

### 3.2. LLM-Simulated Reward Strategy

*   **Concept**: This strategy leverages a Large Language Model (LLM) to act as a user simulator. The LLM "evaluates" the job recommendation in the context of the applicant's profile and generates a qualitative response (e.g., "APPLY", "SAVE", "IGNORE"), which is then mapped to a numerical reward.
*   **Implementation**:
    *   This strategy is primarily managed by the `LLMSimulatorEnv` class, which inherits from `JobRecommendationEnv`.
    *   The core logic resides in `LLMSimulatorEnv.step()`, which calls `simulate_user_response()`.

    1.  **Prompt Construction (`_build_llm_prompt`)**:
        *   A detailed prompt is constructed to instruct the LLM. The prompt includes:
            *   The applicant's textual biography (e.g., `applicant_profile.get('bio')`).
            *   The applicant's textual resume (e.g., `applicant_profile.get('resume_string')`).
            *   The recommended job's title (e.g., `job.get("job_title")`).
            *   The recommended job's description (e.g., `job.get("description")`).
        *   The LLM is explicitly asked to choose ONLY ONE of the predefined responses: "APPLY", "SAVE", "CLICK", or "IGNORE", based on whether the job is a "perfect fit" for the seeker.
        *   The prompt structure uses `<s>[INST] ... [/INST]` markers for instruction tuning, as seen in `LLMSimulatorEnv._build_llm_prompt`.
        *   Example prompt snippet from `_build_llm_prompt`:
            ```
            <s>[INST] You are simulating a job seeker's response to a job recommendation.  

            JOB SEEKER PROFILE:  
            {applicant_bio}  

            RESUME:  
            {applicant_resume}  

            JOB RECOMMENDATION:  
            Title: {job_title}  
            Description:  
            {job_description}  

            Based on the job seeker's profile, would this role be a **perfect fit** they're **excited about**? Choose ONLY ONE:  
            - APPLY (Strong alignment—clear match with skills and interests)  
            - SAVE (Potential fit, but needs further review)  
            - CLICK (Mild interest—requires more details)  
            - IGNORE (Irrelevant or unappealing)  

            Respond with JUST ONE WORD: APPLY, SAVE, CLICK, or IGNORE. No explanations. [/INST]
            ```

    2.  **LLM Interaction (`_get_llm_decision`)**:
        *   The constructed prompt is tokenized.
        *   The LLM (`self.llm_model`) generates a response using `self.llm_model.generate()`.
        *   Key generation parameters from `STRATEGY_CONFIG["llm"]` in `config/config.py` include:
            *   `max_new_tokens` (e.g., 10, as set in `_get_llm_decision` to ensure a short response).
            *   `temperature` (e.g., 0.2, for less random outputs).
            *   `do_sample=False` (greedy decoding).
        *   The LLM model used is specified by `STRATEGY_CONFIG["llm"]["model_name"]` (e.g., "mistralai/Mistral-7B-Instruct-v0.1").
        *   Quantization settings (`STRATEGY_CONFIG["llm"]["quantization"]`) can be applied to the LLM for efficiency.

    3.  **Response Mapping (`_map_llm_response`)**:
        *   The raw textual output from the LLM (e.g., "APPLY") is mapped to one of the predefined categories ("APPLY", "SAVE", "CLICK", "IGNORE"). This mapping is robust to minor variations like case or extra whitespace.
        *   If the response is unclear, it defaults to "IGNORE".

    4.  **Numerical Reward Assignment**:
        *   The mapped categorical response is then converted into a numerical reward using `self.reward_scheme`. This scheme is defined in `STRATEGY_CONFIG["llm"]["response_mapping"]` in `config/config.py`.
        *   Example mapping from `config.py`:
            ```python
            "response_mapping": {
                "APPLY": 1.0,
                "SAVE": 0.5,
                "CLICK": 0.0,
                "IGNORE": -0.1
            }
            ```
        *   So, if the LLM responds with "APPLY", the numerical reward $R_{\text{LLM}}$ would be 1.0.

*   **Purpose**: To provide a richer, more nuanced reward signal that potentially captures complex aspects of job-applicant compatibility beyond simple semantic similarity.
*   **Environment Class**: `LLMSimulatorEnv` (when `reward_strategy` is "llm").

### 3.3. Hybrid Reward Strategy

*   **Concept**: This strategy combines the cosine similarity reward and the LLM-simulated reward, allowing for a weighted blend of the two. This can be useful for "scaffolding" the agent's learning, perhaps starting with more reliable (but simpler) cosine rewards and gradually incorporating the more complex LLM feedback.
*   **Implementation**:
    *   This strategy is managed by the `HybridEnv` class, which inherits from `LLMSimulatorEnv`.
    *   In `HybridEnv.step()`:
        1.  The cosine reward ($R_{\text{cosine\_scaled}}$ or $R_{\text{cosine}}$) is calculated using `self.calculate_cosine_reward(action_idx)`.
        2.  The LLM-simulated reward ($R_{\text{LLM}}$) is obtained as described in section 3.2.
        3.  A `cosine_weight` ($w_{\text{cos}}$), typically between 0.0 and 1.0, determines the blend. This weight is an attribute of the `HybridEnv` instance (`self.cosine_weight`).
        4.  The final reward is calculated as:
            $$R_{\text{hybrid}} = (w_{\text{cos}} \times R_{\text{cosine\_component}}) + ((1 - w_{\text{cos}}) \times R_{\text{LLM}})$$
            where $R_{\text{cosine\_component}}$ is the (potentially scaled) cosine reward.
    *   **Cosine Weight Dynamics**:
        *   The `cosine_weight` can be static or dynamic.
        *   The `DynaQAgent` can update this weight during training (e.g., via `agent_hybrid.update_cosine_weight()`), potentially annealing it from an initial value to a final value over a number of episodes.
        *   Configuration for annealing is found in `STRATEGY_CONFIG["hybrid"]` in `config/config.py`:
            *   `initial_cosine_weight`: Starting weight for cosine similarity.
            *   `final_cosine_weight`: Target weight for cosine similarity after annealing.
            *   `annealing_episodes`: Number of episodes over which to anneal the weight.
        *   The `HybridEnv` itself can have its `cosine_weight` set directly via `env.set_cosine_weight(weight)`.
*   **Purpose**: To leverage the strengths of both simpler semantic rewards (for stability or initial learning) and richer LLM-based feedback (for nuanced preference modeling), potentially transitioning from one to the other or using a fixed blend.
*   **Environment Class**: `HybridEnv` (when `reward_strategy` is "hybrid").

## 4. Impact on Agent Learning

The chosen reward strategy directly shapes the experiences $(s_t, a_t, R_{t+1}, s_{t+1})$ stored in the replay buffer. Consequently, it influences:
*   **Q-Network Updates**: The $R_{t+1}$ term in the TD target $y_j = R_{j+1} + \gamma Q(s_{j+1}, a_j; \theta^{-})$ is directly from the active reward strategy.
*   **World Model Learning**: The world model $M(s,a;\phi)$ learns to predict $R_{t+1}$, so it learns to predict cosine similarities, LLM-simulated rewards, or their hybrid combination, depending on the strategy.
*   **Planning**: Simulated experiences generated during planning use rewards $\hat{R}$ predicted by the world model, which are therefore indirect reflections of the chosen reward strategy.

By carefully designing and selecting the reward strategy, we aim to train a Dyna-Q agent that effectively learns the complex dynamics of job recommendation and provides high-quality suggestions to applicants. 