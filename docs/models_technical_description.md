# Neural Models Technical Description

This document provides a detailed technical description of the neural network models used in our Dyna-Q based recommendation system. The system employs two main neural networks: a Q-Network for value function approximation and a World Model for environment dynamics prediction.

## 1. Q-Network Architecture

The Q-Network (`QNetwork` class in `src/models/q_network.py`) is a deep neural network that approximates the action-value function $Q(s, a)$. This function estimates the expected cumulative discounted reward of taking action $a$ in state $s$.

### 1.1 Architecture Details

The network takes as input:
- State vector of dimension 768 (applicant embeddings)
- Action vector of dimension 1536 (job embeddings)

The architecture consists of:
```python
# From config/config.py
MODEL_CONFIG = {
    "q_network": {
        "state_dim": 768,  # Dimension of applicant embeddings
        "action_dim": 1536,  # Dimension of job embeddings
        "hidden_dims": [1024, 512, 256],  # Hidden layer dimensions
        "dropout_rate": 0.2,  # Dropout rate
        "activation": "relu"  # Activation function
    }
}
```

The network architecture follows this structure:
1. Input Layer: Concatenates state and action vectors (2304 dimensions)
2. Hidden Layer 1: 1024 neurons with ReLU activation
3. Hidden Layer 2: 512 neurons with ReLU activation
4. Hidden Layer 3: 256 neurons with ReLU activation
5. Output Layer: Single neuron (Q-value)

### 1.2 Forward Pass

The forward pass through the network is implemented as follows:
```python
def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # Concatenate state and action
    x = torch.cat([state, action], dim=1)
    
    # Forward pass through hidden layers
    for layer in self.layers:
        x = layer(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
    
    # Output layer (no activation - Q-value can be any real number)
    q_value = self.output_layer(x)
    
    return q_value
```

### 1.3 Training Process

The Q-Network is trained using the following loss function:

$L(\theta_Q) = \mathbb{E}_{(s,a,r,s') \sim D}[(r + \gamma \max_{a'} Q(s', a'; \theta_{Q'}) - Q(s, a; \theta_Q))^2]$

Where:
- $\theta_Q$ are the Q-Network parameters
- $\theta_{Q'}$ are the target network parameters
- $\gamma$ is the discount factor
- $D$ is the replay buffer

The update process is implemented in the `update_q_network` method:
```python
def update_q_network(self, states, actions, rewards, next_states):
    # Compute current Q values
    current_q_values = self.q_network(states, actions)
    
    # Compute target Q values
    with torch.no_grad():
        next_q_values = rewards + self.gamma * self.target_network(next_states, actions)
    
    # Compute loss and update
    loss = self.q_criterion(current_q_values, next_q_values)
    self.q_optimizer.zero_grad()
    loss.backward()
    self.q_optimizer.step()
```

## 2. World Model Architecture

The World Model (`WorldModel` class in `src/models/world_model.py`) is a neural network that learns to predict the environment dynamics, specifically the immediate reward $R(s,a)$ for a given state-action pair.

### 2.1 Architecture Details

The network takes as input:
- Concatenated state-action vector of dimension 2304 (768 + 1536)

The architecture consists of:
```python
# From config/config.py
MODEL_CONFIG = {
    "world_model": {
        "input_dim": 768+1536,  # state_dim + action_dim
        "hidden_dims": [1024, 512],  # Hidden layer dimensions
        "dropout_rate": 0.2,  # Dropout rate
        "activation": "relu"  # Activation function
    }
}
```

The network architecture follows this structure:
1. Input Layer: 2304 dimensions (concatenated state-action)
2. Hidden Layer 1: 1024 neurons with ReLU activation
3. Hidden Layer 2: 512 neurons with ReLU activation
4. Output Layer: Single neuron (predicted reward)

### 2.2 Forward Pass

The forward pass through the World Model is implemented as follows:
```python
def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # Concatenate state and action
    x = torch.cat([state, action], dim=1)
    
    # Forward pass through reward prediction network
    for layer in self.reward_layers:
        x = layer(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
    
    # Output layer for reward prediction
    predicted_reward = self.reward_output(x)
    
    return predicted_reward
```

### 2.3 Training Process

The World Model is trained to minimize the mean squared error between predicted and actual rewards:

$L(\theta_M) = \mathbb{E}_{(s,a,r) \sim D}[(r - M(s, a; \theta_M))^2]$

Where:
- $\theta_M$ are the World Model parameters
- $M(s, a; \theta_M)$ is the predicted reward
- $r$ is the actual reward
- $D$ is the replay buffer

## 3. Model Integration in Dyna-Q

The models are integrated into the Dyna-Q algorithm through the `DynaQAgent` class. The agent maintains:
1. A Q-Network for action-value estimation
2. A target Q-Network for stable learning
3. A World Model for environment dynamics prediction

The models work together in the following way:
1. The Q-Network is used to select actions using an ε-greedy policy
2. The World Model is used to generate simulated experiences during planning
3. The target Q-Network is used to compute stable Q-value targets

This integration allows the agent to:
- Learn from real experiences (direct RL)
- Learn from simulated experiences (model-based RL)
- Combine both approaches for more efficient learning

## 4. Model Persistence

Both models implement save and load functionality for persistence:

```python
def save(self, path: str) -> None:
    torch.save({
        'state_dict': self.state_dict(),
        'input_dim': self.input_dim,
        'hidden_dims': self.hidden_dims,
        'dropout_rate': self.dropout_rate,
        'activation': self.activation_name
    }, path)

@classmethod
def load(cls, path: str, device: torch.device = None) -> 'Model':
    checkpoint = torch.load(path, map_location=device)
    model = cls(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout_rate=checkpoint['dropout_rate'],
        activation=checkpoint['activation']
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model
```

This allows for:
- Saving trained models for later use
- Loading pretrained models
- Transfer learning between different tasks
- Model checkpointing during training 