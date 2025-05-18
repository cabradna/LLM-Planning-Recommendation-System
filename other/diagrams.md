# Embedding Architecture Diagrams

## Option 1: Embedding Strategy Overview

```mermaid
graph TD
    subgraph "Input Data"
        J["Job Descriptions"] --> JE["Gemini LLM Extraction"]
        C["Candidate Profiles"] --> CE["Gemini LLM Generation"]
    end

    JE --> JHS["Job Hard Skills"]
    JE --> JSS["Job Soft Skills"]
    CE --> CHS["Candidate Hard Skills"]
    CE --> CSS["Candidate Soft Skills"]

    subgraph "Embedding Models"
        FT["Fine-tuned Sentence2Vec<br/>(all-MiniLM-L6-v2-finetuned)"]
        VB["Vanilla Sentence2Vec<br/>(all-MiniLM-L6-v2)"]
    end

    JHS --> FT
    CHS --> FT
    JSS --> VB
    CSS --> VB

    FT --> HSEMB["Hard Skills Embeddings<br/>(384-dim vectors)"]
    VB --> SSEMB["Soft Skills Embeddings<br/>(384-dim vectors)"]

    subgraph "MongoDB Storage"
        HSEMB --> JDB["job_embeddings<br/>Collection"]
        SSEMB --> JDB
        HSEMB --> CDB["candidates_embeddings<br/>Collection"]
        SSEMB --> CDB
    end
```

# Training Process Diagrams

## Dyna-Q Training Workflow (Compact Version)

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 10, 'rankSpacing': 15}}}%%
flowchart LR
    %% Main components with reduced spacing
    DB[(DB)]---ENV
    ENV[Environment]---R[Reward]---RB
    
    subgraph RS[Reward Strategies]
        direction LR
        R1[Cosine]---R2[LLM]---R3[Hybrid]
    end
    
    subgraph AG[Agent]
        direction LR
        RB[Buffer]-->RL[RL]
        RB-->ML[Model]
        ML-->PL[Planning]
        PL-->RL
        RL-->QN[Q-Net]
        RL-.->TQN[Target]
    end
    
    QN-->ENV
    R1 & R2 & R3-.->R

    %% Style definitions for more compact nodes
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef compact font-size:12px,padding:3px;
    class DB,ENV,R,RB,RL,ML,PL,QN,TQN,R1,R2,R3 compact;
```

## Dyna-Q Training Workflow (Detailed)

```mermaid
flowchart TD
    %% Main components
    subgraph "Neural Network Components"
        QN["Q-Network<br/>(Q(s,a;θ))"]
        TQN["Target Q-Network<br/>(Q(s,a;θ⁻))"]
        WM["World Model<br/>(Reward Predictor)"]
    end

    subgraph "Training Strategies"
        CS["Cosine Similarity<br/>Strategy"]
        LLM["LLM Feedback<br/>Strategy"]
        HYB["Hybrid Strategy<br/>(Cosine+LLM)"]
    end

    %% Data flow for initial setup
    INIT["Initialize Networks<br/>& Parameters"] --> QN & TQN & WM
    INIT --> |"Copy Weights<br/>θ⁻ = θ"| QN --> TQN
    
    %% Core training loop
    subgraph "Training Loop"
        direction TB
        AS["Action Selection<br/>(ε-greedy Policy)"]
        EI["Environment Interaction<br/>(State, Action → Reward)"]
        ES["Experience Storage<br/>(Replay Buffer)"]
        QL["Q-Learning Update<br/>(Direct RL)"]
        ML["Model Learning<br/>(Update World Model)"]
        PL["Planning<br/>(Simulated Experience)"]
        TU["Target Network Update<br/>(Every C Steps)"]
    end
    
    %% Specific training paths
    subgraph "Agent Training Paths"
        AP["agent_cosine<br/>(Pretrain Method)"] --> |"Batch Learning<br/>from Dataset"| QL & ML
        AH["agent_hybrid<br/>(Train Method)"] --> |"Online Learning<br/>with Environment"| AS
    end
    
    %% Connect the strategy to reward generation
    CS -.-> |"Reward = cos(applicant, job)"| EI
    LLM -.-> |"Reward from LLM Response"| EI
    HYB -.-> |"Weighted Combination<br/>R = α·Rcosine + (1-α)·RLLM"| EI
    
    %% Main workflow connections
    AS --> EI
    EI --> ES
    ES --> QL
    ES --> ML
    QL --> |"If steps_done % C == 0"| TU
    TU --> TQN
    ML --> WM
    WM --> PL
    PL --> QL
    
    %% MongoDB connection for job/applicant data
    DB[(MongoDB<br/>job_embeddings<br/>candidates_embeddings)] --> |"Fetch Job &<br/>Applicant Vectors"| EI
    
    %% Reward annealing for hybrid strategy
    ANNEAL["Cosine Weight Annealing<br/>(For Hybrid Strategy)"] -.-> |"α gradually decreases<br/>over training"| HYB
```

# RL Formulation Diagram

## Job Recommendation MDP

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 15, 'rankSpacing': 25}}}%%
flowchart TD
    %% MDP Components
    subgraph MDP["Markov Decision Process"]
        direction LR
        S[State s]
        A[Action a]
        R[Reward R]
        P[Transition P]
        
        S --> A
        A --> R
        R --> P
        P -.-> |"st+1 = st"| S
    end
    
    %% Specific representations for this problem
    subgraph ST["State"]
        direction LR
        SV["v_applicant<br/>Applicant Vector"]
    end
    
    subgraph ACT["Action"]
        direction LR
        AV["v_job<br/>Job Vector"]
    end
    
    subgraph RWD["Reward Strategies"]
        direction LR
        COS["Cosine Similarity<br/>cos(v_applicant, v_job)"]
        LLMR["LLM Feedback<br/>f_LLM(s,a)"]
        HYB["Hybrid Strategy<br/>α·Rcos + (1-α)·RLLM"]
    end
    
    %% Q-Learning components
    subgraph QL["Q-Learning"]
        direction TB
        QF["Q(s,a;θ) ≈ Q*(s,a)<br/>Q-Network"]
        TD["Target: Rt+1 + γ·Q(st+1,at;θ-)<br/>TD Learning"]
        L["Loss: (Target - Q(st,at;θ))²<br/>MSE"]
        
        QF --> TD
        TD --> L
        L --> |"Update"| QF
    end
    
    %% Connect MDP to specific representations
    S --- ST
    A --- ACT
    R --- RWD
    
    %% Connect to Q-Learning
    MDP <--> QL
    
    %% Special note on policy
    POL["Policy π(a|s)<br/>ε-greedy<br/>• Explore (ε): Random job<br/>• Exploit (1-ε): argmax Q(s,a;θ)"]
    
    POL --> A
    QF --> POL
    
    %% Style definitions
    classDef mdpNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef implNode fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef highlight fill:#ffe0b2,stroke:#e65100,stroke-width:2px;
    
    class S,A,R,P mdpNode;
    class SV,AV,COS,LLMR,HYB,QF,TD,L implNode;
    class POL highlight;
```

