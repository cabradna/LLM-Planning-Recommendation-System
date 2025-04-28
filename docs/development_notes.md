# Neural Dyna-Q Notebook Development Notes

## Overview

`neural_dyna_q_notebook.py` is a Python script formatted as a Jupyter Notebook that implements the Neural Dyna-Q job recommendation system. This script serves as an executable demonstration of the core algorithms developed in the LLM-Planning-Recommendation-System repository, specifically focusing on the neural network-based Dyna-Q reinforcement learning approach to job recommendations.

The file is designed to be run either in Google Colab or locally, with appropriate error handling to manage different execution environments. It contains both code cells and markdown cells (using the `# %%` and `# %% [markdown]` format) which makes it compatible with notebook interfaces like Jupyter, Colab, or VS Code's interactive Python.

## Purpose and Functionality

The primary purpose of this notebook is to demonstrate and execute the Neural Dyna-Q job recommendation algorithm, which:

1. Uses reinforcement learning to personalize job recommendations for specific candidates
2. Combines model-free learning with model-based planning (Dyna-Q approach)
3. Utilizes neural networks to approximate value functions and environment dynamics
4. Addresses the cold-start problem through multiple pre-training strategies

The notebook implements three different reward generation strategies:
- **Cosine Similarity**: Uses semantic similarity between candidate and job embeddings
- **LLM Feedback**: Leverages large language models to simulate candidate responses
- **Hybrid Approach**: Combines both methods with customizable weighting

## File Structure

The notebook is organized into several logical sections:

1. **Introduction**: Explains the system's purpose and key features
2. **Setup Environment**: Configures the execution environment (Colab or local)
3. **Import Project Modules**: Loads required modules with fallback implementations
4. **Database Connection**: Connects to MongoDB or creates a mock database
5. **Data Loading**: Fetches candidate and job embeddings
6. **Model Initialization**: Creates neural network models (Q-Network and World Model)
7. **Environment Setup**: Configures the job recommendation environment
8. **LLM Integration**: Sets up LLM for feedback-based reward strategies (if applicable)
9. **Replay Buffer Setup**: Initializes experience replay for reinforcement learning
10. **Pretraining Phase**: Trains models using supervised learning on generated data
11. **Online Learning Phase**: Implements the Dyna-Q algorithm for online learning
12. **Evaluation and Visualization**: Displays performance metrics and visualizations
13. **Test the Recommendation System**: Generates job recommendations using the trained model
14. **Conclusion**: Summarizes the implementation and results

## Running the Notebook

### In Google Colab

1. Upload the `neural_dyna_q_notebook.py` file to Google Drive
2. Open Google Colab (https://colab.research.google.com)
3. Use File > Upload notebook > Choose file from your Google Drive
4. Execute the cells sequentially

Alternatively, you can directly import from GitHub:
1. In Google Colab, go to File > Open notebook
2. Select the "GitHub" tab
3. Enter the repository URL: https://github.com/cabradna/LLM-Planning-Recommendation-System
4. Select the notebook file

The notebook will automatically:
- Clone the repository
- Install required dependencies
- Set up the environment for execution

### In VS Code

1. Open the file in VS Code
2. Make sure you have the Python and Jupyter extensions installed
3. Use the interactive Python feature to run individual cells
4. Execute cells sequentially using the "Run Cell" button

### In Jupyter Notebook/Lab

1. Convert the file to .ipynb format using the `jupytext` tool:
   ```
   pip install jupytext
   jupytext --to notebook neural_dyna_q_notebook.py
   ```
2. Open the resulting .ipynb file in Jupyter Notebook/Lab
3. Execute the cells sequentially

## Dependencies

The notebook requires the following main dependencies:
- Python 3.8+
- PyTorch
- NumPy
- MongoDB (or uses a mock database for demonstration)
- Matplotlib
- tqdm
- Transformers (for LLM integration, optional)

The script includes robust error handling to manage missing dependencies:
- PyTorch installation is attempted if missing
- Mock implementations of required classes are provided if imports fail
- A synthetic database is created if MongoDB connection fails
- LLM-based strategies fall back to cosine similarity if requirements aren't met

## Configuration Options

Several parameters can be adjusted to customize the execution:

- **Reward Strategy**: Choose between "cosine", "llm", or "hybrid"
- **Training Parameters**: Adjust batch size, learning rates, epochs, etc.
- **Model Architecture**: Modify neural network hidden dimensions, dropout rates
- **Database Connection**: Configure MongoDB connection details

## Note About Magic Commands

The file contains Jupyter notebook magic commands (lines starting with `!` or `%`) that are used for:
- Cloning the repository (`!git clone ...`)
- Changing directory (`%cd ...`)
- Installing packages (`!pip install ...`)

These commands will cause linter errors when viewed as a Python script but are valid when executing in a notebook environment.

## Fault Tolerance

One of the key features of the notebook is its comprehensive error handling:

1. Each major component is wrapped in try-except blocks
2. Fallback mechanisms are provided for missing dependencies or failed operations
3. Simulated data is generated if real data is unavailable
4. Mock implementations of classes are provided if imports fail

This ensures the notebook can run in various environments, providing an educational demonstration even if some components are unavailable.

## Output and Visualization

The notebook generates the following outputs:
- Training loss plots for both Q-Network and World Model
- Learning curves showing episode rewards and average Q-values
- A list of top job recommendations with their associated scores
- Detailed progress messages during training

## For Developers

If you're extending or modifying this notebook:

1. Maintain the error handling structure to ensure it remains robust
2. Test in both Colab and local environments
3. Keep the fallback implementations updated if you modify the core classes
4. Consider the Jupyter notebook formatting (`# %%` and `# %% [markdown]`) when editing

## Known Issues

- The linter will flag Jupyter magic commands (`!` and `%`) as errors when viewed as a Python file
- Running with the LLM-based reward strategy requires significant computational resources
- Some environments may require manual installation of dependencies

## Conclusion

The `neural_dyna_q_notebook.py` file serves as both a demonstration and an educational tool for understanding the Neural Dyna-Q job recommendation system. It showcases the integration of reinforcement learning, neural networks, and potentially LLMs for creating personalized job recommendations while addressing the cold-start problem in recommendation systems. 