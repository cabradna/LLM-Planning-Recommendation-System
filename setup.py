from setuptools import setup, find_packages

setup(
    name="neural_model",
    version="0.1.0",
    description="Dyna-Q Job Recommender System with MongoDB Integration and LLM Pre-training",
    author="CS138 Project Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pymongo>=4.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.8.0",
) 