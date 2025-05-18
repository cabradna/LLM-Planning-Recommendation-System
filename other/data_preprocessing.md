# Data Processing Workflow for Job Data Analysis
[Back to Index](../index.md)
## 1. Overview of the Data Pipeline

Our project involves several steps to process job posting data and candidate profiles for reinforcement learning applications:

1. **Data Collection**: Gathering job posting datasets from Kaggle sources.
2. **Data Preprocessing**: Cleaning and standardizing the job data.
3. **LLM-Based Data Mining (Jobs)**: Using Gemini to extract structured information (technical skills, soft skills, experience) from job descriptions.
4. **LLM-Based Candidate Generation (Candidates)**: Using Gemini to generate synthetic candidate profiles, including bios, resumes, and skill sets.
5. **Feature Engineering & Embedding**:
    * Employing Sentence2Vec for generating embeddings.
    * Fine-tuning a Sentence2Vec model using contrastive learning for hard skill embeddings (jobs and candidates).
    * Using a vanilla Sentence2Vec model for soft skill embeddings (jobs and candidates).
6. **MongoDB Integration**: Storing all processed textual data and their corresponding embeddings in a NoSQL database.
7. **Model Training & Application**: (Future Work) Utilizing the processed data and embeddings for downstream ML tasks like skill recommendation and job matching.

## 2. Data Sources

We're working with three main datasets from Kaggle:

1. **Google Jobs Search Results** (`gsearch_jobs.csv`) - 287MB, 61.3K entries
2. **LinkedIn Job Postings** (`postings.csv`) - 556MB, 124K entries 
3. **Data Analyst Jobs** (`DataAnalyst.csv`) - 7.57MB, 2,253 entries

## 3. Data Preprocessing Steps

### 3.1 Initial Data Exploration

I used the `DataFrameSummarizer` class to analyze our datasets before importing them to MongoDB:

```python
from data_tools import DataFrameSummarizer

# Load dataset
df = pd.read_csv("kaggle_datasets/gsearch_jobs.csv")

# Create summarizer
summarizer = DataFrameSummarizer(df)

# Display overview
summarizer.display_summary()
```

This provides detailed information about:
- Dataset dimensions
- Column types
- Missing value percentages
- Value distributions

### 3.2 Data Cleaning

Key preprocessing steps, primarily detailed in `data_preprocessing/project/data_preprocessing.ipynb`, included:
- Handling missing values: Imputing or removing entries with missing critical information.
- Standardizing text fields: Normalizing case, removing special characters, and correcting common inconsistencies.
- Removing duplicates: Identifying and eliminating redundant job postings.
- Normalizing column names: Ensuring consistent naming conventions across different data sources.

## 4. LLM-Based Data Mining & Candidate Generation with Gemini

Google's Gemini model played a crucial role in two main areas: extracting structured data from job descriptions and generating synthetic candidate profiles.

### 4.1 Extracting Structured Data from Job Descriptions

I used Google's Gemini model to extract:
- Technical skills (programming languages, tools)
- Soft skills (communication, teamwork)
- Experience requirements (education, years of experience)

```python
# Format job data for the extractor
jobs_dict = {
    job_id: {
        'description': job.get('description', ''),
        'job_title': job.get('job_title', '')
    }
    for job in batch
}

# Extract skills data
results = extractor.extract_job_data(jobs=jobs_dict)
```

### 4.2 Candidate Profile Generation

Gemini was also employed to generate synthetic candidate profiles to create a diverse dataset for model training and evaluation. These profiles are stored in `data_preprocessing/misc_data/candidates.json`.

**Structure of `candidates.json`**:

The `candidates.json` file contains a main key `"candidates"` which holds a list of candidate objects. Each candidate object has the following structure:

```json
{
    "bio": "A detailed narrative of the candidate's professional background, skills, experience, and career aspirations.",
    "resume": "A structured resume format detailing work experience, education, skills, and projects.",
    "hard_skills": ["List", "of", "technical", "skills"],
    "soft_skills": ["List", "of", "interpersonal", "skills"]
}
```
For example, a candidate entry might look like this (snippet from `data_preprocessing/misc_data/candidates.json`):
```json
// Snippet from data_preprocessing/misc_data/candidates.json
{
    "bio": "Alex Chen is an analytical and results-oriented Data Scientist with four years of dedicated experience transforming complex datasets into actionable insights...",
    "resume": "Alex Chen\n[City, State] | [Phone Number] | [Email Address] | [LinkedIn Profile URL]\n\nProfessional Summary\n\nHighly motivated Data Scientist with 4 years of experience...",
    "hard_skills": [
      "Python",
      "SQL",
      "Pandas",
      "NumPy",
      "Scikit-learn",
      // ... more hard skills
    ],
    "soft_skills": [
      "Analytical Thinking",
      "Problem-Solving",
      "Communication",
      // ... more soft skills
    ]
}
```
This structured generation allows for a rich dataset of candidates with varied experiences and skill sets.

### 4.3 Batch Processing with Rate Limiting (for Job Data Extraction)

To avoid API rate limits:

```python
# Constants for rate limiting
RATE_LIMIT_PER_MINUTE = 30
SECONDS_BETWEEN_REQUESTS = (60.0 / RATE_LIMIT_PER_MINUTE) * 1.1  # Add 10% buffer

# Rate limiting implementation
current_time = time.time()
elapsed_since_last_request = current_time - last_request_time

if elapsed_since_last_request < SECONDS_BETWEEN_REQUESTS:
    sleep_time = SECONDS_BETWEEN_REQUESTS - elapsed_since_last_request
    time.sleep(sleep_time)

# Process batch
results = extractor.extract_job_data(jobs=jobs_dict)
last_request_time = time.time()  # Update last request time
```

## 5. Feature Engineering: Embedding Strategy

To represent textual data (skills, job titles, etc.) in a way that machine learning models can understand, we employed an embedding strategy centered around Sentence2Vec models.

### 5.1 Initial Model Evaluation and Sentence2Vec Adoption

We initially considered different embedding techniques, including comparing FastText with Sentence2Vec for representing hard skills. Sentence2Vec demonstrated superior performance in capturing the semantic nuances of technical skills, leading to its adoption for all skill and textual embeddings.

The core of our embedding generation relies on the `Sentence2VecEncoder` class, found in `data_preprocessing/preprocessing_utils/sentence2vec_utils.py`. This class handles the loading of pretrained or fine-tuned SentenceTransformer models.

```python
# From data_preprocessing/preprocessing_utils/sentence2vec_utils.py
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sentence2VecEncoder:
    """
    Utility class for loading and using SentenceTransformer models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Sentence2Vec encoder
        
        Args:
            model_path (str, optional): Path to the trained model. 
                                        If None, uses a default fine-tuned model path.
        """
        self.model_path = model_path or "../misc_data/all-MiniLM-L6-v2-finetuned" # Default path to a fine-tuned model
        self.model = None
        
    def load_model(self) -> None:
        """
        Load the SentenceTransformer model.
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}. Attempting to use base model name {os.path.basename(self.model_path)} instead.")
                # Fallback to loading by name if path doesn't exist (e.g., 'all-MiniLM-L6-v2')
                self.model = SentenceTransformer(os.path.basename(self.model_path))
            else:
                self.model = SentenceTransformer(self.model_path)
                logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts (str or list): Single text or list of texts to encode.
            normalize (bool): Whether to normalize the embeddings.
            
        Returns:
            numpy.ndarray: Embeddings for the input texts.
        """
        if self.model is None:
            self.load_model()
            
        try:
            if isinstance(texts, str):
                texts = [texts]
            embeddings = self.model.encode(texts, normalize_embeddings=normalize)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
```

### 5.2 Fine-Tuning for Hard Skills via Contrastive Learning

For job hard skills and candidate hard skills, we fine-tuned a Sentence2Vec model (specifically, a model based on `all-MiniLM-L6-v2` architecture, resulting in the model saved at `../misc_data/all-MiniLM-L6-v2-finetuned` as indicated in `sentence2vec_utils.py`). This fine-tuning process utilized a contrastive learning approach.

Contrastive learning aims to teach the model to produce similar embeddings for similar concepts (e.g., different phrasings of the same skill) and dissimilar embeddings for unrelated concepts. This involved preparing pairs or triplets of data samples (anchor, positive, negative examples) and training the model to minimize the distance between anchor-positive pairs while maximizing the distance between anchor-negative pairs. The specific implementation details of the contrastive learning setup are part of the model training scripts (not detailed in `sentence2vec_utils.py` itself, but this file uses the *output* of that fine-tuning).

### 5.3 Vanilla Sentence2Vec for Soft Skills

For soft skills (both for jobs and candidates), we used a vanilla (non-fine-tuned) Sentence2Vec model, likely a standard pre-trained model from the `sentence-transformers` library (e.g., `all-MiniLM-L6-v2` before our fine-tuning). This approach was deemed sufficient for capturing the general semantics of soft skills without requiring specialized fine-tuning.

The `encode` method from the `Sentence2VecEncoder` class (shown above) is used to generate these embeddings for all text inputs, whether using the fine-tuned model for hard skills or a vanilla model for soft skills (by initializing the encoder with the appropriate model path).

## 6. MongoDB Integration

All processed textual data and their corresponding embeddings are stored in MongoDB. The database structure is designed to efficiently store and retrieve job and candidate information. This structure is detailed in `neural_model/docs/database_structure.md`.

### 6.1 Setting Up MongoDB

MongoDB was chosen for its flexibility with unstructured data and document-oriented approach.

```python
# MongoDB connection code
from utils.mongo_setup import setup_mongodb

# Connect to MongoDB
client, db, collection = setup_mongodb(
    db_name="rl_jobsdb", 
    collection_name="all_jobs"
)
```

### 6.2 Importing Data to MongoDB

The `MongoImporter` class handles loading CSV data into MongoDB:

```python
from utils.mongo_utils import MongoImporter

# Create importer
importer = MongoImporter(
    db_name="rl_jobsdb",
    collection_name="all_jobs"
)

# Import all CSV files from directory
results = importer.import_all_files("kaggle_datasets/")

# Close connection
importer.close()
```

#### How the MongoImporter Works:

1. **Initialization**:
   ```python
   def __init__(self, mongo_uri=None, db_name="rl_jobsdb", collection_name="all_jobs", db_path="../mongo_db/"):
       if mongo_uri is None:
           # Use standard local MongoDB connection
           mongo_uri = "mongodb://localhost:27017/"
           
       # Create database directory
       db_path = Path(db_path)
       db_path.mkdir(parents=True, exist_ok=True)
           
       self.client = pymongo.MongoClient(mongo_uri)
       self.db = self.client[db_name]
       self.collection = self.db[collection_name]
   ```

2. **CSV Processing**:
   ```python
   def process_csv_file(self, file_path, chunk_size=1000):
       # Read file in chunks to handle large datasets
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           documents = []
           
           for idx, row in chunk.iterrows():
               # Create document from row
               doc = {
                   "doc_id": f"{file_name.split('.')[0]}_{global_idx}",
                   "source_file": file_name,
                   "original_index": original_index,
                   "job_title": job_title,
                   "description": description,
                   "metadata": {...}  # Additional fields
               }
               documents.append(doc)
           
           # Bulk insert documents
           self.collection.insert_many(documents, ordered=False)
   ```

### 6.3 Final MongoDB Schema

The `rl_jobsdb` database in MongoDB now comprises several collections to store textual data and their corresponding embeddings. The primary collections and their schemas are outlined below (based on `neural_model/docs/database_structure.md`):

#### 6.3.1 `jobs_text`
*   **Purpose**: Stores all processed job postings from source datasets, including text extracted by Gemini.
*   **Schema**:
    ```json
    {
        "_id": "ObjectId",
        "doc_id": "String",                  // Custom identifier (e.g., "source_file_index")
        "source_file": "String",             // Original CSV filename
        "original_index": "Number/String",   // Index from source data
        "job_title": "String",
        "description": "String",
        "metadata": "Object",                // Additional fields from source data
        "technical_skills": ["String"],      // Extracted by Gemini
        "soft_skills": ["String"],           // Extracted by Gemini
        "experience_requirements": ["String"] // Extracted by Gemini
    }
    ```

#### 6.3.2 `job_embeddings`
*   **Purpose**: Stores vector embeddings for job titles and skills.
*   **Schema**:
    ```json
    {
        "_id": "ObjectId",
        "original_job_id": "ObjectId",       // Reference to _id in jobs_text
        "job_title_embeddings": ["Number"],  // Vector for job title
        "tech_skills_vectors": ["Number"],   // Aggregated vector for technical skills (fine-tuned model)
        "soft_skills_embeddings": ["Number"],// Aggregated vector for soft skills (vanilla model)
        "experience_requirements_embeddings": ["Number"] // Optional vector for experience
    }
    ```
    *Each embedding vector is 384-dimensional.*

#### 6.3.3 `candidates_text`
*   **Purpose**: Stores textual information for synthetic candidate profiles generated by Gemini.
*   **Schema**:
    ```json
    {
        "_id": "ObjectId",
        "bio": "String",                     // Candidate biography/profile from candidates.json
        "skills": ["String"],                // Combined list of hard and soft skills from candidates.json
        "experience": ["Object"],            // Work experience details (if parsed from resume in candidates.json)
        "hard_skills": ["String"],           // From candidates.json
        "soft_skills": ["String"]            // From candidates.json
    }
    ```
    *(Note: The schema in `database_structure.md` was slightly different; this reflects the structure from `candidates.json` more directly for skills, and `experience` might need further definition based on how it's parsed from resumes).*

#### 6.3.4 `candidates_embeddings`
*   **Purpose**: Stores embedding representations of candidate skills.
*   **Schema**:
    ```json
    {
        "_id": "ObjectId",
        "original_candidate_id": "ObjectId",     // Reference to _id in candidates_text
        "hard_skill_embeddings": ["Number"],     // Aggregated vector for hard skills (fine-tuned model)
        "soft_skills_embeddings": ["Number"]     // Aggregated vector for soft skills (vanilla model)
    }
    ```
    *Each embedding vector is 384-dimensional.*

#### 6.3.5 Collection Relationships
- `job_embeddings.original_job_id` references `jobs_text._id`.
- `candidates_embeddings.original_candidate_id` references `candidates_text._id`.

### 6.4 Iterating Through MongoDB Documents

The `JobIterator` class provides efficient iteration through job documents:

```python
from utils.mongo_utils import JobIterator

# Create iterator (returns documents in batches)
iterator = JobIterator(
    batch_size=100,
    query={"technical_skills": {"$exists": False}}  # Only unprocessed jobs
)

# Process batches of documents
for batch in iterator:
    for job in batch:
        # Process each job document
        print(job["job_title"])
```

## 7. MongoDB Backup and Maintenance

### 7.1 Backing Up MongoDB

```python
from utils.mongo_utils import backup_mongodb

# Backup the database
backup_mongodb(
    db_name="rl_jobsdb", 
    backup_path="../mongo_db/"
)
```

### 7.2 MongoDB Database Structure (Summary)

- **Database**: `rl_jobsdb`
- **Collections**:
  - `all_jobs`: Main collection with processed job postings
  - `test_case`: Collection for testing and validation

## 8. Next Steps

1. **Refine Feature Engineering**: Further refine how aggregated skill embeddings are created and combined for jobs and candidates.
2. **Develop Matching Algorithms**: Implement algorithms to match candidates to jobs based on their textual data and embeddings. This includes exploring similarity metrics for hard skills (fine-tuned embeddings) and soft skills (vanilla embeddings).
3. **Train Recommendation Models**: Train ML models using the structured job data, candidate profiles, and their embeddings to power a skill recommendation system.
4. **Build RL Environment**: Design and implement a reinforcement learning environment where an agent can learn optimal job application or skill acquisition strategies.
5. **API Development**: Create APIs to serve matching results and recommendations.
6. **Testing and Validation**: Rigorously test all components, including data integrity, embedding quality, and model performance.

## 9. Technical Challenges and Solutions

### Challenge 1: Handling Large CSV Files
- **Solution**: Used pandas' chunk processing to reduce memory usage

### Challenge 2: Rate Limiting LLM API Calls
- **Solution**: Implemented controlled sleep periods between requests

### Challenge 3: Inconsistent Data Formats
- **Solution**: Created a flexible document schema in MongoDB

### Challenge 4: Extracting Structured Data from Job Descriptions
- **Solution**: Utilized Gemini's text understanding capabilities with carefully crafted prompts for job features.

### Challenge 5: Generating Diverse and Realistic Candidate Profiles
- **Solution**: Leveraged Gemini to create synthetic candidate data, with prompts designed to ensure a range of skills and experiences as seen in `data_preprocessing/misc_data/candidates.json`.

### Challenge 6: Effective Semantic Representation of Skills
- **Solution**: Adopted Sentence2Vec. Fine-tuned a model using contrastive learning for nuanced hard skill embeddings, and used a vanilla model for general soft skill embeddings. Code for embedding generation is in `data_preprocessing/preprocessing_utils/sentence2vec_utils.py`.

### Challenge 7: Structuring Data for Efficient Retrieval and Analysis
- **Solution**: Designed a multi-collection MongoDB schema (`jobs_text`, `job_embeddings`, `candidates_text`, `candidates_embeddings`) to separate textual data from embeddings, detailed in `neural_model/docs/database_structure.md`.
