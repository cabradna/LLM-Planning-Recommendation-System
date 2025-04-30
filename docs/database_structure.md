
# MongoDB Database Structure Overview

## Database: `rl_jobsdb`

### Collections and Document Schemas

#### 1. `jobs_text`
**Purpose**: Primary collection storing all processed job postings from source datasets.

**Schema**:
```
{
    "_id": ObjectId,                   // MongoDB unique identifier
    "doc_id": String,                  // Custom identifier ("source_file_index")
    "source_file": String,             // Original CSV filename
    "original_index": Number/String,   // Index from source data
    "job_title": String,               // Job position title
    "description": String,             // Full job description text
    "metadata": Object,                // Additional fields from source data
    "technical_skills": Array<String>, // List of extracted technical skills
    "soft_skills": Array<String>,      // List of extracted soft skills
    "experience_requirements": Array<String> // Education/experience requirements
}
```

#### 2. `job_embeddings`
**Purpose**: Stores vector embeddings for job titles and skills for semantic matching.

**Schema**:
```
{
    "_id": ObjectId,                   // MongoDB unique identifier
    "original_job_id": ObjectId,       // Reference to job in jobs_text collection
    "job_title_embeddings": Array<Number>, // Vector representation of job title
    "tech_skills_vectors": Array<Number>   // Vector representation of technical skills
    "soft_skills_embeddings": Array<Number> // Vector representation of soft skills
}
```

#### 3. `candidates_text`
**Purpose**: Stores textual candidate information data for job matching.

**Schema**:
```
{
    "_id": ObjectId,                   // MongoDB unique identifier
    "bio": String,                     // Candidate biography/profile
    "skills": Array<String>,           // Candidate skills
    "experience": Array<Object>        // Work experience details
}
```

#### 4. `candidates_embeddings`
**Purpose**: Stores embedding representation of candidate information

**Schema**:
```
{
    "_id": ObjectId,                            // MongoDB unique identifier
    "original_candidate_id": ObjectId,          // Reference to candidate in candidates_text collection
    "hard_skill_embeddings": Array<Number>,     // Embedding representation of handidate technical or hard skills
    "soft_skills_embeddings": Array<Number>      // Embedding representation of candidate hard or soft skills
}
```

### Collection Relationships

- `jobs_text` collection has an index on `_id` field
- `job_embeddings` collection has a field `original_job_id` that references `_id` on `jobs_text` for relating embeddings to textual, human-readable data
- `candidates_text` collection has an index on `_id` field
- `candidates_embeddings` collection has a field `original_candidate_id` that references `_id` on `candidates_text` for relating embeddings to textual, human-readable data