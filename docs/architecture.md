# HIRESCORE Architecture

**Version:** 1.0

---

# Project Goal

HIRESCORE is an AI-powered resume evaluation platform designed to help job seekers improve their resumes and understand how well they match specific job opportunities.

The platform aims to provide:

- Resume quality evaluation

- ATS compatibility analysis

- Resume-job matching

- Missing skill recommendations

- Personalized resume improvement suggestions

At this stage, the objective is to build a reliable ML/NLP pipeline using publicly available datasets.

---

# Available Datasets

The project currently contains nine processed datasets.

## Job-side datasets

### postings.csv

Primary dataset containing job postings.

Contains:

- Job title

- Description

- Company ID

- Salary information

- Experience level

- Work type

- Skills description

- Views

- Applications

Role:

Core dataset for job information.

---

### companies.csv

Contains company metadata.

Examples:

- Company name

- Description

- Location

- Company size

- Website

Role:

Provides company information for each posting.

---

### company_industries.csv

Maps companies to industries.

Role:

Adds industry information to companies.

---

### salaries.csv

Contains salary information linked using Job ID.

Role:

Enriches job postings with additional salary data.

---

## Candidate-side dataset

### Resume.csv

Contains real resumes.

Includes:

- Resume text

- HTML version

- Resume category

Role:

Primary dataset representing candidates.

This dataset remains completely separate from the job datasets because no relationship exists between resumes and job postings.

---

## Supporting datasets

### job_skills.csv

Contains mappings between job titles and common skills.

Although it cannot be directly merged with postings using IDs, it can later assist in:

- Skill extraction

- Skill normalization

- Resume analysis

---

### skills.csv

Contains a small skill taxonomy.

Role:

Reference vocabulary for skill normalization.

---

## Ignored datasets

### global_ai_jobs.csv

This dataset appears to contain synthetic market indicators such as:

- Automation risk

- Hiring difficulty

- Job security score

These features cannot be verified against real-world postings and therefore will not be used in the core ML pipeline.

---

### industries.csv

Currently unused because no reliable key connects it to the remaining datasets.

---

# Data Architecture

The project follows a star-schema inspired architecture.

## Company Dimension

```

Companies

        +

Company Industries

        ↓

dim_companies

```

---

## Job Dimension

```

Postings

      +

Salaries

      +

dim_companies

      ↓

dim_postings

```

---

## Resume Dimension

```

Resume.csv

      ↓

dim_resumes

```

This table remains independent.

---

## Skill Resources

```

skills.csv

job_skills.csv

```

These are used as supporting resources for skill extraction and normalization.

---

# Master Tables

The project will eventually maintain the following logical tables.

## dim_companies

Contains company information.

Examples:

- Company ID

- Company name

- Industry

- Description

- Size

- Location

---

## dim_postings

Contains enriched job postings.

Examples:

- Job ID

- Company ID

- Job title

- Description

- Salary

- Experience

- Work type

- Skills

- Views

- Applications

---

## dim_resumes

Contains cleaned resume information.

Examples:

- Resume ID

- Resume category

- Resume text

Later this table will also contain extracted features such as:

- Skills

- Education

- Experience

- Certifications

---

# Resume-Job Pair Generation

There is no historical mapping between resumes and jobs.

Therefore, training samples will be generated dynamically.

Each sample will consist of:

```

Resume

      +

Job Posting

      ↓

Resume-Job Pair

```

From every pair, interaction features will be calculated.

Examples include:

- Semantic similarity

- Skill overlap

- Experience alignment

- Education alignment

- Salary compatibility

---

# Planned ML Pipeline

```

Raw Datasets

      │

      ▼

Cleaning

      │

      ▼

Profiling

      │

      ▼

Exploratory Data Analysis

      │

      ▼

Relationship Analysis

      │

      ▼

Master Tables

      │

      ▼

Resume NLP

      │

      ▼

Job NLP

      │

      ▼

Skill Extraction

      │

      ▼

Text Embeddings

      │

      ▼

Resume-Job Pair Generation

      │

      ▼

Feature Engineering

      │

      ▼

Scoring Engine

      │

      ▼

FastAPI Backend

      │

      ▼

React Frontend

```

---

# Planned Product Features

The first version of HIRESCORE will provide:

- Resume quality score

- ATS compatibility score

- Resume-job match score

- Missing skill recommendations

- Resume improvement suggestions

The "HireScore" presented to users will be a composite score derived from these components rather than a true supervised hiring probability.

---

# Known Limitations

Current datasets do not contain:

- Resume-to-job applications

- Interview outcomes

- Hiring decisions

- Candidate success labels

Because of this, supervised prediction of hiring probability is not currently possible.

The project will instead rely on NLP, embeddings, similarity scoring, and engineered features to estimate candidate-job fit.

Future versions may incorporate real application outcomes to train supervised ranking models.

---

# Current Project Status

Completed:

- Dataset collection

- Data cleaning

- Data profiling

- Exploratory data analysis

- Cross-dataset relationship analysis

Next milestone:

Build the master tables and begin feature engineering for resume-job matching.