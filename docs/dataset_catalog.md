# HireScore Dataset Catalog

Based on the dataset inspection report (17 datasets). All columns, keys, and notes below are derived strictly from the inspection output — nothing is inferred beyond what the report shows.

> Note: two files share the name `job_skills.csv` but have different schemas (one keyed on `job_link` with 12,217 rows, one keyed on `job_id` with 213,768 rows). They are listed separately below and labeled accordingly.

---

## 1. job_market.csv


| Field                 | Details                                                                                                                                                                                                                                                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Raw job listing data with location and salary fields                                                                                                                                                                                                                                                                                                      |
| **Primary Key**       | None identifiable — no unique ID column present                                                                                                                                                                                                                                                                                                           |
| **Foreign Keys**      | None identifiable — no shared key with other datasets                                                                                                                                                                                                                                                                                                     |
| **Important Columns** | `Job Title`, `Salary`, `Monthly Salary`, `City`, `State`                                                                                                                                                                                                                                                                                                  |
| **Use in HireScore**  | **No**                                                                                                                                                                                                                                                                                                                                                    |
| **Reason**            | No identifiable key to join with any other dataset, isolating it from the rest of the pipeline. `Salary` is stored as a string (not a clean numeric), `Monthly Salary` is missing in 338/835 rows, `Locality` is missing in 581/835 rows, and there are 5 duplicate rows. Data quality is too poor and the dataset too disconnected to justify inclusion. |


---

## 2. job_postings.csv


| Field                 | Details                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Purpose**           | Job posting metadata and processing status (scrape pipeline tracking + job attributes)                                                                       |
| **Primary Key**       | `job_link` (unique per posting; row count matches the linked `job_skills.csv`/`job_summary.csv` files exactly)                                               |
| **Foreign Keys**      | None inbound — `job_link` is referenced by `job_skills.csv` (link version) and `job_summary.csv`                                                             |
| **Important Columns** | `job_title`, `company`, `job_location`, `job_level`, `job_type`, `search_position`                                                                           |
| **Use in HireScore**  | **Yes**                                                                                                                                                      |
| **Reason**            | Clean, complete (near-zero missing values), and forms the anchor table joining to skills and summaries via `job_link`. Directly supports job recommendation. |


---

## 3. job_skills.csv (link version — job_link, job_skills)


| Field                 | Details                                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Raw extracted skill text per job posting                                                                                    |
| **Primary Key**       | `job_link` (one row per job, 12,217 rows matches `job_postings.csv`)                                                        |
| **Foreign Keys**      | `job_link` → `job_postings.csv.job_link`                                                                                    |
| **Important Columns** | `job_skills`                                                                                                                |
| **Use in HireScore**  | **Yes**                                                                                                                     |
| **Reason**            | Directly needed for skill extraction and matching against resume skills. Only 5 missing values out of 12,217 — very usable. |


---

## 4. job_summary.csv


| Field                 | Details                                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Full job description/summary text per posting                                                                     |
| **Primary Key**       | `job_link`                                                                                                        |
| **Foreign Keys**      | `job_link` → `job_postings.csv.job_link`                                                                          |
| **Important Columns** | `job_summary`                                                                                                     |
| **Use in HireScore**  | **Yes**                                                                                                           |
| **Reason**            | No missing values, complete text field. Needed for TF-IDF/cosine similarity between resumes and job descriptions. |


---

## 5. global_ai_jobs.csv


| Field                 | Details                                                                                                                                                                                                                                                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | AI job market dataset with compensation, demand, and workplace metrics                                                                                                                                                                                                                                                                          |
| **Primary Key**       | `id`                                                                                                                                                                                                                                                                                                                                            |
| **Foreign Keys**      | None identifiable — no shared key with any other dataset                                                                                                                                                                                                                                                                                        |
| **Important Columns** | `job_role`, `ai_specialization`, `experience_level`, `experience_years`, `salary_usd`, `education_required`, `industry`, `skill_demand_score`                                                                                                                                                                                                   |
| **Use in HireScore**  | **Yes**                                                                                                                                                                                                                                                                                                                                         |
| **Reason**            | Zero missing values across all 35 columns and 90,000 rows — very clean. Even though it can't be joined to the other 16 datasets, it stands alone as a strong training source for salary prediction (`salary_usd`, `experience_years`, `education_required`, `industry`) and adds demand-side context (`skill_demand_score`, `automation_risk`). |


---

## 6. Resume.csv


| Field                 | Details                                                                                                                                                                          |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Resume text corpus with category labels                                                                                                                                          |
| **Primary Key**       | `ID`                                                                                                                                                                             |
| **Foreign Keys**      | None                                                                                                                                                                             |
| **Important Columns** | `Resume_str`, `Category`                                                                                                                                                         |
| **Use in HireScore**  | **Yes**                                                                                                                                                                          |
| **Reason**            | Core dataset for resume analysis, NER, and classification. Complete (0 missing values) with a labeled `Category` field usable for supervised classification (e.g., Naive Bayes). |


---

## 7. companies.csv


| Field                 | Details                                                                                                                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Company profile/master data                                                                                                                                                                                                           |
| **Primary Key**       | `company_id`                                                                                                                                                                                                                          |
| **Foreign Keys**      | None outbound — referenced by `company_industries.csv`, `company_specialities.csv`, `employee_counts.csv`, and `postings.csv`                                                                                                         |
| **Important Columns** | `name`, `description`, `company_size`, `city`, `state`, `country`                                                                                                                                                                     |
| **Use in HireScore**  | **Yes**                                                                                                                                                                                                                               |
| **Reason**            | Central company reference table needed to enrich job postings with company context for job recommendation. `company_size` is missing in 2,774/24,473 rows but the key columns (`company_id`, `name`, `country`) are largely complete. |


---

## 8. company_industries.csv


| Field                 | Details                                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Maps each company to its industry                                                                          |
| **Primary Key**       | `company_id` (24,375 rows, one industry per company)                                                       |
| **Foreign Keys**      | `company_id` → `companies.csv.company_id`                                                                  |
| **Important Columns** | `industry`                                                                                                 |
| **Use in HireScore**  | **Yes**                                                                                                    |
| **Reason**            | Zero missing values, and industry classification supports job recommendation filtering/matching by sector. |


---

## 9. company_specialities.csv


| Field                 | Details                                                                                                                                                                                                                                                                                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Lists specialities associated with each company (many-to-one)                                                                                                                                                                                                                                                                                              |
| **Primary Key**       | None single-column — composite (`company_id`, `speciality`), since a company has many rows                                                                                                                                                                                                                                                                 |
| **Foreign Keys**      | `company_id` → `companies.csv.company_id`                                                                                                                                                                                                                                                                                                                  |
| **Important Columns** | `speciality`                                                                                                                                                                                                                                                                                                                                               |
| **Use in HireScore**  | **No**                                                                                                                                                                                                                                                                                                                                                     |
| **Reason**            | Not essential to the core HireScore pipeline (resume analysis, skill extraction, job recommendation, salary prediction). High cardinality (169,387 rows) with free-text specialities that would need significant normalization before adding value beyond what `company_industries.csv` already provides. Can be revisited as a future enrichment feature. |


---

## 10. employee_counts.csv


| Field                 | Details                                                                                                                                                                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Time-series of employee/follower counts per company                                                                                                                                                                                            |
| **Primary Key**       | None single-column — composite (`company_id`, `time_recorded`)                                                                                                                                                                                 |
| **Foreign Keys**      | `company_id` → `companies.csv.company_id`                                                                                                                                                                                                      |
| **Important Columns** | `employee_count`, `follower_count`, `time_recorded`                                                                                                                                                                                            |
| **Use in HireScore**  | **No**                                                                                                                                                                                                                                         |
| **Reason**            | Out of scope for the current pipeline — this is a company growth/size time-series, not related to resumes, skills, job matching, or salary directly. No missing values, so it's clean, but not needed for the four target use cases right now. |


---

## 11. benefits.csv


| Field                 | Details                                                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Lists benefit types offered per job posting                                                                                                                                                       |
| **Primary Key**       | None single-column — composite (`job_id`, `type`)                                                                                                                                                 |
| **Foreign Keys**      | `job_id` → `postings.csv.job_id`                                                                                                                                                                  |
| **Important Columns** | `type`, `inferred`                                                                                                                                                                                |
| **Use in HireScore**  | **No**                                                                                                                                                                                            |
| **Reason**            | Not core to resume analysis, skill extraction, job recommendation similarity, or salary prediction. Could be a secondary "job attractiveness" feature later, but not needed for the MVP pipeline. |


---

## 12. job_industries.csv


| Field                 | Details                                                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Maps job postings to industries                                                                                                          |
| **Primary Key**       | Composite (`job_id`, `industry_id`)                                                                                                      |
| **Foreign Keys**      | `job_id` → `postings.csv.job_id`; `industry_id` → `industries.csv.industry_id`                                                           |
| **Important Columns** | `industry_id`                                                                                                                            |
| **Use in HireScore**  | **Yes**                                                                                                                                  |
| **Reason**            | Zero missing values, and directly supports filtering/recommending jobs by industry when joined with `industries.csv` and `postings.csv`. |


---

## 13. job_skills.csv (id version — job_id, skill_abr)


| Field                 | Details                                                                                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Maps job postings to abbreviated skill codes                                                                                                                         |
| **Primary Key**       | Composite (`job_id`, `skill_abr`)                                                                                                                                    |
| **Foreign Keys**      | `job_id` → `postings.csv.job_id`; `skill_abr` → `skills.csv.skill_abr`                                                                                               |
| **Important Columns** | `skill_abr`                                                                                                                                                          |
| **Use in HireScore**  | **Yes**                                                                                                                                                              |
| **Reason**            | Zero missing values across 213,768 rows. This is the primary structured skill-to-job mapping table, essential for skill extraction and resume-to-job skill matching. |


---

## 14. salaries.csv


| Field                 | Details                                                                                                                                                                                                                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Salary figures per job posting                                                                                                                                                                                                                                                                      |
| **Primary Key**       | `salary_id`                                                                                                                                                                                                                                                                                         |
| **Foreign Keys**      | `job_id` → `postings.csv.job_id`                                                                                                                                                                                                                                                                    |
| **Important Columns** | `max_salary`, `min_salary`, `med_salary`, `pay_period`, `currency`, `compensation_type`                                                                                                                                                                                                             |
| **Use in HireScore**  | **Yes**                                                                                                                                                                                                                                                                                             |
| **Reason**            | This is the primary target-variable source for salary prediction. Note significant missingness (`med_salary` missing in 33,947/40,785 rows, `max_salary`/`min_salary` missing in 6,838/40,785 rows), so imputation or fallback logic (e.g., using `postings.csv.normalized_salary`) will be needed. |


---

## 15. industries.csv


| Field                 | Details                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Purpose**           | Lookup table of industry names                                                                                                                               |
| **Primary Key**       | `industry_id`                                                                                                                                                |
| **Foreign Keys**      | None outbound — referenced by `job_industries.csv` and (by value) `company_industries.csv`                                                                   |
| **Important Columns** | `industry_name`                                                                                                                                              |
| **Use in HireScore**  | **Yes**                                                                                                                                                      |
| **Reason**            | Small lookup table (422 rows) needed to convert `industry_id` codes into readable industry names for job recommendation. `industry_name` missing in 34 rows. |


---

## 16. skills.csv


| Field                 | Details                                                                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Lookup table of skill abbreviation codes                                                                                                            |
| **Primary Key**       | `skill_abr`                                                                                                                                         |
| **Foreign Keys**      | None outbound — referenced by `job_skills.csv` (id version)                                                                                         |
| **Important Columns** | `skill_name`                                                                                                                                        |
| **Use in HireScore**  | **Yes**                                                                                                                                             |
| **Reason**            | Small, complete lookup table (35 rows, 0 missing) required to decode `skill_abr` codes into human-readable skill names for skill extraction output. |


---

## 17. postings.csv


| Field                 | Details                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Large, detailed job postings dataset with descriptions, salary, and metadata                                                                                                                                                                                                                                                                                                                                                                                                |
| **Primary Key**       | `job_id`                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Foreign Keys**      | `company_id` → `companies.csv.company_id`                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Important Columns** | `title`, `description`, `location`, `formatted_experience_level`, `skills_desc`, `max_salary`, `min_salary`, `med_salary`, `normalized_salary`, `remote_allowed`, `work_type`                                                                                                                                                                                                                                                                                               |
| **Use in HireScore**  | **Yes**                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Reason**            | The richest job postings dataset (123,849 rows, 31 columns) with free-text `description` for NLP/similarity matching and multiple salary fields for salary prediction. Note heavy missingness in several columns (`max_salary`/`min_salary` missing in 94,056/123,849 rows, `med_salary` missing in 117,569/123,849 rows, `skills_desc` missing in 121,410/123,849 rows), so these fields should be treated as sparse/optional signals rather than relied upon universally. |


---

# Recommendations by Use Case

## 1. Resume Analysis

- **Resume.csv** — primary corpus, includes `Resume_str` (raw text) and `Category` (label) for classification/NER.
- **job_summary.csv** — job description text to compare resumes against via TF-IDF/cosine similarity.
- **postings.csv** (`description` column) — secondary source of job description text for the same comparison, at larger scale.

## 2. Skill Extraction

- **job_skills.csv (link version)** — raw `job_skills` text per job, tied to `job_postings.csv`.
- **job_skills.csv (id version)** — structured `skill_abr` codes per job, tied to `postings.csv`.
- **skills.csv** — lookup to decode `skill_abr` into readable skill names.
- **postings.csv** (`skills_desc` column) — supplementary free-text skill descriptions, though sparse (~98% missing).

## 3. Job Recommendation

- **postings.csv** — main job pool (title, description, location, experience level, work type).
- **job_postings.csv** — secondary job pool with `job_level`, `job_type`, `search_position`.
- **companies.csv** — company context (size, location, description).
- **company_industries.csv** + **industries.csv** — industry classification for filtering/matching.
- **job_industries.csv** — links jobs to industries for the same purpose.

## 4. Salary Prediction

- **salaries.csv** — primary salary target table linked to `postings.csv` via `job_id` (note high missingness in `med_salary`).
- **postings.csv** (`max_salary`, `min_salary`, `med_salary`, `normalized_salary`) — alternate/supplementary salary signal, also sparse.
- **global_ai_jobs.csv** — standalone but clean and complete dataset with `salary_usd`, `experience_years`, `education_required`, `industry`, and demand-side features (`skill_demand_score`, `automation_risk`) — strong candidate for training a salary prediction model, even though it can't be joined to the LinkedIn-style tables above.

