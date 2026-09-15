# Project Title
[TO BE DECIDED]

## Course
IT135IU – Introduction to Data Science

## Project Objective
This project uses real-world data to answer one actionable decision question for a clearly defined stakeholder.

- **Decision Question:** [TO BE DECIDED]
- **Stakeholder:** [TO BE DECIDED]
- **Success Criteria:** [TO BE DECIDED]
- **Dataset:** [TO BE DECIDED]

## Team Roles

| Member | Role | Main Responsibility |
|---|---|---|
| [Member 1] | Product Owner | Decision question, stakeholder alignment, success criteria, final recommendation |
| [Member 2] | Data Engineer | Data sourcing, loading, merging, validation, working dataset creation |
| [Member 3] | Data Analyst | Profiling, cleaning, descriptive statistics, statistical analysis |
| [Member 4] | Visualisation Lead | Charts, visual communication, presentation figures |
| [Member 5] | Reproducibility Lead | Repository structure, README, environment setup, seeds, rerun reliability |
| [Member 6] | Ethics & Privacy Officer | Source/license tracking, privacy review, bias/limitations, AI-use documentation |

## Project Workflow
Decision Question → Data → Cleaning → Analysis → Findings → Recommendation

## Repository Structure
- `config/`: shared paths and random seed configuration.
- `data/`: raw, interim, and processed datasets (folder structure tracked with `.gitkeep`).
- `notebooks/`: staged notebook workflow from loading to final reproducible analysis.
- `src/`: reusable helper functions for loading, cleaning, analysis, and visualisation.
- `reports/`: output artifacts (`figures/`, `tables/`) and planning/log templates.
- `docs/`: project governance, decision framework, data source records, ethics/privacy, AI usage.
- `presentation/`: slide preparation guidance.

## Data
- **Dataset Name:** [TO BE DECIDED]
- **Publisher:** [TO BE DECIDED]
- **URL:** [TO BE DECIDED]
- **Date Retrieved:** [TO BE DECIDED]
- **Dataset Size (rows/columns):** [TO BE DECIDED]
- **Terms of Use / License:** [TO BE DECIDED]

## Setup
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Running the Project
Use notebooks in this order:

`01_data_loading.ipynb` → `02_data_profiling_cleaning.ipynb` → `03_data_analysis.ipynb` → `04_visualisation.ipynb` → `final_analysis.ipynb`

## Reproducibility
- Environment dependencies are documented in `requirements.txt`.
- A shared random seed is defined in `config/config.py`.
- Paths are relative via `pathlib`, not local absolute machine paths.
- The final notebook must run from first cell to last cell without errors.

## Responsible Data Use
See:
- `docs/ethics_privacy.md`
- `docs/ai_usage.md`
- `LICENSE_NOTES.md`

## Contributors

| Member | Role | Contact |
|---|---|---|
| [Member 1] | Product Owner | [placeholder] |
| [Member 2] | Data Engineer | [placeholder] |
| [Member 3] | Data Analyst | [placeholder] |
| [Member 4] | Visualisation Lead | [placeholder] |
| [Member 5] | Reproducibility Lead | [placeholder] |
| [Member 6] | Ethics & Privacy Officer | [placeholder] |

## Project Checklist

### Question
- [ ] Genuine decision question
- [ ] Stakeholder identified
- [ ] Success criteria defined

### Data
- [ ] Dataset meets required scale
- [ ] Source documented
- [ ] Terms of use documented
- [ ] Required variables available
- [ ] Merge validated

### Cleaning
- [ ] Data profile complete
- [ ] Missing data addressed
- [ ] Duplicates addressed
- [ ] Cleaning log complete

### Analysis
- [ ] Analysis directly addresses decision question
- [ ] Descriptive statistics complete
- [ ] At least one statistical relationship examined
- [ ] Limitations stated
- [ ] No unsupported causal claims

### Visualisation
- [ ] At least 3 relevant figures
- [ ] Axes labelled
- [ ] Units shown
- [ ] Captions written
- [ ] Each figure has a takeaway

### Recommendation
- [ ] Actionable recommendation
- [ ] Evidence stated
- [ ] Assumptions stated
- [ ] Conditions that would change recommendation stated

### Reproducibility
- [ ] Notebook runs from start to finish
- [ ] README complete
- [ ] Environment documented
- [ ] Random seed fixed where needed

### Responsible Use
- [ ] Privacy reviewed
- [ ] Bias/limitation identified
- [ ] Third-party code documented
- [ ] AI use documented
