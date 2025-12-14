# Data Directory

```
data/
├── raw/         # encrypted or access-controlled source exports
├── processed/   # de-identified CSVs used in notebooks/models
└── README.md    # this file
```

## Expected files
- `raw/` contains REDCap/Qualtrics exports for questionnaires + gameplay logs (excluded from repo).
- `processed/train.csv` should merge Context/Response text, questionnaires, and metadata columns. Required columns: `Context`, `Response`, plus study variables you plan to analyze.

Document any transformation scripts you run (e.g., `notebooks/01_eda_questionnaires.ipynb`) so the preprocessing is reproducible.
