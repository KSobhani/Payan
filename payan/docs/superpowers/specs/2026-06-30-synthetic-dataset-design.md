# Synthetic Dataset Generation & Preprocessing — Design Spec

## Context

This is phase 1 of the approved MSc thesis "ارائه یک سیستم پیشنهاددهنده هوشمند برای
تخصیص پروژه‌های پژوهشی به مجریان..." (An LLM-Based Recommender System for Matching
Research Projects to Investigators). Per the approved proposal's Gantt chart, the
first executable step after proposal approval is: **ساخت مجموعه‌داده مصنوعی و
پیش‌پردازش متون** (build the synthetic dataset and preprocess the text), budgeted at
2 months.

The proposal's methodology (بخش ۵) specifies a 3-step synthetic data construction
process:
1. Generate 50 virtual researchers (`Researcher_001`..`Researcher_050`), each with
   one primary specialty, 5 keywords, and 15 symbolic paper titles, via controlled
   random functions (no LLM).
2. For each researcher, generate research projects at three difficulty levels
   (easy / medium / hard), using GPT-4o with engineered prompts — especially for
   medium and hard abstracts — totaling 500 projects.
3. Quality control (completeness, dedup) and save as CSV.

This dataset is later split 80/20 (train/test, stratified by difficulty) and used to
evaluate three model architectures (TF-IDF baseline, ParsBERT-only, ParsBERT +
structural features) — that evaluation work is a **later** phase, out of scope here.

### Known proposal inconsistency (resolved with user)

The proposal's step-by-step methodology text (50 researchers × 3 projects = 150)
does not match its sample-size section (500 projects, ~167 per difficulty level).
**Resolved**: keep 50 researchers as the base (per the methodology section), and
give each researcher 10 projects distributed across difficulty levels so the total
is exactly 500, split as close to 167/167/166 as possible (see "Difficulty
distribution" below).

### SEMAT standard alignment

The user supplied `IRIF_Book_V1.0.pdf` — the official "الگوی ملی اطلاعات پژوهشی
ایران" (SEMAT national data model, 1390), a 267-page entity-relationship data
dictionary. Relevant excerpts reviewed:
- `PERSON_RESEARCHINTEREST` / `..._KEYWORD`: researchers have keyword-tagged
  research interests — matches our researcher keyword design.
- `PERSON_EXPERTISEANDSKILLS` / `..._KEYWORD`: expertise keywords — same pattern.
- `PERSON_OU.CLAS_ID_SLVL` / `GRADE`: academic rank is a controlled-vocabulary code
  (هیأت علمی مرتبه/پایه), not free text.
- `PROJECT_PERSON.CLAS_ID_ROLE`: person-project assignment carries a role code
  (مجری اصلی / همکار / ناظر), confirming the proposal's "سمت در پروژه" concept.

Two refinements adopted from this:
- `academic_rank` uses the controlled vocabulary: مربی / استادیار / دانشیار / استاد.
- Each project-assignment record carries a `role` field, fixed to `"مجری اصلی"` for
  this dataset (no co-investigators/supervisors are generated), keeping the schema
  forward-compatible with the proposal's later supervisor-recommendation feature.

## Goals

- Produce a reproducible, open, synthetic Persian-language dataset of 500 research
  projects assigned to 50 virtual researchers, structurally modeled on SEMAT, usable
  as the input to later embedding/recommendation experiments.
- Produce a cleaned/preprocessed version of the project text ready for ParsBERT
  embedding extraction (a later phase).
- Keep generation code modular and reusable from a Kaggle-hosted notebook, per the
  user's stated execution environment preference.

## Out of scope

- Train/test split (belongs to the evaluation phase, section 5 of the proposal).
- Embedding extraction, similarity computation, evaluation metrics.
- Any use of real researcher data (the proposal explicitly avoids this for privacy
  reasons; SID/IranDoc are only referenced for *style*, not extraction).

## Project structure

```
D:\Projects\payan\
├── data/
│   ├── raw/
│   │   ├── researchers.csv      # 50 virtual researchers
│   │   └── projects.csv         # 500 raw generated projects
│   └── processed/
│       └── projects_clean.csv   # projects.csv + clean_text column
├── src/
│   ├── config.py        # specialty taxonomy, keyword banks, paths, RNG seed
│   ├── researchers.py   # generate_researchers() — no LLM
│   ├── projects.py      # generate_projects() — calls OpenAI GPT-4o
│   └── preprocess.py    # normalize / tokenize / stopword-remove (Hazm)
├── notebooks/
│   └── 01_build_dataset.ipynb   # orchestrator, Kaggle-compatible
├── tests/
│   └── test_dataset.py  # schema + completeness + count validation
├── requirements.txt
├── .env.example          # OPENAI_API_KEY=...
└── README.md
```

Generation logic lives in `src/` (testable, importable, diffable in git); the
notebook only calls these functions and displays/saves results. This keeps the
Kaggle requirement satisfiable without losing code review-ability.

## Step 1 — Researcher generation (`src/researchers.py`)

No LLM calls — pure controlled randomness (seeded for reproducibility), per the
proposal's explicit wording for this step.

Specialty taxonomy: ~12–15 CS subfields modeled loosely on ACM CCS top-level
categories (e.g., یادگیری ماشین, شبکه‌های کامپیوتری, امنیت اطلاعات, پردازش زبان
طبیعی, بینایی ماشین, سیستم‌های توزیع‌شده, پایگاه داده, مهندسی نرم‌افزار, هوش
مصنوعی, رایانش ابری, اینترنت اشیا, گرافیک کامپیوتری, الگوریتم و نظریه محاسبه).
Each specialty has an associated keyword bank (~15-20 candidate keywords) and a set
of title-pattern templates used to synthesize paper titles.

Per researcher (`Researcher_001`..`Researcher_050`):

| field | source |
|---|---|
| `researcher_id` | `Researcher_001`..`Researcher_050` |
| `specialty` | 1 specialty, sampled from taxonomy |
| `keywords` | 5 keywords sampled from that specialty's keyword bank |
| `paper_titles` | 15 titles generated by combining title-pattern templates with the researcher's keywords (e.g. «بهبود [روش] برای [مسئله] با استفاده از [تکنیک]») |
| `academic_rank` | sampled from {مربی, استادیار, دانشیار, استاد} (SEMAT-aligned controlled vocabulary), weighted toward استادیار/دانشیار |
| `num_papers` | derived from `len(paper_titles)` = 15 (fixed, matches proposal) |
| `topic_diversity` | a controlled-random score (0–1) representing how spread the researcher's 15 papers are across keywords/sub-topics — used later as a structural feature |
| `activity_index` | a controlled-random score representing recent publication activity, used later as a structural feature |

Output: `data/raw/researchers.csv`, one row per researcher; `keywords` and
`paper_titles` stored as `|`-joined strings (simple, avoids CSV-in-CSV escaping
issues; downstream code splits on `|`).

## Step 2 — Project generation (`src/projects.py`, GPT-4o)

### Difficulty distribution (50 researchers × 10 projects = 500)

Each researcher gets a base of 3 easy + 3 medium + 3 hard (9 projects), plus one
"bonus" project whose difficulty rotates by researcher index `i` (0-based):
- `i % 3 == 0` → bonus easy (researchers 0,3,...,48 → 17 researchers)
- `i % 3 == 1` → bonus medium (researchers 1,4,...,49 → 17 researchers)
- `i % 3 == 2` → bonus hard (researchers 2,5,...,47 → 16 researchers)

Totals: **167 easy + 167 medium + 166 hard = 500**, matching the proposal's sample
size section exactly, while keeping "10 projects per researcher" and "3 base
difficulty levels" intact.

### Prompt strategy per difficulty (GPT-4o, structured JSON output)

Every project is generated by GPT-4o (not just medium/hard) — using the same model
for all three keeps text style/quality consistent; only the *prompt* changes:

- **آسان (easy)**: prompt instructs the model to write a title + abstract that
  directly reuses the researcher's own keywords and phrasing from their paper
  titles → high lexical overlap by construction.
- **متوسط (medium)**: prompt instructs the model to stay within the researcher's
  specialty conceptually but explicitly avoid reusing the given keywords verbatim
  (paraphrase) → conceptual overlap, no shared vocabulary.
- **سخت (hard)**: prompt blends the researcher's primary specialty with one other,
  randomly chosen, distant specialty from the taxonomy, asking for an
  interdisciplinary project abstract → requires deep semantic inference to match
  back to the researcher.

Each API call requests structured JSON: `{title, abstract, keywords}` (3-7
keywords). A fixed system prompt establishes domain (Persian academic CS research
proposals) and length constraints (title ~1 line, abstract ~100-180 words, in
Persian).

### Output schema (`data/raw/projects.csv`)

| field | description |
|---|---|
| `project_id` | sequential id |
| `title` | generated title (Persian) |
| `abstract` | generated abstract (Persian) |
| `keywords` | generated keywords, `|`-joined |
| `researcher_id` | assigned investigator (FK to researchers.csv) |
| `role` | fixed `"مجری اصلی"` (SEMAT-aligned, forward-compatible field) |
| `difficulty` | `easy` \| `medium` \| `hard` |

### Cost/reliability notes

~500 short GPT-4o calls, structured output, estimated well under $5 in API cost.
`projects.py` retries on transient API errors (simple bounded retry, not a generic
resilience framework) and writes incrementally so a partial run can resume rather
than re-spending budget from scratch.

## Step 3 — Preprocessing (`src/preprocess.py`, Hazm)

Applied to `title + abstract + keywords` (concatenated) per the proposal's
preprocessing description:
1. Character normalization (unify ي/ی and ك/ک forms, strip diacritics) using Hazm's
   `Normalizer`.
2. Sentence/word tokenization via Hazm.
3. Stopword removal: Hazm's standard Persian stopword list + a supplementary
   50-word manual list of high-frequency, low-information terms (to be curated
   during implementation).
4. Output joined back into a single `clean_text` string column.

Dedup/completeness (also this step, per proposal):
- Drop rows with any of `title`/`abstract`/`keywords` empty.
- Drop duplicate rows by exact `title` match.

Output: `data/processed/projects_clean.csv` = `projects.csv` + `clean_text` column,
post-dedup/completeness filtering.

## Validation (`tests/test_dataset.py`)

Lightweight pytest checks, not full TDD (most logic is stochastic generation or LLM
calls, not unit-testable business logic):
- `researchers.csv` has exactly 50 rows, all required fields non-null, exactly 5
  keywords and 15 paper titles per row.
- `projects.csv` has exactly 500 rows; difficulty counts are 167/167/166; every
  `researcher_id` FK resolves to a row in `researchers.csv`; every researcher has
  exactly 10 assigned projects.
- `projects_clean.csv` has no duplicate titles and no empty `title`/`abstract`/
  `keywords`/`clean_text`.

## Dependencies

`openai`, `pandas`, `numpy`, `hazm`, `python-dotenv`, `pytest` (dev). Captured in
`requirements.txt`.
