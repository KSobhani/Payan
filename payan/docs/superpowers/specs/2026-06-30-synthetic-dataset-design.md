# Synthetic Dataset Generation & Preprocessing — Design Spec

**Revised: 2026-07-01** (based on supervisor feedback — see revision notes below)

## Context

This is phase 1 of the approved MSc thesis "ارائه یک سیستم پیشنهاددهنده هوشمند برای
تخصیص پروژه‌های پژوهشی به مجریان..." (An LLM-Based Recommender System for Matching
Research Projects to Investigators). Per the approved proposal's Gantt chart, the
first executable step after proposal approval is: **ساخت مجموعه‌داده مصنوعی و
پیش‌پردازش متون** (build the synthetic dataset and preprocess the text), budgeted at
2 months.

### Revision notes (2026-07-01)

Supervisor review of the initial design (2026-06-30) produced the following
requirements, all incorporated below:

1. Specialties must be closely related sub-fields of CS/AI/SE — no unrelated domains
   (medicine, economics, mechanics). Exactly **10 specialties**.
2. Scale up to **100 researchers** (10 per specialty) and **~1300 projects**.
3. Researchers are not uniformly single-specialty; multi-specialty distribution
   is cumulative (see Step 1).
4. Specialty is determined by **both** self-declaration and empirical evidence from
   project history — not self-declaration alone.
5. Each project must have rich content beyond title+abstract: introduction/problem
   statement, literature review, methodology summary, results summary.
6. Projects carry roles: مجری (principal investigator), ناظر (supervisor),
   همکار (collaborator) — not just مجری.

Two additional design decisions made during revision:

- **Difficulty levels retained** (easy/medium/hard) but redefined to reflect the
  new multi-specialty structure (see Step 2).
- **`specialty_weights`** is a post-hoc derived field computed after all projects are
  generated, not populated during researcher generation.

### Known proposal inconsistency (resolved)

The proposal's step-by-step methodology text (50 researchers × 3 projects = 150)
does not match its sample-size section (500 projects, ~167 per difficulty level).
With the supervisor's scale-up to 100 researchers and ~13 projects each, this
inconsistency is superseded. The final target is **100 researchers × 13 projects as
مجری = 1300 projects**.

### SEMAT standard alignment

The user supplied `IRIF_Book_V1.0.pdf` — the official "الگوی ملی اطلاعات پژوهشی
ایران" (SEMAT national data model, 1390). Relevant excerpts reviewed:

- `PERSON_RESEARCHINTEREST` / `..._KEYWORD`: researcher keyword-tagged interests —
  matches our keyword design.
- `PERSON_EXPERTISEANDSKILLS` / `..._KEYWORD`: expertise keywords — same pattern.
- `PERSON_OU.CLAS_ID_SLVL`: academic rank is a controlled-vocabulary code —
  `academic_rank` uses مربی / استادیار / دانشیار / استاد.
- `PROJECT_PERSON.CLAS_ID_ROLE`: person-project role code — مجری / ناظر / همکار,
  forward-compatible with the proposal's later supervisor-recommendation feature.

## Goals

- Produce a reproducible, open, synthetic Persian-language dataset of ~1300 research
  projects assigned to 100 virtual researchers, structurally modeled on SEMAT and
  rich enough for LLM-based semantic matching experiments.
- Produce a cleaned/preprocessed version of project text ready for embedding
  extraction (later phase).
- Keep generation code modular and runnable from a Kaggle-hosted notebook.

## Out of scope

- Train/test split (evaluation phase).
- Embedding extraction, similarity computation, evaluation metrics.
- Any use of real researcher data (privacy; SID/IranDoc referenced for style only).

## Specialty taxonomy (10 domains)

All domains are CS sub-fields with meaningful pairwise overlap, enabling realistic
cross-domain (hard) projects:

| # | Domain (Persian) | Adjacent domains |
|---|---|---|
| 1 | یادگیری ماشین و داده‌کاوی | 2, 3, 4, 10 |
| 2 | پردازش زبان طبیعی | 1, 4 |
| 3 | بینایی ماشین و پردازش تصویر | 1, 4 |
| 4 | هوش مصنوعی و سیستم‌های خبره | 1, 2, 3 |
| 5 | امنیت سایبری و رمزنگاری | 7, 8 |
| 6 | مهندسی نرم‌افزار و معماری سیستم | 8, 9 |
| 7 | شبکه‌های کامپیوتری و انتقال داده | 5, 8, 9 |
| 8 | رایانش ابری و سیستم‌های توزیع‌شده | 6, 7, 9 |
| 9 | اینترنت اشیاء و سیستم‌های نهفته | 7, 8 |
| 10 | پایگاه داده و بازیابی اطلاعات | 1, 4, 6 |

Each domain has an associated keyword bank (~20 candidate keywords) and
title-pattern templates, defined in `src/config.py`.

## Project structure

```
D:\Projects\payan\
├── data/
│   ├── raw/
│   │   ├── researchers.csv          # 100 virtual researchers
│   │   ├── projects.csv             # ~1300 generated projects (rich content)
│   │   └── project_assignments.csv  # researcher–project role mapping
│   └── processed/
│       └── projects_clean.csv       # projects.csv + clean_text column
├── src/
│   ├── config.py          # taxonomy, keyword banks, templates, paths, RNG seed
│   ├── researchers.py     # generate_researchers() — no LLM
│   ├── projects.py        # generate_projects() — GPT-4o, two-phase
│   ├── assignments.py     # assign_roles() — no LLM
│   └── preprocess.py      # normalize / tokenize / stopword-remove (Hazm)
├── notebooks/
│   └── 01_build_dataset.ipynb   # orchestrator, Kaggle-compatible
├── tests/
│   └── test_dataset.py
├── requirements.txt
├── .env.example           # OPENAI_API_KEY=...
└── README.md
```

Generation logic lives in `src/` (testable, diffable); the notebook orchestrates
and saves results. This preserves Kaggle compatibility without losing reviewability.

## Step 1 — Researcher generation (`src/researchers.py`)

No LLM calls — pure controlled randomness (seeded), per the proposal's wording.

### Multi-specialty distribution (cumulative, as specified by supervisor)

From 100 researchers:

| Group | Count | # Specialties |
|---|---|---|
| Single-specialty | 70 | 1 |
| Two-specialty | 10 | exactly 2 |
| Three-specialty | 10 | exactly 3 |
| Four-specialty | 5 | exactly 4 |
| Five-specialty | 5 | exactly 5 |
| **Total** | **100** | |

Additional specialties beyond the primary are sampled from the **adjacent domains**
list (see taxonomy table) to ensure realistic thematic overlap rather than random
cross-domain combinations.

### Self-declared vs. empirical specialty

- `self_declared_specialties`: list of domains the researcher explicitly claims
  (set at generation time, before any project is produced).
- `specialty_weights`: a per-domain score reflecting actual project history — 
  computed **post-hoc** after all projects are generated (Step 2), then written
  back to `researchers.csv`. Formula:

  ```
  weight[domain] = (projects_as_manager_in_domain) / (total_projects_as_manager)
  ```

  This enables the "hard" difficulty category (see Step 2): a researcher may have
  non-zero `specialty_weights` in a domain not in their `self_declared_specialties`,
  because they were assigned hard cross-domain projects that got recorded in their
  history.

### Output schema (`data/raw/researchers.csv`)

| field | source |
|---|---|
| `researcher_id` | `Researcher_001`..`Researcher_100` |
| `name` | pseudonym: `پژوهشگر_001`..`پژوهشگر_100` |
| `academic_rank` | sampled from {مربی, استادیار, دانشیار, استاد}, weighted toward استادیار/دانشیار |
| `university` | pseudonym: `دانشگاه_A`..`دانشگاه_J` (10 virtual universities) |
| `department` | derived from primary specialty |
| `self_declared_specialties` | `\|`-joined domain names |
| `research_keywords` | 5 keywords per declared specialty, sampled from domain keyword bank, `\|`-joined |
| `paper_titles` | 15 title-pattern-generated titles covering all declared specialties, `\|`-joined |
| `specialty_weights` | empty at generation; filled after Step 2 (post-hoc) |
| `num_papers` | fixed 15 (matches proposal) |
| `topic_diversity` | controlled-random 0–1 score; higher for multi-specialty researchers |
| `activity_index` | controlled-random 0–1 score (recent publication activity) |

## Step 2 — Project generation (`src/projects.py`, GPT-4o)

### Project count and assignment (100 researchers × 13 = 1300 projects)

Each researcher serves as **مجری** (principal investigator) on exactly 13 projects.
Additionally, most projects also have a **ناظر** and/or **همکار**, assigned from
other researchers (Step 3). The 1300 figure is the count of projects by مجری
assignment, matching the supervisor's target range of 1200–1500.

### Difficulty levels — redefined for multi-specialty structure

| Level | Definition | Matching challenge |
|---|---|---|
| **آسان** | Project in researcher's **primary** declared specialty; direct keyword reuse from their keyword list | TF-IDF baseline sufficient |
| **متوسط** | Project in researcher's **secondary** declared specialty (or primary with vocabulary shift); no shared keywords — paraphrase only | Requires semantic understanding beyond keyword matching |
| **سخت** | Project blends two of the researcher's adjacent specialties (cross-domain), OR is in a specialty present in project history but **not** in self-declaration | Requires inferring from research history; tests LLM advantage over baseline |

For single-specialty researchers, hard projects blend their one specialty with a
randomly chosen adjacent domain.

### Distribution per researcher (13 projects)

| Difficulty | Count | Notes |
|---|---|---|
| آسان | 5 | primary specialty, direct keywords |
| متوسط | 5 | secondary (or paraphrased primary) |
| سخت | 3 | cross-domain or history-only match |

Total across 100 researchers: **500 easy + 500 medium + 300 hard = 1300**.

### Two-phase GPT-4o generation (cost control)

Generating 1300 projects × 6 rich sections in a single prompt each would exceed
context and cost limits. Two-phase approach:

**Phase A — Core generation** (1 API call per project, ~300–400 tokens output):
```json
{
  "title": "...",
  "abstract": "... (100–150 words)",
  "keywords": ["...", "..."],
  "specialty_domain": "..."
}
```

**Phase B — Section expansion** (1 API call per project, ~600–800 tokens output):
Using Phase A's title+abstract as anchor, generate:
```json
{
  "introduction": "... (80–120 words)",
  "literature_review": "... (80–120 words)",
  "methodology_summary": "... (60–80 words)",
  "results_summary": "... (60–80 words)"
}
```

Total: ~2 × 1300 = 2600 API calls. Estimated cost at GPT-4o pricing: **under $15**.

Both phases use a fixed Persian-language system prompt establishing domain (Iranian
academic CS research) and style (formal academic Persian). Difficulty-specific
instructions are injected into the user prompt:

- **آسان**: "از کلیدواژه‌های دقیق پژوهشگر در متن استفاده کن."
- **متوسط**: "مفهوماً در حوزه تخصصی پژوهشگر بمان ولی از کلیدواژه‌های مستقیم استفاده نکن."
- **سخت**: "پروژه‌ای بین‌حوزه‌ای بنویس که حوزه‌های [A] و [B] را ترکیب کند."

All calls use structured JSON output mode. `projects.py` writes incrementally to
CSV after each project so a partial run can resume without re-spending API budget.
Simple bounded retry (max 3 attempts) on transient API errors.

### Output schema (`data/raw/projects.csv`)

| field | description |
|---|---|
| `project_id` | sequential id (`PRJ_0001`..`PRJ_1300`) |
| `title` | generated title (Persian) |
| `specialty_domain` | primary domain of the project |
| `abstract` | generated abstract (Persian, ~100–150 words) |
| `introduction` | intro + problem statement (~80–120 words) |
| `literature_review` | prior work summary (~80–120 words) |
| `methodology_summary` | method overview (~60–80 words) |
| `results_summary` | findings/contributions (~60–80 words) |
| `keywords` | generated keywords, `\|`-joined (3–7) |
| `manager_id` | FK → `researchers.csv` (مجری) |
| `difficulty` | `easy` \| `medium` \| `hard` |
| `year` | synthetic year, random in range 1395–1403 (Solar Hijri) |

## Step 3 — Role assignment (`src/assignments.py`)

Assigns ناظر and همکار to projects after all 1300 projects are generated.
No LLM — pure rule-based selection from the researcher pool.

### Role distribution (across 1300 projects)

| Scenario | % | Count |
|---|---|---|
| مجری + ناظر | 55% | ~715 projects |
| مجری + ناظر + ۱ همکار | 30% | ~390 projects |
| مجری only | 15% | ~195 projects |

### Assignment rules

- **ناظر**: must have `academic_rank` ∈ {دانشیار, استاد}; must have the project's
  `specialty_domain` in their `self_declared_specialties` or adjacent domains; must
  not be the same person as مجری.
- **همکار**: sampled from researchers whose specialties are adjacent to the project
  domain; any academic rank; must not be مجری or ناظر for this project.

### Output schema (`data/raw/project_assignments.csv`)

| field | description |
|---|---|
| `project_id` | FK → `projects.csv` |
| `researcher_id` | FK → `researchers.csv` |
| `role` | `مجری` \| `ناظر` \| `همکار` |

After assignments are complete, `specialty_weights` is computed per researcher and
written back to `researchers.csv`.

## Step 4 — Preprocessing (`src/preprocess.py`, Hazm)

Applied to concatenation of `title + abstract + introduction + literature_review +
methodology_summary + results_summary` per project:

1. Character normalization (unify ي/ی and ك/ک, strip diacritics) — Hazm `Normalizer`.
2. Word tokenization — Hazm.
3. Stopword removal — Hazm standard list + ~50-word supplementary manual list
   (curated during implementation).
4. Output joined back as single `clean_text` string.

Dedup/completeness:
- Drop rows missing any of `title` / `abstract` / `keywords`.
- Drop duplicate rows by exact `title` match.

Output: `data/processed/projects_clean.csv` = `projects.csv` + `clean_text` column.

## Validation (`tests/test_dataset.py`)

- `researchers.csv`: exactly 100 rows; all required fields non-null; `specialty_weights`
  sums to 1.0 per researcher (post-hoc fill); multi-specialty counts match distribution
  (70/10/10/5/5).
- `projects.csv`: exactly 1300 rows; difficulty counts 500/500/300; every `manager_id`
  resolves to a row in `researchers.csv`; every researcher has exactly 13 projects
  as manager.
- `project_assignments.csv`: every `project_id` has exactly one `مجری` row; ناظر rows
  satisfy rank constraint (دانشیار/استاد); no researcher appears twice in the same
  project.
- `projects_clean.csv`: no duplicate titles; no empty `clean_text`.

## Dependencies

`openai`, `pandas`, `numpy`, `hazm`, `python-dotenv`, `pytest` (dev).
Captured in `requirements.txt`.

## Cost summary

| Component | Calls | Estimated cost |
|---|---|---|
| Phase A (core generation) | 1300 | ~$4 |
| Phase B (section expansion) | 1300 | ~$9 |
| **Total** | **2600** | **~$13** |
