# Synthetic Dataset Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that generates 100 synthetic Persian-language researchers and 1300 research projects (with rich content), assigns roles, preprocesses text with Hazm, and validates the complete dataset.

**Architecture:** Five focused `src/` modules (config → researchers → projects → assignments → preprocess) called in order by a Kaggle-compatible notebook; pytest validates schema/counts after generation. Each module is independently testable — GPT-4o calls are mocked in tests so the full test suite runs without an API key.

**Tech Stack:** Python 3.10+, `openai>=1.0`, `pandas>=2.0`, `numpy`, `hazm`, `python-dotenv`, `pytest`

---

## File Map

| File | Responsibility |
|---|---|
| `src/__init__.py` | empty — makes `src` importable |
| `src/config.py` | 10-specialty taxonomy, keyword banks, title templates, all constants and paths |
| `src/researchers.py` | `generate_researchers()` — pure RNG, no LLM |
| `src/projects.py` | `generate_projects()` — GPT-4o two-phase, incremental CSV write |
| `src/assignments.py` | `assign_roles()`, `compute_specialty_weights()` — rule-based, no LLM |
| `src/preprocess.py` | `preprocess_projects()` — Hazm normalize → tokenize → stopword filter |
| `tests/__init__.py` | empty |
| `tests/conftest.py` | shared `mock_paths` fixture that redirects CSVs to `tmp_path` |
| `tests/test_researchers.py` | unit tests for researcher generation |
| `tests/test_projects.py` | unit tests with mocked OpenAI client |
| `tests/test_assignments.py` | unit tests for role assignment and weight computation |
| `tests/test_preprocess.py` | unit tests for text cleaning |
| `tests/test_dataset.py` | post-generation integration tests (run after full pipeline) |
| `notebooks/01_build_dataset.ipynb` | orchestrator — imports src/, runs full pipeline, Kaggle-compatible |
| `requirements.txt` | pinned dependencies |
| `.env.example` | `OPENAI_API_KEY=...` template |

---

## Task 1 — Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
openai>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
hazm>=0.9.0
python-dotenv>=1.0.0
pytest>=7.0.0
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-...
```

- [ ] **Step 3: Create empty init files**

```bash
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example src/__init__.py tests/__init__.py
git commit -m "feat: project scaffold — deps and package init"
```

---

## Task 2 — Specialty Taxonomy (`src/config.py`)

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write config.py**

```python
from pathlib import Path
import random

SEED = 42

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESEARCHERS_CSV = DATA_RAW / "researchers.csv"
PROJECTS_CSV = DATA_RAW / "projects.csv"
ASSIGNMENTS_CSV = DATA_RAW / "project_assignments.csv"
PROJECTS_CLEAN_CSV = DATA_PROCESSED / "projects_clean.csv"

GPT_MODEL = "gpt-4o"

N_RESEARCHERS = 100
PROJECTS_PER_RESEARCHER = 13
DIFFICULTY_DIST = {"easy": 5, "medium": 5, "hard": 3}

ACADEMIC_RANKS = ["مربی", "استادیار", "دانشیار", "استاد"]
RANK_WEIGHTS = [0.10, 0.40, 0.35, 0.15]

SUPERVISOR_RANKS = {"دانشیار", "استاد"}

UNIVERSITIES = [f"دانشگاه_{chr(65 + i)}" for i in range(10)]

# Cumulative multi-specialty distribution: 70×1, 10×2, 10×3, 5×4, 5×5
SPECIALTY_COUNT_DIST = [1] * 70 + [2] * 10 + [3] * 10 + [4] * 5 + [5] * 5

SPECIALTIES = {
    "یادگیری ماشین و داده‌کاوی": {
        "adjacent": [
            "پردازش زبان طبیعی",
            "بینایی ماشین و پردازش تصویر",
            "هوش مصنوعی و سیستم‌های خبره",
            "پایگاه داده و بازیابی اطلاعات",
        ],
        "keywords": [
            "یادگیری عمیق", "شبکه عصبی", "درخت تصمیم", "جنگل تصادفی",
            "ماشین بردار پشتیبان", "خوشه‌بندی", "دسته‌بندی", "رگرسیون",
            "داده‌کاوی", "ویژگی‌سازی", "کاهش ابعاد", "یادگیری تقویتی",
            "یادگیری انتقالی", "پیش‌پردازش داده", "تحلیل مؤلفه اصلی",
            "آموزش ناظر", "آموزش بدون ناظر", "الگوریتم k-means",
            "بیش‌برازش", "مجموعه‌داده برچسب‌دار",
        ],
        "title_templates": [
            "بهبود {method} برای {task} با استفاده از {technique}",
            "ارائه روش {technique} جهت {task} در حوزه {domain}",
            "پیش‌بینی {target} با بهره‌گیری از {method}",
            "مقایسه الگوریتم‌های {method} در مسئله {task}",
            "طراحی {technique} بهینه برای {task}",
        ],
    },
    "پردازش زبان طبیعی": {
        "adjacent": [
            "یادگیری ماشین و داده‌کاوی",
            "هوش مصنوعی و سیستم‌های خبره",
        ],
        "keywords": [
            "تحلیل احساسات", "تشخیص موجودیت", "ترجمه ماشینی", "تولید متن",
            "خلاصه‌سازی", "پاسخ به سؤال", "تشخیص زبان", "تجزیه نحوی",
            "تعبیه کلمه", "مدل زبانی", "توکن‌سازی", "بازیابی اطلاعات",
            "طبقه‌بندی متن", "شناسایی گفتار", "تبدیل متن به گفتار",
            "دسته‌بندی اسناد", "NER", "BERT", "واژه‌سازی", "حذف ایست‌واژه",
        ],
        "title_templates": [
            "تحلیل {task} در متون {domain} با رویکرد {technique}",
            "ارائه مدل {technique} برای {task} در زبان فارسی",
            "بهبود دقت {task} با استفاده از {method}",
            "طراحی سیستم {task} مبتنی بر {technique}",
            "کاربرد {technique} در {task} متون فارسی",
        ],
    },
    "بینایی ماشین و پردازش تصویر": {
        "adjacent": [
            "یادگیری ماشین و داده‌کاوی",
            "هوش مصنوعی و سیستم‌های خبره",
        ],
        "keywords": [
            "تشخیص شیء", "تقسیم‌بندی تصویر", "تشخیص چهره", "ردیابی حرکت",
            "طبقه‌بندی تصویر", "شبکه کانولوشنی", "تشخیص لبه",
            "پردازش تصویر پزشکی", "افزایش داده", "تشخیص نقطه کلیدی",
            "بازسازی سه‌بعدی", "تشخیص متن در تصویر", "انتقال سبک", "GAN",
            "تشخیص ژست", "پردازش ویدئو", "یادگیری چندوجهی",
            "سگمانتیشن معنایی", "OCR", "تشخیص اشیاء دور",
        ],
        "title_templates": [
            "تشخیص {target} در تصاویر با استفاده از {technique}",
            "بهبود دقت {task} با رویکرد {method}",
            "ارائه روش {technique} برای {task} در تصاویر {domain}",
            "طراحی سیستم {task} مبتنی بر شبکه {technique}",
            "کاربرد {technique} در {task}",
        ],
    },
    "هوش مصنوعی و سیستم‌های خبره": {
        "adjacent": [
            "یادگیری ماشین و داده‌کاوی",
            "پردازش زبان طبیعی",
            "بینایی ماشین و پردازش تصویر",
        ],
        "keywords": [
            "سیستم خبره", "استنتاج منطقی", "پایگاه دانش", "موتور استنتاج",
            "منطق فازی", "الگوریتم ژنتیک", "بهینه‌سازی تکاملی",
            "برنامه‌ریزی هوشمند", "عامل هوشمند", "سیستم چندعاملی",
            "استدلال مبتنی بر مورد", "انتولوژی", "شبکه بیزی",
            "سیستم توصیه‌گر", "اتوماسیون", "تصمیم‌گیری هوشمند",
            "یادگیری فعال", "هوش گروهی", "خودکارسازی فرآیند", "یادگیری آنلاین",
        ],
        "title_templates": [
            "طراحی {technique} هوشمند برای {task}",
            "ارائه سیستم خبره {domain} با رویکرد {method}",
            "بهینه‌سازی {task} با استفاده از {technique}",
            "کاربرد {technique} در {task} با رویکرد هوش مصنوعی",
            "توسعه عامل هوشمند برای {task}",
        ],
    },
    "امنیت سایبری و رمزنگاری": {
        "adjacent": [
            "شبکه‌های کامپیوتری و انتقال داده",
            "رایانش ابری و سیستم‌های توزیع‌شده",
        ],
        "keywords": [
            "رمزنگاری متقارن", "رمزنگاری نامتقارن", "امضای دیجیتال",
            "پروتکل امن", "شناسایی نفوذ", "ارزیابی آسیب‌پذیری", "فایروال",
            "احراز هویت", "کنترل دسترسی", "حمله سایبری", "امنیت شبکه",
            "بلاکچین", "مدیریت کلید", "رمزنگاری جستجوپذیر", "حریم خصوصی داده",
            "پروتکل TLS", "رمزنگاری کوانتومی", "هش", "امنیت پروتکل", "نفوذپذیری",
        ],
        "title_templates": [
            "ارائه پروتکل {technique} برای {task}",
            "بهبود امنیت {domain} با استفاده از {method}",
            "طراحی سیستم {task} مبتنی بر {technique}",
            "تحلیل آسیب‌پذیری {domain} و ارائه راه‌حل {technique}",
            "کاربرد {technique} در امنیت {domain}",
        ],
    },
    "مهندسی نرم‌افزار و معماری سیستم": {
        "adjacent": [
            "رایانش ابری و سیستم‌های توزیع‌شده",
            "اینترنت اشیاء و سیستم‌های نهفته",
        ],
        "keywords": [
            "معماری میکروسرویس", "الگوی طراحی", "تست نرم‌افزار",
            "یکپارچه‌سازی مداوم", "DevOps", "مهندسی نیازمندی‌ها",
            "معماری مبتنی بر سرویس", "چابک", "Refactoring", "کیفیت کد",
            "معماری لایه‌ای", "پیاده‌سازی API", "کانتینر", "Docker",
            "معماری رویداد‌محور", "بدهی فنی", "پوشش کد",
            "معماری هگزاگونال", "Domain Driven Design", "SOLID",
        ],
        "title_templates": [
            "ارائه معماری {technique} برای {domain}",
            "بهبود کیفیت {task} با استفاده از {method}",
            "طراحی سیستم {domain} مبتنی بر {technique}",
            "کاربرد {technique} در توسعه {task}",
            "مقایسه معماری‌های {technique} در {domain}",
        ],
    },
    "شبکه‌های کامپیوتری و انتقال داده": {
        "adjacent": [
            "امنیت سایبری و رمزنگاری",
            "رایانش ابری و سیستم‌های توزیع‌شده",
            "اینترنت اشیاء و سیستم‌های نهفته",
        ],
        "keywords": [
            "پروتکل مسیریابی", "شبکه بی‌سیم", "کیفیت سرویس", "مدیریت ترافیک",
            "SDN", "پروتکل TCP/IP", "شبکه 5G", "تأخیر شبکه", "پهنای باند",
            "شبکه توری", "پروتکل OSPF", "بار شبکه", "شبکه اقتضایی", "NFV",
            "مدیریت منابع شبکه", "آنتن MIMO", "شبکه حسگر",
            "تخصیص کانال", "شبکه P2P", "پروتکل HTTP",
        ],
        "title_templates": [
            "بهبود {task} در شبکه‌های {domain} با رویکرد {technique}",
            "ارائه پروتکل {technique} برای {task}",
            "مدیریت {task} در شبکه {domain} با استفاده از {method}",
            "طراحی الگوریتم {technique} برای بهینه‌سازی {task}",
            "کاربرد {technique} در {task} شبکه‌های بی‌سیم",
        ],
    },
    "رایانش ابری و سیستم‌های توزیع‌شده": {
        "adjacent": [
            "مهندسی نرم‌افزار و معماری سیستم",
            "شبکه‌های کامپیوتری و انتقال داده",
            "اینترنت اشیاء و سیستم‌های نهفته",
        ],
        "keywords": [
            "پردازش ابری", "مقیاس‌پذیری", "تحمل خطا", "Kubernetes",
            "زیرساخت بی‌سرور", "محاسبه لبه", "ذخیره‌سازی توزیع‌شده",
            "پردازش موازی", "مجازی‌سازی", "SLA", "تخصیص منابع",
            "مدیریت بار کاری", "پردازش جریان داده", "زمان‌بندی وظایف",
            "فضای ابری ترکیبی", "پردازش دسته‌ای", "اتفاق‌گرایی",
            "MapReduce", "قرارداد هوشمند", "توافق توزیع‌شده",
        ],
        "title_templates": [
            "بهینه‌سازی {task} در محیط {domain} با رویکرد {technique}",
            "ارائه روش {technique} برای مدیریت {task} در ابر",
            "طراحی سیستم توزیع‌شده {domain} مبتنی بر {technique}",
            "کاربرد {technique} در بهبود {task}",
            "مقیاس‌پذیری {domain} با استفاده از {technique}",
        ],
    },
    "اینترنت اشیاء و سیستم‌های نهفته": {
        "adjacent": [
            "شبکه‌های کامپیوتری و انتقال داده",
            "رایانش ابری و سیستم‌های توزیع‌شده",
        ],
        "keywords": [
            "سیستم نهفته", "میکروکنترلر", "پروتکل MQTT", "مدیریت انرژی",
            "حسگر هوشمند", "شهر هوشمند", "خانه هوشمند",
            "بهینه‌سازی مصرف انرژی", "ارتباط ماشین‌به‌ماشین", "RTOS",
            "اینترنت صنعتی", "پردازش لبه‌ای", "امنیت IoT", "پروتکل Zigbee",
            "اتصال‌پذیری", "مانیتورینگ از راه دور", "بروزرسانی OTA",
            "کشاورزی هوشمند", "پوشیدنی هوشمند", "دیجیتال توئین",
        ],
        "title_templates": [
            "طراحی سیستم {task} هوشمند مبتنی بر {technique}",
            "بهینه‌سازی مصرف انرژی در {domain} با رویکرد {technique}",
            "ارائه معماری {technique} برای {task} در محیط IoT",
            "کاربرد {technique} در {domain} هوشمند",
            "مدیریت {task} در سیستم‌های نهفته با {method}",
        ],
    },
    "پایگاه داده و بازیابی اطلاعات": {
        "adjacent": [
            "یادگیری ماشین و داده‌کاوی",
            "هوش مصنوعی و سیستم‌های خبره",
            "مهندسی نرم‌افزار و معماری سیستم",
        ],
        "keywords": [
            "پایگاه داده رابطه‌ای", "پایگاه داده گراف", "بازیابی معنایی",
            "فهرست‌گذاری", "SQL", "پایگاه داده NoSQL", "بهینه‌سازی پرس‌وجو",
            "انبار داده", "OLAP", "پایگاه داده توزیع‌شده", "تراکنش پایگاه داده",
            "نرمال‌سازی", "مدل ER", "موتور جستجو", "بازیابی اطلاعات",
            "رتبه‌بندی نتایج", "فهرست معکوس", "پایگاه داده زمانی",
            "داده‌های حجیم", "ETL",
        ],
        "title_templates": [
            "بهینه‌سازی {task} در پایگاه داده {domain} با رویکرد {technique}",
            "ارائه روش {technique} برای {task}",
            "طراحی سیستم {task} مبتنی بر {technique}",
            "کاربرد {technique} در بهبود {task} پایگاه داده",
            "مقایسه روش‌های {technique} در {task}",
        ],
    },
}

SPECIALTY_LIST = list(SPECIALTIES.keys())
```

- [ ] **Step 2: Smoke-test config import**

```bash
python -c "from src.config import SPECIALTY_LIST, N_RESEARCHERS; print(len(SPECIALTY_LIST), N_RESEARCHERS)"
```

Expected output: `10 100`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: specialty taxonomy and project constants in config.py"
```

---

## Task 3 — Researcher Generation (`src/researchers.py`, TDD)

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_researchers.py`
- Create: `src/researchers.py`

- [ ] **Step 1: Write conftest.py**

```python
# tests/conftest.py
import pytest
from src import config


@pytest.fixture
def mock_paths(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    monkeypatch.setattr(config, "DATA_RAW", raw)
    monkeypatch.setattr(config, "DATA_PROCESSED", processed)
    monkeypatch.setattr(config, "RESEARCHERS_CSV", raw / "researchers.csv")
    monkeypatch.setattr(config, "PROJECTS_CSV", raw / "projects.csv")
    monkeypatch.setattr(config, "ASSIGNMENTS_CSV", raw / "project_assignments.csv")
    monkeypatch.setattr(config, "PROJECTS_CLEAN_CSV", processed / "projects_clean.csv")
    return tmp_path
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_researchers.py
import pytest
import pandas as pd
from src import config
from src.researchers import generate_researchers


@pytest.fixture
def df(mock_paths):
    return generate_researchers()


def test_row_count(df):
    assert len(df) == 100


def test_required_columns(df):
    required = [
        "researcher_id", "name", "academic_rank", "university", "department",
        "self_declared_specialties", "research_keywords", "paper_titles",
        "specialty_weights", "num_papers", "topic_diversity", "activity_index",
    ]
    assert all(c in df.columns for c in required)


def test_researcher_ids(df):
    assert df["researcher_id"].iloc[0] == "Researcher_001"
    assert df["researcher_id"].iloc[-1] == "Researcher_100"


def test_multi_specialty_distribution(df):
    counts = df["self_declared_specialties"].str.split("|").str.len()
    assert (counts == 1).sum() == 70
    assert (counts == 2).sum() == 10
    assert (counts == 3).sum() == 10
    assert (counts == 4).sum() == 5
    assert (counts == 5).sum() == 5


def test_keywords_per_specialty(df):
    for _, row in df.iterrows():
        n_sp = len(row["self_declared_specialties"].split("|"))
        n_kw = len(row["research_keywords"].split("|"))
        assert n_kw == n_sp * 5, f"Researcher {row['researcher_id']}: expected {n_sp*5} keywords, got {n_kw}"


def test_paper_titles_count(df):
    for _, row in df.iterrows():
        assert len(row["paper_titles"].split("|")) == 15


def test_num_papers_fixed(df):
    assert (df["num_papers"] == 15).all()


def test_academic_rank_valid(df):
    assert set(df["academic_rank"]).issubset(set(config.ACADEMIC_RANKS))


def test_specialty_weights_empty_initially(df):
    assert (df["specialty_weights"] == "").all()


def test_topic_diversity_range(df):
    assert df["topic_diversity"].between(0.0, 1.0).all()


def test_activity_index_range(df):
    assert df["activity_index"].between(0.0, 1.0).all()


def test_csv_saved(mock_paths):
    generate_researchers()
    assert config.RESEARCHERS_CSV.exists()


def test_reproducible(mock_paths):
    df1 = generate_researchers()
    df2 = generate_researchers()
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_additional_specialties_from_adjacent(df):
    for _, row in df.iterrows():
        specs = row["self_declared_specialties"].split("|")
        if len(specs) > 1:
            primary = specs[0]
            adjacent = config.SPECIALTIES[primary]["adjacent"]
            all_valid = set(adjacent) | set(config.SPECIALTY_LIST)
            for sp in specs[1:]:
                assert sp in all_valid
```

- [ ] **Step 3: Run tests — verify they all FAIL**

```bash
pytest tests/test_researchers.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.researchers'`

- [ ] **Step 4: Implement src/researchers.py**

```python
# src/researchers.py
import random
import pandas as pd
from src import config


def _make_rng() -> random.Random:
    return random.Random(config.SEED)


def _sample_keywords(rng: random.Random, specialty: str, n: int = 5) -> list[str]:
    pool = config.SPECIALTIES[specialty]["keywords"]
    return rng.sample(pool, min(n, len(pool)))


def _fill_template(template: str, keywords: list[str], specialty: str) -> str:
    domain = specialty.split("و")[0].strip()
    kw = keywords[:3] + keywords[:3]  # pad if needed
    return template.format(
        method=kw[0], task=kw[1], technique=kw[2],
        target=kw[0], domain=domain,
    )


def _generate_paper_titles(rng: random.Random, specialties: list[str], n: int = 15) -> list[str]:
    titles = []
    per_sp = n // len(specialties)
    extra = n % len(specialties)
    for idx, sp in enumerate(specialties):
        count = per_sp + (1 if idx < extra else 0)
        kws = config.SPECIALTIES[sp]["keywords"]
        templates = config.SPECIALTIES[sp]["title_templates"]
        for _ in range(count):
            tmpl = rng.choice(templates)
            sample = rng.sample(kws, 3)
            titles.append(_fill_template(tmpl, sample, sp))
    rng.shuffle(titles)
    return titles[:n]


def _topic_diversity(rng: random.Random, n_specialties: int) -> float:
    base = (n_specialties - 1) / 4.0
    noise = rng.uniform(-0.05, 0.05)
    return round(min(1.0, max(0.0, base + noise)), 3)


def generate_researchers() -> pd.DataFrame:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    rng = _make_rng()

    counts = list(config.SPECIALTY_COUNT_DIST)
    rng.shuffle(counts)

    specialty_list = config.SPECIALTY_LIST
    n_sp = len(specialty_list)
    rows = []

    for i in range(config.N_RESEARCHERS):
        primary = specialty_list[i % n_sp]
        n_specs = counts[i]

        if n_specs == 1:
            specialties = [primary]
        else:
            adjacent = config.SPECIALTIES[primary]["adjacent"]
            pool = [s for s in adjacent if s != primary]
            if len(pool) < n_specs - 1:
                pool = [s for s in specialty_list if s != primary]
            extra = rng.sample(pool, min(n_specs - 1, len(pool)))
            specialties = [primary] + extra

        keywords: list[str] = []
        for sp in specialties:
            keywords.extend(_sample_keywords(rng, sp, 5))

        paper_titles = _generate_paper_titles(rng, specialties, 15)
        rank = rng.choices(config.ACADEMIC_RANKS, weights=config.RANK_WEIGHTS, k=1)[0]

        rows.append({
            "researcher_id": f"Researcher_{i + 1:03d}",
            "name": f"پژوهشگر_{i + 1:03d}",
            "academic_rank": rank,
            "university": rng.choice(config.UNIVERSITIES),
            "department": primary,
            "self_declared_specialties": "|".join(specialties),
            "research_keywords": "|".join(keywords),
            "paper_titles": "|".join(paper_titles),
            "specialty_weights": "",
            "num_papers": 15,
            "topic_diversity": _topic_diversity(rng, len(specialties)),
            "activity_index": round(rng.uniform(0.3, 1.0), 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(config.RESEARCHERS_CSV, index=False, encoding="utf-8-sig")
    return df
```

- [ ] **Step 5: Run tests — all must pass**

```bash
pytest tests/test_researchers.py -v
```

Expected: `13 passed`

- [ ] **Step 6: Commit**

```bash
git add src/researchers.py tests/conftest.py tests/test_researchers.py
git commit -m "feat: researcher generation with multi-specialty distribution"
```

---

## Task 4 — Project Generation (`src/projects.py`, GPT-4o)

**Files:**
- Create: `tests/test_projects.py`
- Create: `src/projects.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_projects.py
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src import config
from src.researchers import generate_researchers
from src.projects import (
    _build_phase_a_prompt,
    _build_phase_b_prompt,
    _call_gpt,
    generate_projects,
)

FAKE_RESEARCHER = {
    "researcher_id": "Researcher_001",
    "self_declared_specialties": "یادگیری ماشین و داده‌کاوی|پردازش زبان طبیعی",
    "research_keywords": "یادگیری عمیق|شبکه عصبی|تحلیل احساسات|تشخیص موجودیت|BERT",
    "paper_titles": "عنوان ۱|عنوان ۲|عنوان ۳",
    "academic_rank": "استادیار",
    "department": "یادگیری ماشین و داده‌کاوی",
}

FAKE_PHASE_A = {
    "title": "پیش‌بینی احساسات با شبکه عصبی",
    "abstract": "این پژوهش روشی برای تحلیل احساسات ارائه می‌دهد.",
    "keywords": ["شبکه عصبی", "تحلیل احساسات"],
    "specialty_domain": "یادگیری ماشین و داده‌کاوی",
}

FAKE_PHASE_B = {
    "introduction": "در این پژوهش مسئله تحلیل احساسات بررسی می‌شود.",
    "literature_review": "پژوهش‌های پیشین نشان داده‌اند که...",
    "methodology_summary": "از شبکه عصبی LSTM استفاده شده است.",
    "results_summary": "نتایج نشان دهنده بهبود ۱۵ درصدی دقت است.",
}


def _mock_openai(phase_a_data=None, phase_b_data=None):
    client = MagicMock()
    responses = []
    if phase_a_data is not None:
        r_a = MagicMock()
        r_a.choices[0].message.content = json.dumps(phase_a_data, ensure_ascii=False)
        responses.append(r_a)
    if phase_b_data is not None:
        r_b = MagicMock()
        r_b.choices[0].message.content = json.dumps(phase_b_data, ensure_ascii=False)
        responses.append(r_b)
    client.chat.completions.create.side_effect = responses
    return client


def test_phase_a_prompt_easy_uses_keywords():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "easy", "یادگیری ماشین و داده‌کاوی")
    assert "کلیدواژه" in prompt
    assert "یادگیری عمیق" in prompt or "research_keywords" in prompt.lower() or "کلیدواژه‌های" in prompt


def test_phase_a_prompt_medium_avoids_direct_keywords():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "medium", "پردازش زبان طبیعی")
    assert "بدون" in prompt or "استفاده نکن" in prompt or "واژگان متفاوت" in prompt


def test_phase_a_prompt_hard_blends_two_specialties():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "hard", "یادگیری ماشین و داده‌کاوی", second_specialty="پردازش زبان طبیعی")
    assert "پردازش زبان طبیعی" in prompt
    assert "یادگیری ماشین" in prompt


def test_phase_b_prompt_references_title_and_abstract():
    prompt = _build_phase_b_prompt(FAKE_PHASE_A["title"], FAKE_PHASE_A["abstract"])
    assert FAKE_PHASE_A["title"] in prompt
    assert "introduction" in prompt or "مقدمه" in prompt


def test_call_gpt_returns_parsed_dict(mock_paths):
    client = _mock_openai(phase_a_data=FAKE_PHASE_A)
    result = _call_gpt(client, "system prompt", "user prompt")
    assert result == FAKE_PHASE_A


def test_call_gpt_retries_on_json_error(mock_paths):
    client = MagicMock()
    bad_response = MagicMock()
    bad_response.choices[0].message.content = "not json"
    good_response = MagicMock()
    good_response.choices[0].message.content = json.dumps(FAKE_PHASE_A)
    client.chat.completions.create.side_effect = [bad_response, good_response]
    result = _call_gpt(client, "sys", "user", max_retries=2)
    assert result == FAKE_PHASE_A


def test_generate_projects_count(mock_paths):
    researchers_df = generate_researchers()
    client = MagicMock()

    def side_effect(*args, **kwargs):
        r = MagicMock()
        # Alternate between Phase A and Phase B responses
        calls = client.chat.completions.create.call_count
        if calls % 2 == 1:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r

    client.chat.completions.create.side_effect = side_effect

    df = generate_projects(researchers_df, client=client)
    assert len(df) == 1300


def test_generate_projects_difficulty_distribution(mock_paths):
    researchers_df = generate_researchers()
    client = MagicMock()

    def side_effect(*args, **kwargs):
        r = MagicMock()
        calls = client.chat.completions.create.call_count
        if calls % 2 == 1:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r

    client.chat.completions.create.side_effect = side_effect
    df = generate_projects(researchers_df, client=client)
    assert (df["difficulty"] == "easy").sum() == 500
    assert (df["difficulty"] == "medium").sum() == 500
    assert (df["difficulty"] == "hard").sum() == 300


def test_generate_projects_schema(mock_paths):
    researchers_df = generate_researchers()
    client = MagicMock()
    r = MagicMock()
    r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
    r2 = MagicMock()
    r2.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
    client.chat.completions.create.side_effect = [r, r2] * 1300

    df = generate_projects(researchers_df, client=client)
    required_cols = [
        "project_id", "title", "specialty_domain", "abstract",
        "introduction", "literature_review", "methodology_summary",
        "results_summary", "keywords", "manager_id", "difficulty", "year",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_projects_resume(mock_paths):
    """If CSV already has some rows, skip them and append only missing ones."""
    researchers_df = generate_researchers()
    # Pre-populate CSV with first researcher's projects
    first_id = researchers_df["researcher_id"].iloc[0]
    existing = pd.DataFrame([{
        "project_id": f"PRJ_{i+1:04d}",
        "title": f"عنوان {i}", "specialty_domain": "یادگیری ماشین و داده‌کاوی",
        "abstract": "چکیده", "introduction": "مقدمه", "literature_review": "پیشینه",
        "methodology_summary": "روش", "results_summary": "نتایج",
        "keywords": "کلید۱", "manager_id": first_id,
        "difficulty": ["easy", "medium", "hard"][i % 3],
        "year": 1400, "seq": i,
    } for i in range(13)])
    existing.to_csv(config.PROJECTS_CSV, index=False, encoding="utf-8-sig")

    call_count = {"n": 0}
    client = MagicMock()
    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        r = MagicMock()
        if call_count["n"] % 2 == 1:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.choices[0].message.content = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r
    client.chat.completions.create.side_effect = side_effect

    df = generate_projects(researchers_df, client=client)
    # Should have skipped 13 existing projects
    assert len(df) == 1300
    # Should only have called API for remaining 99×13×2 = 2574 times
    assert call_count["n"] == (1300 - 13) * 2
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
pytest tests/test_projects.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.projects'`

- [ ] **Step 3: Implement src/projects.py**

```python
# src/projects.py
import json
import random
import time
import pandas as pd
from src import config

SYSTEM_PROMPT = (
    "تو یک پژوهشگر ارشد دانشگاهی ایرانی در حوزه علوم کامپیوتر و مهندسی نرم‌افزار هستی. "
    "وظیفه‌ات تولید طرح‌های پژوهشی واقع‌گرایانه به زبان فارسی رسمی دانشگاهی است. "
    "خروجی را فقط به‌صورت JSON معتبر ارائه بده، بدون توضیح اضافه."
)


def _build_phase_a_prompt(
    researcher: dict,
    difficulty: str,
    specialty: str,
    second_specialty: str | None = None,
) -> str:
    keywords = researcher["research_keywords"].replace("|", "، ")
    titles_sample = "، ".join(researcher["paper_titles"].split("|")[:5])

    if difficulty == "easy":
        return (
            f"پژوهشگر {researcher['researcher_id']} در حوزه «{specialty}» تخصص دارد.\n"
            f"کلیدواژه‌های پژوهشی او: {keywords}\n"
            f"نمونه عناوین مقالات: {titles_sample}\n\n"
            "یک طرح پژوهشی آسان بنویس که:\n"
            "- مستقیماً از کلیدواژه‌های ذکرشده استفاده کند (overlap واژگانی بالا)\n"
            f"- در حوزه اصلی «{specialty}» باشد\n\n"
            "خروجی JSON: {\"title\": \"...\", \"abstract\": \"...(100-150 کلمه)\", "
            "\"keywords\": [...], \"specialty_domain\": \"...\"}"
        )
    elif difficulty == "medium":
        return (
            f"پژوهشگر {researcher['researcher_id']} در حوزه «{specialty}» تخصص دارد.\n"
            f"کلیدواژه‌های پژوهشی او: {keywords}\n\n"
            "یک طرح پژوهشی متوسط بنویس که:\n"
            f"- مفهوماً در حوزه «{specialty}» باشد\n"
            "- از کلیدواژه‌های مستقیم ذکرشده استفاده نکن (واژگان متفاوت، مفهوم مشابه)\n\n"
            "خروجی JSON: {\"title\": \"...\", \"abstract\": \"...(100-150 کلمه)\", "
            "\"keywords\": [...], \"specialty_domain\": \"...\"}"
        )
    else:  # hard
        sp2 = second_specialty or specialty
        return (
            f"یک طرح پژوهشی بین‌رشته‌ای بنویس که حوزه‌های «{specialty}» و «{sp2}» را ترکیب کند.\n"
            "این طرح باید:\n"
            "- ترکیب واقعی و منطقی دو حوزه باشد\n"
            f"- از مفاهیم هر دو حوزه «{specialty}» و «{sp2}» بهره ببرد\n"
            "- چالش اصلی، انگیزه و رویکرد را تشریح کند\n\n"
            "خروجی JSON: {\"title\": \"...\", \"abstract\": \"...(100-150 کلمه)\", "
            "\"keywords\": [...], \"specialty_domain\": \"...\"}"
        )


def _build_phase_b_prompt(title: str, abstract: str) -> str:
    return (
        f"بر اساس طرح پژوهشی زیر، بخش‌های تکمیلی را به فارسی رسمی بنویس:\n"
        f"عنوان: {title}\n"
        f"چکیده: {abstract}\n\n"
        "خروجی JSON:\n"
        "{\"introduction\": \"...(80-120 کلمه)\", "
        "\"literature_review\": \"...(80-120 کلمه)\", "
        "\"methodology_summary\": \"...(60-80 کلمه)\", "
        "\"results_summary\": \"...(60-80 کلمه)\"}"
    )


def _call_gpt(client, system: str, user: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.GPT_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, Exception) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def generate_projects(
    researchers_df: pd.DataFrame,
    client=None,
) -> pd.DataFrame:
    from openai import OpenAI
    import os

    if client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.SEED + 1)

    # Load existing progress
    done: set[tuple] = set()
    if config.PROJECTS_CSV.exists():
        existing = pd.read_csv(config.PROJECTS_CSV)
        for _, row in existing.iterrows():
            done.add((row["manager_id"], row["difficulty"], int(row["seq"])))
    else:
        pd.DataFrame(columns=[
            "project_id", "title", "specialty_domain", "abstract",
            "introduction", "literature_review", "methodology_summary",
            "results_summary", "keywords", "manager_id", "difficulty",
            "year", "seq",
        ]).to_csv(config.PROJECTS_CSV, index=False, encoding="utf-8-sig")

    rows_written: list[dict] = []
    if config.PROJECTS_CSV.exists():
        rows_written = pd.read_csv(config.PROJECTS_CSV).to_dict("records")

    for _, researcher in researchers_df.iterrows():
        rid = researcher["researcher_id"]
        specialties = researcher["self_declared_specialties"].split("|")
        primary = specialties[0]
        secondary = specialties[1] if len(specialties) > 1 else primary

        # Build task list: difficulty × seq
        tasks: list[tuple[str, int, str, str | None]] = []
        for seq in range(5):
            tasks.append(("easy", seq, primary, None))
        for seq in range(5):
            tasks.append(("medium", seq, secondary, None))
        # Hard: pick an adjacent specialty as second
        adjacent = config.SPECIALTIES[primary]["adjacent"]
        hard_pool = [s for s in adjacent if s != primary]
        if not hard_pool:
            hard_pool = [s for s in config.SPECIALTY_LIST if s != primary]
        second_sp = rng.choice(hard_pool)
        for seq in range(3):
            tasks.append(("hard", seq, primary, second_sp))

        for difficulty, seq, specialty, second_specialty in tasks:
            if (rid, difficulty, seq) in done:
                continue

            phase_a_prompt = _build_phase_a_prompt(
                researcher.to_dict(), difficulty, specialty, second_specialty
            )
            phase_a = _call_gpt(client, SYSTEM_PROMPT, phase_a_prompt)

            phase_b_prompt = _build_phase_b_prompt(
                phase_a.get("title", ""), phase_a.get("abstract", "")
            )
            phase_b = _call_gpt(client, SYSTEM_PROMPT, phase_b_prompt)

            row = {
                "project_id": f"PRJ_{len(rows_written) + 1:04d}",
                "title": phase_a.get("title", ""),
                "specialty_domain": phase_a.get("specialty_domain", specialty),
                "abstract": phase_a.get("abstract", ""),
                "introduction": phase_b.get("introduction", ""),
                "literature_review": phase_b.get("literature_review", ""),
                "methodology_summary": phase_b.get("methodology_summary", ""),
                "results_summary": phase_b.get("results_summary", ""),
                "keywords": "|".join(phase_a.get("keywords", [])),
                "manager_id": rid,
                "difficulty": difficulty,
                "year": rng.randint(1395, 1403),
                "seq": seq,
            }
            rows_written.append(row)
            done.add((rid, difficulty, seq))

            # Append single row immediately (incremental write)
            pd.DataFrame([row]).to_csv(
                config.PROJECTS_CSV, mode="a", header=False, index=False, encoding="utf-8-sig"
            )

    return pd.read_csv(config.PROJECTS_CSV)


def get_openai_client():
    """Returns a real OpenAI client. Call this from the notebook."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

- [ ] **Step 4: Run tests — all must pass**

```bash
pytest tests/test_projects.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/projects.py tests/test_projects.py
git commit -m "feat: GPT-4o two-phase project generation with incremental write and resume"
```

---

## Task 5 — Role Assignment + Specialty Weights (`src/assignments.py`, TDD)

**Files:**
- Create: `tests/test_assignments.py`
- Create: `src/assignments.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_assignments.py
import pytest
import pandas as pd
from src import config
from src.researchers import generate_researchers
from src.assignments import assign_roles, compute_specialty_weights


@pytest.fixture
def researchers(mock_paths):
    return generate_researchers()


@pytest.fixture
def projects(researchers):
    """Minimal fake projects DataFrame."""
    rows = []
    for _, r in researchers.iterrows():
        specs = r["self_declared_specialties"].split("|")
        for seq, diff in enumerate(["easy"] * 5 + ["medium"] * 5 + ["hard"] * 3):
            rows.append({
                "project_id": f"PRJ_{len(rows)+1:04d}",
                "title": f"طرح {len(rows)+1}",
                "specialty_domain": specs[0],
                "abstract": "چکیده",
                "introduction": "مقدمه",
                "literature_review": "پیشینه",
                "methodology_summary": "روش",
                "results_summary": "نتایج",
                "keywords": "کلید",
                "manager_id": r["researcher_id"],
                "difficulty": diff,
                "year": 1400,
                "seq": seq,
            })
    return pd.DataFrame(rows)


def test_assignments_row_count(researchers, projects):
    asgn = assign_roles(projects, researchers)
    # Every project must have at least a manager row
    assert len(asgn) >= len(projects)


def test_every_project_has_manager(researchers, projects):
    asgn = assign_roles(projects, researchers)
    mgr = asgn[asgn["role"] == "مجری"]
    assert set(mgr["project_id"]) == set(projects["project_id"])


def test_manager_id_matches_projects(researchers, projects):
    asgn = assign_roles(projects, researchers)
    mgr = asgn[asgn["role"] == "مجری"].set_index("project_id")
    proj = projects.set_index("project_id")
    for pid in proj.index:
        assert mgr.loc[pid, "researcher_id"] == proj.loc[pid, "manager_id"]


def test_supervisor_rank_constraint(researchers, projects):
    asgn = assign_roles(projects, researchers)
    supervisors = asgn[asgn["role"] == "ناظر"]
    if len(supervisors) == 0:
        pytest.skip("No supervisors assigned")
    rank_map = researchers.set_index("researcher_id")["academic_rank"]
    for _, row in supervisors.iterrows():
        rank = rank_map[row["researcher_id"]]
        assert rank in config.SUPERVISOR_RANKS, f"Supervisor {row['researcher_id']} has invalid rank {rank}"


def test_no_duplicate_roles_per_project(researchers, projects):
    asgn = assign_roles(projects, researchers)
    dupes = asgn.groupby(["project_id", "researcher_id"]).size()
    assert (dupes == 1).all(), "Same researcher assigned twice to same project"


def test_role_distribution_approx(researchers, projects):
    asgn = assign_roles(projects, researchers)
    project_roles = asgn.groupby("project_id")["role"].apply(set)
    n_supervisor = sum("ناظر" in roles for roles in project_roles)
    n_collaborator = sum("همکار" in roles for roles in project_roles)
    n_total = len(projects)
    # Supervisor: 55% + 30% = 85% → between 75% and 95%
    assert 0.75 * n_total <= n_supervisor <= 0.95 * n_total
    # Collaborator: ~30% → between 20% and 40%
    assert 0.20 * n_total <= n_collaborator <= 0.45 * n_total


def test_compute_specialty_weights(researchers, projects):
    asgn = assign_roles(projects, researchers)
    updated = compute_specialty_weights(projects, asgn, researchers)
    for _, row in updated.iterrows():
        weights_str = row["specialty_weights"]
        assert weights_str != "", f"Researcher {row['researcher_id']} has empty weights"
        weights = dict(item.split(":") for item in weights_str.split("|"))
        total = sum(float(v) for v in weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1 for {row['researcher_id']}"
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
pytest tests/test_assignments.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.assignments'`

- [ ] **Step 3: Implement src/assignments.py**

```python
# src/assignments.py
import random
import pandas as pd
from src import config


def assign_roles(
    projects_df: pd.DataFrame,
    researchers_df: pd.DataFrame,
) -> pd.DataFrame:
    rng = random.Random(config.SEED + 2)

    rank_map = researchers_df.set_index("researcher_id")["academic_rank"].to_dict()
    specialty_map = researchers_df.set_index("researcher_id")["self_declared_specialties"].to_dict()

    # Pre-build pools: supervisor-eligible researchers per specialty
    supervisor_pool: dict[str, list[str]] = {}
    collaborator_pool: dict[str, list[str]] = {}
    for _, r in researchers_df.iterrows():
        rid = r["researcher_id"]
        rank = r["academic_rank"]
        specs = r["self_declared_specialties"].split("|")
        primary = specs[0]
        adjacent = config.SPECIALTIES[primary]["adjacent"]
        relevant_specs = set(specs) | set(adjacent)

        for sp in relevant_specs:
            if rank in config.SUPERVISOR_RANKS:
                supervisor_pool.setdefault(sp, []).append(rid)
            collaborator_pool.setdefault(sp, []).append(rid)

    rows: list[dict] = []
    for _, project in projects_df.iterrows():
        pid = project["project_id"]
        mgr = project["manager_id"]
        domain = project["specialty_domain"]

        rows.append({"project_id": pid, "researcher_id": mgr, "role": "مجری"})

        # Determine scenario
        roll = rng.random()
        assign_supervisor = roll < 0.85  # 55% + 30%
        assign_collaborator = roll < 0.30

        if assign_supervisor:
            pool = [r for r in supervisor_pool.get(domain, []) if r != mgr]
            if not pool:
                # Fallback: any supervisor-eligible researcher
                pool = [
                    r["researcher_id"] for _, r in researchers_df.iterrows()
                    if r["academic_rank"] in config.SUPERVISOR_RANKS and r["researcher_id"] != mgr
                ]
            if pool:
                supervisor = rng.choice(pool)
                rows.append({"project_id": pid, "researcher_id": supervisor, "role": "ناظر"})

                if assign_collaborator:
                    collab_pool = [
                        r for r in collaborator_pool.get(domain, [])
                        if r != mgr and r != supervisor
                    ]
                    if collab_pool:
                        collaborator = rng.choice(collab_pool)
                        rows.append({"project_id": pid, "researcher_id": collaborator, "role": "همکار"})

    df = pd.DataFrame(rows)
    df.to_csv(config.ASSIGNMENTS_CSV, index=False, encoding="utf-8-sig")
    return df


def compute_specialty_weights(
    projects_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    researchers_df: pd.DataFrame,
) -> pd.DataFrame:
    managers = assignments_df[assignments_df["role"] == "مجری"]
    proj_domain = projects_df.set_index("project_id")["specialty_domain"]

    # For each researcher, count projects managed per domain
    manager_projects = managers.copy()
    manager_projects["specialty_domain"] = manager_projects["project_id"].map(proj_domain)
    counts = (
        manager_projects.groupby(["researcher_id", "specialty_domain"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("researcher_id")["count"].sum()

    weight_strings: dict[str, str] = {}
    for rid, group in counts.groupby("researcher_id"):
        total = totals[rid]
        parts = [f"{row['specialty_domain']}:{row['count'] / total:.3f}" for _, row in group.iterrows()]
        weight_strings[rid] = "|".join(parts)

    df = researchers_df.copy()
    df["specialty_weights"] = df["researcher_id"].map(weight_strings).fillna("")
    df.to_csv(config.RESEARCHERS_CSV, index=False, encoding="utf-8-sig")
    return df
```

- [ ] **Step 4: Run tests — all must pass**

```bash
pytest tests/test_assignments.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/assignments.py tests/test_assignments.py
git commit -m "feat: role assignment with rank constraints and specialty weight computation"
```

---

## Task 6 — Text Preprocessing (`src/preprocess.py`, TDD)

**Files:**
- Create: `tests/test_preprocess.py`
- Create: `src/preprocess.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocess.py
import pytest
import pandas as pd
from src.preprocess import preprocess_projects

SAMPLE_PROJECTS = pd.DataFrame([
    {
        "project_id": "PRJ_0001",
        "title": "یادگیری عمیق برای تشخیص تصویر",
        "abstract": "در این پژوهش از شبکه‌های عصبی عمیق برای تشخیص تصویر استفاده شده است.",
        "introduction": "مسئله تشخیص تصویر یکی از مسائل مهم است.",
        "literature_review": "پژوهش‌های پیشین در این حوزه شامل می‌شود.",
        "methodology_summary": "از مدل ResNet استفاده شده است.",
        "results_summary": "دقت ۹۵ درصد حاصل شد.",
        "keywords": "یادگیری عمیق|تشخیص تصویر",
        "manager_id": "Researcher_001",
        "difficulty": "easy",
        "year": 1400,
        "seq": 0,
        "specialty_domain": "بینایی ماشین و پردازش تصویر",
    },
    {
        "project_id": "PRJ_0002",
        "title": "تحلیل احساسات متون فارسی",
        "abstract": "تحلیل احساسات با استفاده از مدل BERT فارسی.",
        "introduction": "مقدمه‌ای بر تحلیل احساسات.",
        "literature_review": "مرور ادبیات.",
        "methodology_summary": "روش‌شناسی.",
        "results_summary": "نتایج مثبت.",
        "keywords": "تحلیل احساسات|BERT",
        "manager_id": "Researcher_002",
        "difficulty": "medium",
        "year": 1401,
        "seq": 0,
        "specialty_domain": "پردازش زبان طبیعی",
    },
])

DUPLICATE_PROJECTS = pd.concat([
    SAMPLE_PROJECTS,
    SAMPLE_PROJECTS.iloc[[0]],  # duplicate first row
], ignore_index=True)

INCOMPLETE_PROJECTS = pd.concat([
    SAMPLE_PROJECTS,
    pd.DataFrame([{
        "project_id": "PRJ_0003",
        "title": "",  # empty title
        "abstract": "چکیده",
        "introduction": "مقدمه",
        "literature_review": "پیشینه",
        "methodology_summary": "روش",
        "results_summary": "نتایج",
        "keywords": "کلید",
        "manager_id": "Researcher_003",
        "difficulty": "hard",
        "year": 1400,
        "seq": 0,
        "specialty_domain": "یادگیری ماشین و داده‌کاوی",
    }]),
], ignore_index=True)


def test_clean_text_column_added():
    result = preprocess_projects(SAMPLE_PROJECTS)
    assert "clean_text" in result.columns


def test_clean_text_not_empty():
    result = preprocess_projects(SAMPLE_PROJECTS)
    assert result["clean_text"].str.len().gt(0).all()


def test_dedup_by_title():
    result = preprocess_projects(DUPLICATE_PROJECTS)
    assert len(result) == len(SAMPLE_PROJECTS)
    assert result["title"].nunique() == len(result)


def test_drops_empty_title():
    result = preprocess_projects(INCOMPLETE_PROJECTS)
    assert (result["title"] != "").all()
    assert len(result) == 2  # only the two valid rows


def test_arabic_chars_normalized():
    """ي → ی and ك → ک after normalization."""
    df = SAMPLE_PROJECTS.copy()
    df.loc[0, "abstract"] = "اين يك آزمايش است"  # using Arabic ي and ك
    result = preprocess_projects(df)
    # After normalization and tokenization, Arabic forms should not appear
    assert "يك" not in result["clean_text"].iloc[0]


def test_diacritics_removed():
    df = SAMPLE_PROJECTS.copy()
    df.loc[0, "abstract"] = "اَلگوریتمِ یادگیری"
    result = preprocess_projects(df)
    assert "اَ" not in result["clean_text"].iloc[0]


def test_stopwords_removed():
    df = SAMPLE_PROJECTS.copy()
    df.loc[0, "abstract"] = "این یک پژوهش است و نتایج آن نشان می‌دهد"
    result = preprocess_projects(df)
    # Common stopwords like "این", "یک", "و", "است" should be removed
    tokens = set(result["clean_text"].iloc[0].split())
    common_stopwords = {"این", "یک", "و", "است", "آن"}
    assert len(common_stopwords & tokens) < 3
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
pytest tests/test_preprocess.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.preprocess'`

- [ ] **Step 3: Implement src/preprocess.py**

```python
# src/preprocess.py
import pandas as pd
from src import config

CUSTOM_STOPWORDS = {
    "طرح", "پژوهش", "بررسی", "مطالعه", "ارائه", "روش", "استفاده",
    "نتایج", "نشان", "داده", "مورد", "همچنین", "جهت", "برای",
    "حوزه", "زمینه", "رویکرد", "سیستم", "مدل", "الگوریتم",
    "کاربرد", "پیشنهاد", "توسعه", "تحلیل", "بهبود", "افزایش",
    "کاهش", "بهینه", "ساز", "بر", "از", "به", "در", "با",
    "که", "را", "این", "آن", "یک", "ها", "های", "می", "است",
    "شده", "شود", "کند", "دهد", "می‌شود", "می‌کند", "می‌دهد",
}


def preprocess_projects(projects_df: pd.DataFrame) -> pd.DataFrame:
    from hazm import Normalizer, word_tokenize, stopwords_list

    normalizer = Normalizer()
    stopwords = set(stopwords_list()) | CUSTOM_STOPWORDS

    df = projects_df.copy()

    # Drop rows with empty required fields
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    df = df[df["keywords"].notna() & (df["keywords"].str.strip() != "")]

    # Drop duplicate titles
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    def clean_row(row: pd.Series) -> str:
        parts = [
            str(row.get("title", "")),
            str(row.get("abstract", "")),
            str(row.get("introduction", "")),
            str(row.get("literature_review", "")),
            str(row.get("methodology_summary", "")),
            str(row.get("results_summary", "")),
        ]
        text = " ".join(p for p in parts if p and p != "nan")
        text = normalizer.normalize(text)
        tokens = word_tokenize(text)
        tokens = [tok for tok in tokens if tok not in stopwords and len(tok) > 1]
        return " ".join(tokens)

    df["clean_text"] = df.apply(clean_row, axis=1)
    return df


def save_preprocessed(projects_df: pd.DataFrame) -> pd.DataFrame:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = preprocess_projects(projects_df)
    df.to_csv(config.PROJECTS_CLEAN_CSV, index=False, encoding="utf-8-sig")
    return df
```

- [ ] **Step 4: Run tests — all must pass**

```bash
pytest tests/test_preprocess.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: Hazm-based Persian text preprocessing with stopword removal"
```

---

## Task 7 — Full Dataset Validation (`tests/test_dataset.py`)

These tests run **after** the full pipeline has generated all four CSV files.

**Files:**
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write validation tests**

```python
# tests/test_dataset.py
"""
Post-generation integration tests.
Run ONLY after the full pipeline has completed:
  pytest tests/test_dataset.py -v
"""
import pytest
import pandas as pd
from src import config


@pytest.fixture(scope="module")
def researchers():
    if not config.RESEARCHERS_CSV.exists():
        pytest.skip("researchers.csv not found — run pipeline first")
    return pd.read_csv(config.RESEARCHERS_CSV)


@pytest.fixture(scope="module")
def projects():
    if not config.PROJECTS_CSV.exists():
        pytest.skip("projects.csv not found — run pipeline first")
    return pd.read_csv(config.PROJECTS_CSV)


@pytest.fixture(scope="module")
def assignments():
    if not config.ASSIGNMENTS_CSV.exists():
        pytest.skip("project_assignments.csv not found — run pipeline first")
    return pd.read_csv(config.ASSIGNMENTS_CSV)


@pytest.fixture(scope="module")
def projects_clean():
    if not config.PROJECTS_CLEAN_CSV.exists():
        pytest.skip("projects_clean.csv not found — run pipeline first")
    return pd.read_csv(config.PROJECTS_CLEAN_CSV)


# ── Researchers ───────────────────────────────────────────────────────────────

def test_researcher_count(researchers):
    assert len(researchers) == 100


def test_researcher_required_fields(researchers):
    for col in ["researcher_id", "name", "academic_rank", "university", "department",
                 "self_declared_specialties", "research_keywords", "paper_titles",
                 "specialty_weights", "num_papers", "topic_diversity", "activity_index"]:
        assert researchers[col].notna().all(), f"Null values in {col}"


def test_researcher_multi_specialty_distribution(researchers):
    counts = researchers["self_declared_specialties"].str.split("|").str.len()
    assert (counts == 1).sum() == 70
    assert (counts == 2).sum() == 10
    assert (counts == 3).sum() == 10
    assert (counts == 4).sum() == 5
    assert (counts == 5).sum() == 5


def test_specialty_weights_sum_to_one(researchers):
    for _, row in researchers.iterrows():
        weights_str = row["specialty_weights"]
        assert weights_str != "", f"{row['researcher_id']} has empty specialty_weights"
        weights = dict(item.split(":") for item in str(weights_str).split("|"))
        total = sum(float(v) for v in weights.values())
        assert abs(total - 1.0) < 0.01, f"{row['researcher_id']} weights sum to {total}"


# ── Projects ──────────────────────────────────────────────────────────────────

def test_project_count(projects):
    assert len(projects) == 1300


def test_difficulty_distribution(projects):
    assert (projects["difficulty"] == "easy").sum() == 500
    assert (projects["difficulty"] == "medium").sum() == 500
    assert (projects["difficulty"] == "hard").sum() == 300


def test_each_researcher_has_13_projects(projects, researchers):
    per_researcher = projects.groupby("manager_id").size()
    assert (per_researcher == 13).all(), f"Some researchers don't have 13 projects: {per_researcher[per_researcher != 13]}"


def test_manager_ids_valid(projects, researchers):
    valid_ids = set(researchers["researcher_id"])
    assert set(projects["manager_id"]).issubset(valid_ids)


def test_project_required_fields(projects):
    for col in ["project_id", "title", "specialty_domain", "abstract",
                 "introduction", "literature_review", "methodology_summary",
                 "results_summary", "keywords", "manager_id", "difficulty", "year"]:
        assert projects[col].notna().all(), f"Null values in {col}"


# ── Assignments ───────────────────────────────────────────────────────────────

def test_every_project_has_manager_in_assignments(projects, assignments):
    mgr = assignments[assignments["role"] == "مجری"]
    assert set(mgr["project_id"]) == set(projects["project_id"])


def test_supervisor_rank_constraint(assignments, researchers):
    supervisors = assignments[assignments["role"] == "ناظر"]
    rank_map = researchers.set_index("researcher_id")["academic_rank"]
    for _, row in supervisors.iterrows():
        rank = rank_map[row["researcher_id"]]
        assert rank in config.SUPERVISOR_RANKS, f"Invalid supervisor rank: {rank}"


def test_no_researcher_appears_twice_in_same_project(assignments):
    dupes = assignments.groupby(["project_id", "researcher_id"]).size()
    assert (dupes == 1).all()


# ── Preprocessed ─────────────────────────────────────────────────────────────

def test_no_duplicate_titles(projects_clean):
    assert projects_clean["title"].nunique() == len(projects_clean)


def test_no_empty_clean_text(projects_clean):
    assert (projects_clean["clean_text"].str.strip() != "").all()
    assert projects_clean["clean_text"].notna().all()
```

- [ ] **Step 2: Run tests — most should SKIP (CSV files don't exist yet)**

```bash
pytest tests/test_dataset.py -v
```

Expected: all tests show `SKIPPED — run pipeline first`

- [ ] **Step 3: Commit**

```bash
git add tests/test_dataset.py
git commit -m "test: post-generation integration validation suite"
```

---

## Task 8 — Kaggle-Compatible Notebook

**Files:**
- Create: `notebooks/01_build_dataset.ipynb`

- [ ] **Step 1: Create notebook**

Create `notebooks/01_build_dataset.ipynb` with the following cells in order:

**Cell 1 — Setup (Markdown)**
```markdown
# Build Synthetic Dataset
Pipeline: researchers → projects (GPT-4o) → assignments → preprocessing → validation
```

**Cell 2 — Install and imports**
```python
# On Kaggle: uncomment the line below
# !pip install hazm openai python-dotenv -q

import sys
from pathlib import Path

# Works locally (notebooks/) and on Kaggle
repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(repo_root / ".env")

from src import config
from src.researchers import generate_researchers
from src.projects import generate_projects, get_openai_client
from src.assignments import assign_roles, compute_specialty_weights
from src.preprocess import save_preprocessed

print("Imports OK")
print("DATA_RAW:", config.DATA_RAW)
```

**Cell 3 — Step 1: Generate researchers**
```python
print("Generating 100 researchers...")
researchers_df = generate_researchers()
print(f"Done: {len(researchers_df)} researchers")
print(researchers_df[["researcher_id", "academic_rank", "self_declared_specialties"]].head())
```

**Cell 4 — Step 2: Generate projects (GPT-4o)**
```python
print("Generating 1300 projects via GPT-4o (may take 30-60 min)...")
client = get_openai_client()
projects_df = generate_projects(researchers_df, client=client)
print(f"Done: {len(projects_df)} projects")
print(projects_df["difficulty"].value_counts())
```

**Cell 5 — Step 3: Assign roles**
```python
print("Assigning roles (manager / supervisor / collaborator)...")
assignments_df = assign_roles(projects_df, researchers_df)
print(f"Done: {len(assignments_df)} assignment records")
print(assignments_df["role"].value_counts())
```

**Cell 6 — Step 4: Compute specialty weights and write back**
```python
print("Computing specialty weights from project history...")
researchers_df = compute_specialty_weights(projects_df, assignments_df, researchers_df)
print("Sample weights:", researchers_df["specialty_weights"].iloc[0])
```

**Cell 7 — Step 5: Preprocess text**
```python
print("Preprocessing text with Hazm...")
clean_df = save_preprocessed(projects_df)
print(f"Done: {len(clean_df)} clean rows")
print("Sample clean_text:", clean_df["clean_text"].iloc[0][:150])
```

**Cell 8 — Step 6: Validate**
```python
import subprocess
result = subprocess.run(
    ["pytest", "tests/test_dataset.py", "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd=str(repo_root)
)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
```

- [ ] **Step 2: Run cells 1-3 locally (no API call) to verify imports**

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 01_build_dataset.ipynb --output test_run.ipynb 2>&1 | tail -5
```

Expected: cells 1-3 execute without error; cell 4 will fail without an API key (expected).

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_build_dataset.ipynb
git commit -m "feat: Kaggle-compatible orchestrator notebook"
```

---

## Self-Review Against Spec

| Spec requirement | Task(s) covering it |
|---|---|
| 10 CS/AI/SE specialties with adjacency | Task 2 |
| 100 researchers (10 per specialty as primary) | Task 3 |
| Multi-specialty distribution 70/10/10/5/5 | Task 3 + test |
| Secondary specialties from adjacent domains | Task 3 |
| self_declared_specialties field | Task 3 |
| specialty_weights post-hoc from project history | Task 5 |
| researchers.csv schema (all 12 fields) | Task 3 |
| 1300 projects (13 per researcher as مجری) | Task 4 |
| Difficulty distribution 500/500/300 | Task 4 + test |
| Easy: direct keyword reuse prompt | Task 4 |
| Medium: secondary specialty, no keyword overlap | Task 4 |
| Hard: cross-domain blend prompt | Task 4 |
| Two-phase GPT-4o (Phase A + Phase B) | Task 4 |
| Incremental CSV write / resume capability | Task 4 + test |
| projects.csv schema (all 12 fields) | Task 4 |
| Role assignment 55%/30%/15% | Task 5 |
| Supervisor rank constraint (دانشیار/استاد) | Task 5 + test |
| project_assignments.csv schema (3 fields) | Task 5 |
| Hazm normalize → tokenize → stopword | Task 6 |
| Dedup by title + drop empty rows | Task 6 + test |
| projects_clean.csv with clean_text | Task 6 |
| Post-generation validation (all 4 files) | Task 7 |
| Kaggle-compatible notebook | Task 8 |
| Reproducible (seeded RNG) | Task 3 test |
| Cost ~$13 (2600 API calls) | Documented in spec |
