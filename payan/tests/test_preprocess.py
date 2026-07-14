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
    SAMPLE_PROJECTS.iloc[[0]],
], ignore_index=True)

INCOMPLETE_PROJECTS = pd.concat([
    SAMPLE_PROJECTS,
    pd.DataFrame([{
        "project_id": "PRJ_0003",
        "title": "",
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
    assert len(result) == 2


def test_arabic_chars_normalized():
    df = SAMPLE_PROJECTS.copy()
    df.loc[0, "abstract"] = "اين يك آزمايش است"
    result = preprocess_projects(df)
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
    tokens = set(result["clean_text"].iloc[0].split())
    common_stopwords = {"این", "یک", "و", "است", "آن"}
    assert len(common_stopwords & tokens) < 3
