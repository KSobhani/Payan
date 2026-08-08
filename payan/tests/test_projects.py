import json
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src import config
from src.researchers import generate_researchers
from src.projects import (
    _build_phase_a_prompt,
    _build_phase_b_prompt,
    _call_gemini,
    generate_projects,
)

FAKE_RESEARCHER = {
    "researcher_id": "Researcher_001",
    "self_declared_specialties": "یادگیری ماشین و داده‌کاوی|رسانه و ارتباطات",
    "research_keywords": "یادگیری عمیق|شبکه عصبی|تشخیص الگو|پیشگویی|سیستم توصیه‌گر|تحلیل محتوا|رسانه اجتماعی|سواد رسانه‌ای|تأثیر رسانه|جریان اطلاعات",
    "paper_titles": "|".join([f"عنوان {i}" for i in range(1, 16)]),
    "academic_rank": "استادیار",
    "department": "یادگیری ماشین و داده‌کاوی",
}

FAKE_PHASE_A = {
    "title": "پیش‌بینی الگوهای رفتاری با یادگیری عمیق",
    "abstract": "این پژوهش روشی برای تشخیص الگو ارائه می‌دهد.",
    "keywords": ["یادگیری عمیق", "تشخیص الگو"],
    "specialty_domain": "یادگیری ماشین و داده‌کاوی",
}

FAKE_PHASE_B = {
    "introduction": "در این پژوهش مسئله تشخیص الگو بررسی می‌شود.",
    "literature_review": "پژوهش‌های پیشین نشان داده‌اند که...",
    "methodology_summary": "از شبکه عصبی استفاده شده است.",
    "results_summary": "نتایج نشان دهنده بهبود دقت است.",
}


def test_phase_a_prompt_easy_uses_keywords():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "easy", "یادگیری ماشین و داده‌کاوی", "تشخیص الگو")
    assert "کلیدواژه" in prompt or "keywords" in prompt.lower()
    assert "تشخیص الگو" in prompt


def test_phase_a_prompt_medium_avoids_direct_keywords():
    prompt = _build_phase_a_prompt(FAKE_RESEARCHER, "medium", "رسانه و ارتباطات", "تحلیل محتوای رسانه‌ای")
    assert "استفاده نکن" in prompt or "واژگان متفاوت" in prompt


def test_phase_a_prompt_hard_blends_two_specialties():
    prompt = _build_phase_a_prompt(
        FAKE_RESEARCHER, "hard",
        "یادگیری ماشین و داده‌کاوی", "تشخیص الگو",
        "رسانه و ارتباطات", "تحلیل محتوای رسانه‌ای",
    )
    assert "رسانه و ارتباطات" in prompt
    assert "یادگیری ماشین" in prompt


def test_phase_b_prompt_references_title_and_abstract():
    prompt = _build_phase_b_prompt(FAKE_PHASE_A["title"], FAKE_PHASE_A["abstract"])
    assert FAKE_PHASE_A["title"] in prompt
    assert "introduction" in prompt or "مقدمه" in prompt


def test_call_gemini_returns_parsed_dict(mock_paths):
    client = MagicMock()
    r = MagicMock()
    r.text = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
    client.models.generate_content.return_value = r
    result = _call_gemini(client, "system prompt", "user prompt")
    assert result == FAKE_PHASE_A


def test_call_gemini_retries_on_json_error(mock_paths):
    client = MagicMock()
    bad = MagicMock()
    bad.text = "not json"
    good = MagicMock()
    good.text = json.dumps(FAKE_PHASE_A)
    client.models.generate_content.side_effect = [bad, good]
    result = _call_gemini(client, "sys", "user", max_retries=2)
    assert result == FAKE_PHASE_A


def _make_mock_client():
    client = MagicMock()
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        r = MagicMock()
        if call_count["n"] % 2 == 1:
            r.text = json.dumps(FAKE_PHASE_A, ensure_ascii=False)
        else:
            r.text = json.dumps(FAKE_PHASE_B, ensure_ascii=False)
        return r

    client.models.generate_content.side_effect = side_effect
    return client, call_count


def test_generate_projects_count(mock_paths):
    researchers_df = generate_researchers()
    client, _ = _make_mock_client()
    df = generate_projects(researchers_df, client=client)
    assert len(df) == 800


def test_generate_projects_difficulty_distribution(mock_paths):
    researchers_df = generate_researchers()
    client, _ = _make_mock_client()
    df = generate_projects(researchers_df, client=client)
    assert (df["difficulty"] == "easy").sum() == 300
    assert (df["difficulty"] == "medium").sum() == 300
    assert (df["difficulty"] == "hard").sum() == 200


def test_generate_projects_schema(mock_paths):
    researchers_df = generate_researchers()
    client, _ = _make_mock_client()
    df = generate_projects(researchers_df, client=client)
    required_cols = [
        "project_id", "title", "specialty_domain", "abstract",
        "introduction", "literature_review", "methodology_summary",
        "results_summary", "keywords", "manager_id", "difficulty", "year",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_projects_resume(mock_paths):
    researchers_df = generate_researchers()
    first_id = researchers_df["researcher_id"].iloc[0]
    existing = pd.DataFrame([{
        "project_id": f"PRJ_{i+1:04d}",
        "title": f"عنوان {i}", "specialty_domain": "یادگیری ماشین و داده‌کاوی",
        "abstract": "چکیده", "introduction": "مقدمه", "literature_review": "پیشینه",
        "methodology_summary": "روش", "results_summary": "نتایج",
        "keywords": "کلید۱", "manager_id": first_id,
        "difficulty": ["easy", "easy", "easy", "medium", "medium", "medium", "hard", "hard"][i],
        "year": 1400, "seq": i,
    } for i in range(8)])
    existing.to_csv(config.PROJECTS_CSV, index=False, encoding="utf-8-sig")

    client, call_count = _make_mock_client()
    df = generate_projects(researchers_df, client=client)
    assert len(df) == 800
    assert call_count["n"] == (800 - 8) * 2
