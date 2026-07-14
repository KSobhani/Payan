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

    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    df = df[df["keywords"].notna() & (df["keywords"].str.strip() != "")]

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
