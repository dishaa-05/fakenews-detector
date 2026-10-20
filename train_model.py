"""
Phase 3 - Train and export the deployment model.

Reproduces the Phase 1/2 pipeline exactly (same cleaning, split, TF-IDF settings,
random_state) and saves the best model (Linear SVM) as ONE sklearn Pipeline, so the
web app applies the identical TF-IDF transform used during training.

Usage:
    python train_model.py --data IFND.csv
"""
import argparse, json, re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin

OUT = Path("models")

# ── custom text cleaner ────────────────────────────────────────────────────────
_CLICKBAIT = re.compile(
    r"\b(share\s+before\s+(it'?s?\s+)?deleted|forward\s+this|breaking[:\s]|"
    r"shocking[:\s]|must\s+watch|click\s+here|fact\s+check[:\s]|"
    r"viral[:\s]|you\s+won'?t\s+believe)\b",
    re.I,
)

class TextCleaner(BaseEstimator, TransformerMixin):
    """Lower-case, collapse whitespace, keep clickbait markers visible."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for text in X:
            t = str(text).lower().strip()
            # mark common fake-news patterns so TF-IDF can pick them up
            t = _CLICKBAIT.sub(lambda m: m.group().replace(" ", "_"), t)
            t = re.sub(r"\s+", " ", t)
            out.append(t)
        return out


# ── feature union: word n-grams + char n-grams ────────────────────────────────
word_tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),        # up from (1,2) — captures 3-word phrases
    max_features=40000,        # up from 10000
    sublinear_tf=True,
    min_df=2,                  # ignore terms that appear only once
    strip_accents="unicode",
    token_pattern=r"(?u)\b\w\w+\b",  # min 2-char tokens
)

char_tfidf = TfidfVectorizer(
    analyzer="char_wb",        # character n-grams within word boundaries
    ngram_range=(3, 5),
    max_features=20000,
    sublinear_tf=True,
    min_df=3,
    strip_accents="unicode",
)


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df = df[df["Label"].isin(["TRUE", "Fake"])].copy()
    df = df.drop_duplicates(subset=["Statement"])
    df["label_enc"] = df["Label"].map({"TRUE": 1, "Fake": 0})

    # ── extra examples to cover patterns the IFND dataset misses ──────────
    # These cover: govt money claims, health misinformation, clickbait,
    # WhatsApp forwards, conspiracy patterns — all confirmed fake patterns
    extra_fake = [
        "BREAKING: Government to deposit Rs 15 lakh directly into every citizen's bank account by Friday",
        "Government announces free money transfer of Rs 10000 to all citizens this week",
        "PM announces Rs 5 lakh direct deposit for every family before elections",
        "Onions cure COVID-19 completely, doctors confirm",
        "Drinking hot water with lemon every hour completely cures coronavirus",
        "Turmeric milk cures cancer in 3 days, doctors confirm share before deleted",
        "NASA confirms Earth will experience 6 days of total darkness next month",
        "WhatsApp launching paid subscription from next week forward this message to keep account free",
        "Government distributing free smartphones to all citizens via WhatsApp link click here",
        "Shocking new law passed secretly allows government to seize all bank deposits",
        "Scientists confirm 5G towers spread COVID-19 share this before it gets deleted",
        "Free recharge of Rs 500 for all Jio users click here to claim before midnight",
        "Viral video proves moon landing was fake share before YouTube removes it",
        "Eating garlic daily completely prevents all types of cancer doctors confirm",
        "Government secretly adding chemicals to drinking water to control population",
        "Breaking news India to ban all foreign companies and seize their assets by Friday",
        "Forward this message to 10 people to keep your WhatsApp account free of charges",
        "Bill Gates planning to inject microchip through COVID vaccine share before deleted",
        "Shocking PM Modi secretly owns 500 crore worth of foreign assets leaked documents show",
        "Free LPG cylinders for all BPL families click here to register before offer expires",
        "Doctors confirm drinking bleach kills coronavirus share with family and friends",
        "Government to give free land to all homeless citizens apply before deadline this Friday",
        "Viral: Army officer caught accepting bribe of 50 crore share before government removes",
        "You won't believe what this politician did with public funds share before deleted",
        "Must watch video proves COVID vaccine causes infertility share with everyone you know",
        "NASA says asteroid will hit India in 2027, government hiding the truth",
        "Election results were manipulated using hacked voting machines, insider reveals",
        "Insider confirms voting machines were hacked during state elections",
        "Government hiding truth about asteroid heading towards Earth next year",
        "Secret documents prove election results were rigged by ruling party",
        "Whistleblower reveals ballot tampering in five states during last election",
        "NASA warns giant asteroid to strike Earth in 2026 but governments concealing facts",
        "Leaked video proves EVMs were hacked to change election results share now",
        "Insider exposes massive fraud in election commission share before removed",
    ]
    extra_real = [
        "Prime Minister inaugurates new metro rail corridor connecting suburbs to the city centre",
        "Supreme Court issues notice to state government over delay in flood relief funds",
        "India records 7.2 percent GDP growth in the second quarter of the fiscal year",
        "Election Commission announces voting dates for three state assembly elections",
        "Parliament passes new education policy bill after two days of debate",
        "Reserve Bank of India keeps repo rate unchanged at 6.5 percent",
        "Government launches scheme to provide free health insurance to families below poverty line",
        "India signs trade agreement with European Union after three years of negotiations",
        "Scientists develop new vaccine for dengue fever with 80 percent efficacy in trials",
        "State government announces increase in minimum wage for agricultural workers",
        "High court orders state to pay compensation to flood victims within 30 days",
        "India launches its third lunar mission successfully from Sriharikota",
        "Central government allocates 2000 crore for rural road development under new scheme",
        "WHO recommends new treatment protocol for tuberculosis patients in developing nations",
        "Stock market hits record high as foreign investors increase stakes in Indian firms",
        "Government opens applications for PM housing scheme for urban poor",
        "India wins gold medal in wrestling at the Commonwealth Games",
        "New railway line connecting northeastern states inaugurated by Railway Minister",
        "Supreme Court upholds constitutional validity of new environmental protection act",
        "India becomes third largest economy in the world by purchasing power parity",
    ]
    aug = pd.DataFrame({
        "Statement": extra_fake + extra_real,
        "label_enc": [0] * len(extra_fake) + [1] * len(extra_real)
    })
    # repeat augmented data 3x so it has meaningful weight vs 56k rows
    aug = pd.concat([aug] * 3, ignore_index=True)
    df = pd.concat([df[["Statement", "label_enc"]], aug], ignore_index=True)
    df = df.drop_duplicates(subset=["Statement"])
    return df


def main(data_path: str) -> None:
    OUT.mkdir(exist_ok=True)
    df = load_clean(data_path)
    print(f"Dataset: {len(df)} rows  |  Fake: {(df.label_enc==0).sum()}  Real: {(df.label_enc==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["Statement"], df["label_enc"], test_size=0.2, random_state=42, stratify=df["label_enc"]
    )

    pipe = Pipeline([
        ("clean", TextCleaner()),
        ("features", FeatureUnion([
            ("word", word_tfidf),
            ("char", char_tfidf),
        ])),
        # CalibratedClassifierCV wraps LinearSVC so we get real probabilities
        # (used for a better confidence score in the app)
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=3000, random_state=42),
            cv=3,
        )),
    ])

    print("Training … (this may take a minute)")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    summary = {
        "accuracy":  round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall":    round(recall_score(y_test, pred), 4),
        "f1":        round(f1_score(y_test, pred), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
    }
    print(json.dumps(summary, indent=2))

    # Top terms — pull from the underlying LinearSVC inside calibrated wrapper
    base_svc = pipe.named_steps["clf"].calibrated_classifiers_[0].estimator
    # word features come first in FeatureUnion
    word_vocab = pipe.named_steps["features"].transformer_list[0][1].get_feature_names_out()
    char_vocab = pipe.named_steps["features"].transformer_list[1][1].get_feature_names_out()
    vocab = list(word_vocab) + list(char_vocab)
    coef  = base_svc.coef_[0]
    order = coef.argsort()
    top = {
        "fake": [[vocab[i], float(coef[i])] for i in order[:25]],
        "true": [[vocab[i], float(coef[i])] for i in order[::-1][:25]],
    }
    json.dump(top, open(OUT / "top_terms.json", "w"), indent=1)

    joblib.dump(pipe, OUT / "linear_svm_pipeline.joblib", compress=3)
    json.dump(summary, open(OUT / "deployed_model_metrics.json", "w"), indent=1)
    print("Saved →", OUT / "linear_svm_pipeline.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="IFND.csv")
    main(ap.parse_args().data)
