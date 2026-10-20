"""Fake News Detector - Streamlit app (ML Mini Project, Phase 3)."""
from pathlib import Path
import re

import joblib
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "linear_svm_pipeline.joblib"

# Must be defined here so joblib can deserialise the saved pipeline
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
            t = _CLICKBAIT.sub(lambda m: m.group().replace(" ", "_"), t)
            t = re.sub(r"\s+", " ", t)
            out.append(t)
        return out

st.set_page_config(page_title="TruthLens", page_icon=None, layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html { scroll-behavior: smooth; }
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; color: #0F172A; }
.stApp { background: linear-gradient(180deg, #EEF0FF 0px, #F8F9FF 520px, #FFFFFF 760px); }
[data-testid="stStatusWidget"], [data-testid="stSpinner"] { display: none !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], header[data-testid="stHeader"] { display: none; }
.block-container { max-width: 1180px; padding: 0 2rem 0 2rem; }
h1, h2, h3, h4 { color: #0F172A; letter-spacing: -0.02em; }

/* navbar */
.nav a, .nav a:hover, .nav a:visited, .nav a:active { text-decoration:none !important; border-bottom:none !important; }
.logo, .logo:hover, .logo:visited { color:#0F172A !important; letter-spacing:-0.02em; }
.nav-links a:hover { color:#4F46E5 !important; }
.nav { display:flex; align-items:center; justify-content:space-between; padding: 1.3rem 0; }
.logo { display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.2rem; color:#0F172A; text-decoration:none; }
.logo-mark { width:32px; height:32px; border-radius:9px; background:#4F46E5; display:flex; align-items:center; justify-content:center; }
.nav-links a { color:#475569; font-size:.92rem; font-weight:500; text-decoration:none; margin-left:2rem; }
.nav-links a:hover { color:#4F46E5; }

/* hero */
.badge { display:inline-flex; align-items:center; gap:.5rem; background:#fff; border:1px solid #DDE1F5; color:#4F46E5; font-size:.8rem; font-weight:600; padding:.35rem .8rem; border-radius:999px; }
.badge i { width:7px; height:7px; border-radius:50%; background:#22C55E; display:inline-block; }
.hero-h { font-size:3.3rem; font-weight:800; line-height:1.08; margin:0 0 1.1rem 0; letter-spacing:-0.035em; }
.hero-h span { color:#4F46E5; }
.hero-p { margin-top:1.1rem; color:#475569; font-size:1.12rem; line-height:1.65; max-width:520px; margin-bottom:1.8rem; }
.facts { display:flex; gap:2.4rem; }
.facts b { display:block; font-size:1.6rem; font-weight:800; color:#0F172A; }
.facts span { font-size:.82rem; color:#64748B; }

/* checker card */
.st-key-checker { background:#fff; border:1px solid #E2E8F0; border-radius:18px; padding:1.5rem 1.6rem 1.3rem 1.6rem; box-shadow: 0 20px 45px -18px rgba(79,70,229,.25), 0 2px 6px rgba(15,23,42,.05); }
.card-h { font-weight:700; font-size:1.05rem; margin-bottom:.15rem; }
.card-s { color:#64748B; font-size:.86rem; margin-bottom:.8rem; }
.res-top { display:flex; align-items:center; gap:.7rem; margin-top:.4rem; }
.dot { width:12px; height:12px; border-radius:50%; }
.res-word { font-size:1.55rem; font-weight:800; }
.res-tag { margin-left:auto; font-size:.76rem; font-weight:600; padding:.25rem .7rem; border-radius:999px; }
.real .dot { background:#16A34A; } .real .res-word { color:#15803D; } .real .res-tag { background:#DCFCE7; color:#166534; }
.fake .dot { background:#DC2626; } .fake .res-word { color:#B91C1C; } .fake .res-tag { background:#FEE2E2; color:#991B1B; }
.meter { position:relative; height:8px; border-radius:4px; margin:1rem 0 .35rem 0; background:linear-gradient(90deg,#FCA5A5,#E2E8F0 46%,#E2E8F0 54%,#86EFAC); }
.meter i { position:absolute; top:-5px; width:18px; height:18px; border-radius:50%; background:#0F172A; border:3px solid #fff; box-shadow:0 0 0 1px #CBD5E1; margin-left:-9px; }
.meter-l { display:flex; justify-content:space-between; font-size:.74rem; color:#64748B; }
.chip { display:inline-block; padding:.22rem .7rem; margin:0 .4rem .4rem 0; border-radius:8px; font-size:.82rem; font-weight:500; }
.chip.t { background:#F0FDF4; color:#15803D; border:1px solid #BBF7D0; }
.chip.f { background:#FEF2F2; color:#B91C1C; border:1px solid #FECACA; }
.lab { font-size:.76rem; font-weight:600; color:#64748B; margin:1rem 0 .45rem 0; }
.hint { color:#94A3B8; font-size:.8rem; }
.res-box { min-height:250px; }
.empty { min-height:210px; border:1px dashed #CBD5E1; border-radius:12px; display:flex; align-items:center; justify-content:center; text-align:center; color:#94A3B8; font-size:.9rem; margin-top:1.1rem; padding:0 1rem; }
.err { color:#B45309; background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:.7rem .9rem; font-size:.88rem; margin-top:1.1rem; }

/* sections */
.section { padding-top:4.2rem; }
.eyebrow { color:#4F46E5; font-weight:700; font-size:.8rem; letter-spacing:.1em; text-transform:uppercase; }
.sec-h { font-size:2.1rem; font-weight:800; margin:.4rem 0 .6rem 0; letter-spacing:-0.025em; }
.sec-p { color:#64748B; font-size:1.02rem; max-width:620px; line-height:1.6; margin-bottom:1.6rem; }
.feat { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:1.5rem; height:100%; }
.ico { width:44px; height:44px; border-radius:12px; background:#EEF0FF; display:flex; align-items:center; justify-content:center; margin-bottom:1rem; }
.feat h4 { font-size:1.05rem; font-weight:700; margin:0 0 .4rem 0; }
.feat p { color:#64748B; font-size:.92rem; line-height:1.6; margin:0; }
.stat { background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:1.1rem 1.3rem; }
.stat span { font-size:.78rem; font-weight:600; color:#64748B; }
.stat b { display:block; font-size:1.8rem; font-weight:800; color:#0F172A; margin-top:.1rem; }
.panel { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:1.2rem 1.3rem 0.6rem 1.3rem; }


/* hero extras */
.checks { display:flex; flex-direction:column; gap:.65rem; margin-bottom:.4rem; }
.check { display:flex; align-items:center; gap:.7rem; color:#334155; font-size:.98rem; font-weight:500; }
.check-ico { width:22px; height:22px; border-radius:50%; background:#DCFCE7; display:flex; align-items:center; justify-content:center; flex:none; }
.pv-label { font-size:.76rem; font-weight:600; color:#64748B; letter-spacing:.06em; text-transform:uppercase; margin-top:2rem; }
.preview { position:relative; height:215px; margin-top:.8rem; }
.pcard { position:absolute; width:76%; background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:.95rem 1.05rem; box-shadow:0 14px 30px -14px rgba(15,23,42,.22); }
.pcard.a { left:0; top:0; } .pcard.b { right:0; top:92px; }
.ptag { display:inline-block; font-size:.72rem; font-weight:700; padding:.2rem .6rem; border-radius:999px; margin-bottom:.45rem; }
.ptag.real { background:#DCFCE7; color:#166534; } .ptag.fake { background:#FEE2E2; color:#991B1B; }
.ptxt { color:#334155; font-size:.88rem; line-height:1.5; }
/* topics */
.topics { padding-top:3.4rem; display:flex; align-items:center; gap:1rem 1.2rem; flex-wrap:wrap; }
.topics .t-l { font-size:.85rem; color:#64748B; font-weight:600; margin-right:.4rem; }
.topic { background:#fff; border:1px solid #E2E8F0; border-radius:999px; padding:.4rem 1rem; font-size:.86rem; font-weight:500; color:#334155; }
/* tips */
.tip { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:1.4rem; height:100%; }
.tip .num { font-size:1.9rem; font-weight:800; color:#C7D2FE; line-height:1; margin-bottom:.7rem; }
.tip h4 { font-size:1rem; font-weight:700; margin:0 0 .35rem 0; }
.tip p { color:#64748B; font-size:.9rem; line-height:1.55; margin:0; }
[data-testid="stExpander"] { background:#fff; border:1px solid #E2E8F0 !important; border-radius:12px; margin-bottom:.6rem; }
[data-testid="stExpander"] summary { font-weight:600; }

.grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }
.grid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; }
@media (max-width: 800px) { .grid3, .grid4 { grid-template-columns:1fr; } .hero-h { font-size:2.1rem; } .facts { flex-wrap:wrap; gap:1.2rem; } }

/* footer */
.footer { margin-top:5rem; padding:2rem 0 2.4rem 0; border-top:1px solid #E2E8F0; display:flex; justify-content:space-between; gap:2rem; color:#64748B; font-size:.85rem; }
.footer b { color:#0F172A; }

/* result fade-in */
@keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
.result-fade { animation: fadeUp .35s ease; }

/* widgets */
.stTextArea textarea { font-size:.98rem; border-radius:10px; border:1px solid #CBD5E1; background:#F8FAFC; padding:.8rem .9rem; }
.stTextArea textarea:focus { border-color:#4F46E5; box-shadow:0 0 0 3px rgba(79,70,229,.15); background:#fff; }
.stButton > button { width:100%; border-radius:10px; font-weight:600; border:1px solid #CBD5E1; background:#fff; color:#334155; padding:.5rem .8rem; font-size:.85rem; }
.stButton > button:hover { border-color:#4F46E5; color:#4F46E5; background:#fff; }
.stButton > button[kind="primary"] { background:#4F46E5; border-color:#4F46E5; color:#fff; font-size:.95rem; padding:.62rem 1rem; }
.stButton > button[kind="primary"]:hover { background:#4338CA; border-color:#4338CA; color:#fff; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def analyse(model, text):
    # Support both old model (LinearSVC, no predict_proba) and new calibrated model
    cleaned_text = [text]
    try:
        proba = model.predict_proba(cleaned_text)[0]   # new model: calibrated probs
        margin = float(proba[1] - proba[0])
    except AttributeError:
        # old model: raw SVM decision function, normalise to ~(-1, 1)
        clf = model.named_steps["clf"]
        tfidf = model.named_steps["tfidf"]
        vec = tfidf.transform(cleaned_text)
        raw = float(clf.decision_function(vec)[0])
        margin = max(-3, min(3, raw)) / 3              # normalise to (-1, 1)

    # Word contributions — handle both pipeline shapes
    try:
        # new pipeline: clean → features(word, char) → clf
        cleaner = model.named_steps["clean"]
        cleaned = cleaner.transform(cleaned_text)
        word_tfidf = model.named_steps["features"].transformer_list[0][1]
        vec = word_tfidf.transform(cleaned)
        base_svc = model.named_steps["clf"].calibrated_classifiers_[0].estimator
        vocab = word_tfidf.get_feature_names_out()
    except (KeyError, AttributeError):
        # old pipeline: tfidf → clf
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        vec = tfidf.transform(cleaned_text)
        base_svc = clf
        vocab = tfidf.get_feature_names_out()

    contrib = [(vocab[i], float(vec[0, i] * base_svc.coef_[0][i])) for i in vec.nonzero()[1]]
    contrib.sort(key=lambda x: abs(x[1]), reverse=True)
    return margin, contrib[:8]


def strength(m):
    m = abs(m)
    return "High confidence" if m >= 0.5 else "Moderate confidence" if m >= 0.2 else "Low confidence"


SAMPLES = {
    "Government news": "Prime Minister announces new infrastructure projects worth Rs 1 lakh crore for rural development.",
    "Viral claim": "Fact Check: Viral video claims government is distributing free smartphones to all citizens via WhatsApp link.",
    "COVID claim": "Drinking hot water with lemon every hour completely cures coronavirus, doctors confirm.",
}

# ── rule-based pre-filter ─────────────────────────────────────────────────────
_FAKE_RULES = re.compile(
    r"\b("
    r"share before (it.?s )?deleted|forward this( message)?|"
    r"share with (everyone|all|family)|must watch|"
    r"click here to (claim|register|apply|get)|"
    r"before (it gets|youtube|government|they) (deleted?|removes?|bans?)|"
    r"cures? (cancer|covid|coronavirus|diabetes|all disease)|"
    r"completely cures?|"
    r"doctors confirm.{0,40}(cure|prevent|kill).{0,20}(cancer|covid|virus)|"
    r"(kill|cure|prevent).{0,20}(cancer|covid|coronavirus).{0,20}doctors confirm|"
    r"free (smartphone|recharge|laptop|money|land|cylinder).{0,30}(click|apply|register|whatsapp)|"
    r"rs \d+ (lakh|crore|thousand).{0,30}(every citizen|all citizen|direct.?deposit|your account)|"
    r"deposit.{0,30}(every|all).{0,10}citizen|"
    r"government.{0,40}free (smartphone|money|recharge)|"
    r"microchip.{0,30}(vaccine|inject)|"
    r"5g.{0,30}(covid|spread|cause)|"
    r"moon landing.{0,20}fake|"
    r"chemtrail|"
    r"secret(ly)? (owns?|hid|stole|seize)|"
    r"you won.?t believe|"
    r"shocking.{0,20}(secret|proof|leak|video)|"
    r"(election|voting|ballot).{0,30}(manipulated|hacked|rigged|fraud|tampered)|"
    r"(hacked|rigged).{0,30}(election|voting|ballot|machine)|"
    r"insider (reveals?|confirms?|exposes?).{0,40}(fraud|scam|secret|manipulation)|"
    r"government (hiding|hiding|concealing|suppressing).{0,30}(truth|fact|evidence|proof)|"
    r"(asteroid|meteor|comet).{0,30}(hit|strike|destroy|crash).{0,30}(india|earth|world)|"
    r"(india|earth|world).{0,30}(hit|struck).{0,30}(asteroid|meteor)|"
    r"nasa (confirms?|says|warns?).{0,40}(danger|threat|hit|strike|dark|disaster)"
    r")\b",
    re.I,
)

def rule_check(text: str):
    """Return (margin, reason) if a rule fires, else None."""
    if _FAKE_RULES.search(text):
        return -0.95, "rule"
    return None


def use_sample(txt):
    st.session_state.text = txt
    st.session_state.result = None
    run_check()


def run_check():
    text = st.session_state.get("text", "").strip()
    if len(text.split()) < 3:
        st.session_state.result = {"error": "Please enter at least a few words."}
        return
    # rule filter runs first
    rule = rule_check(text)
    if rule:
        margin, _ = rule
        st.session_state.result = {"margin": margin, "contrib": [], "rule_triggered": True}
        return
    margin, contrib = analyse(model, text)
    st.session_state.result = {"margin": margin, "contrib": contrib, "rule_triggered": False}


st.session_state.setdefault("text", "")
st.session_state.setdefault("result", None)
model = load_model()

# ---------------------------------------------------------------- navbar
st.markdown(
    '<div class="nav"><a class="logo" href="#top"><div class="logo-mark">'
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/><path d="M8.5 11l2 2 3.5-4"/></svg></div>TruthLens</a>'
    '<div class="nav-links"><a href="#check">Check news</a><a href="#how">How it works</a><a href="#tips">Tips</a><a href="#faq">FAQ</a></div></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- hero
st.markdown('<div id="check" style="padding-top:2.2rem"></div>', unsafe_allow_html=True)
hl, hr = st.columns([1.05, 1], gap="large")

with hl:
    st.markdown(
        '<div style="height:1.6rem"></div>'
        '<div class="hero-h">Spot <span>fake news</span><br>before you share it</div>'
        '<div class="badge" style="margin-top:.9rem"><i></i>Trained on 56,340 Indian news statements</div>'
        '<div class="hero-p">Paste any headline or claim and get an instant verdict, along with the words that influenced it.</div>'
        '<div class="checks">'
        '<div class="check"><div class="check-ico">TICK</div>Instant result in under a second</div>'
        '<div class="check"><div class="check-ico">TICK</div>Shows which words drove the decision</div>'
        '<div class="check"><div class="check-ico">TICK</div>No sign-up, nothing is stored</div></div>'
        '<div class="pv-label">Example results</div>'
        '<div class="preview">'
        '<div class="pcard a"><span class="ptag real">Likely real</span><div class="ptxt">Prime Minister announces new infrastructure projects worth Rs 1 lakh crore for rural development.</div></div>'
        '<div class="pcard b"><span class="ptag fake">Likely fake</span><div class="ptxt">Fact Check: Viral video claims government is distributing free smartphones to all citizens via WhatsApp link.</div></div>'
        '</div>'.replace("TICK", """<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>"""),
        unsafe_allow_html=True,
    )

with hr:
    with st.container(key="checker"):
        st.markdown('<div class="card-h">Check a news statement</div>', unsafe_allow_html=True)
        if model is None:
            st.error("Model file not found. Run `python train_model.py --data IFND.csv` once.")
        else:
            st.text_area("Statement", key="text", height=120, label_visibility="collapsed",
                         placeholder="Paste a news headline or claim here...")
            st.button("Analyse →", type="primary", on_click=run_check)
            st.markdown('<div style="font-size:.78rem;font-weight:600;color:#64748B;margin:.6rem 0 .2rem 0;letter-spacing:.04em;text-transform:uppercase;">Try an example</div>', unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            for col, (name, txt) in zip((e1, e2, e3), SAMPLES.items()):
                col.button(name, key=f"s_{name}", on_click=use_sample, args=(txt,))

            r = st.session_state.result
            if r is None:
                box = '<div class="empty">Your result will appear here</div>'
            elif "error" in r:
                box = f'<div class="err">{r["error"]}</div>'
            else:
                margin = r["margin"]
                cls, word = ("real", "Likely real") if margin > 0 else ("fake", "Likely fake")
                pos = (max(-1, min(1, margin)) + 1) / 2 * 90 + 5
                chips = "".join(f'<span class="chip {"t" if c > 0 else "f"}">{w}</span>' for w, c in r["contrib"])
                rule_note = '<div class="hint" style="margin-top:.5rem;color:#B45309">⚠ Flagged by pattern detection</div>' if r.get("rule_triggered") else ""
                box = (f'<div class="result-fade {cls}"><div class="res-top"><div class="dot"></div><div class="res-word">{word}</div>'
                       f'<div class="res-tag">{strength(margin)}</div></div></div>'
                       f'<div class="meter"><i style="left:{pos:.1f}%"></i></div>'
                       '<div class="meter-l"><span>Fake</span><span>Neutral</span><span>Real</span></div>'
                       + rule_note
                       + ('<div class="lab">Words that influenced this result</div>' + chips if chips else ""))
            st.markdown(f'<div class="res-box">{box}</div>', unsafe_allow_html=True)
            st.markdown('<div class="hint" style="margin-top:.6rem">AI can make mistakes. Please verify important claims.</div>',
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- topics
st.markdown('<div class="topics"><span class="t-l">Trained on news about</span>'
            + "".join(f'<span class="topic">{t}</span>' for t in ["Government", "Politics", "Elections", "COVID-19", "Violence", "Terror", "Misleading claims"])
            + '</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- how it works
st.markdown('<div class="section" id="how"><div class="eyebrow">How it works</div>'
            '<div class="sec-h">From headline to verdict in three steps</div>'
            '<div class="sec-p">The model has learned how real and fake news tend to be written, using thousands of labelled examples.</div></div>',
            unsafe_allow_html=True)
ICON = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{}</svg>'
feats = [
    (ICON.format('<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><path d="M8 13h8M8 17h5"/>'),
     "1. You paste a statement", "Any news headline or short claim. The text is cleaned and split into words and word pairs."),
    (ICON.format('<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/>'),
     "2. The model scores it", "A Linear SVM compares the wording with 56,340 labelled statements and measures which side it falls on."),
    (ICON.format('<path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/>'),
     "3. You see the verdict", "You get a real or fake call, a confidence level, and the words that pushed the decision."),
]
st.markdown('<div class="grid3">' + "".join(
    f'<div class="feat"><div class="ico">{ic}</div><h4>{t}</h4><p>{d}</p></div>' for ic, t, d in feats) + '</div>',
    unsafe_allow_html=True)

# ---------------------------------------------------------------- tips
st.markdown('<div class="section" id="tips"><div class="eyebrow">Stay alert</div>'
            '<div class="sec-h">Four habits that help you spot fake news</div>'
            '<div class="sec-p">A tool can flag suspicious wording, but good habits catch what it cannot.</div></div>', unsafe_allow_html=True)
TIPS = [("01", "Check the source", "Is it a known outlet? Look for an About page, named authors and a working website."),
        ("02", "Watch the tone", "Shock, panic and phrases like \"share before it is deleted\" are common in fake posts."),
        ("03", "Search for the story", "If a big claim is real, other trusted outlets will be reporting it too."),
        ("04", "Verify photos and dates", "Old photos and videos are often reshared as if they were recent.")]
st.markdown('<div class="grid4">' + "".join(
    f'<div class="tip"><div class="num">{n}</div><h4>{t}</h4><p>{d}</p></div>' for n, t, d in TIPS) + '</div>',
    unsafe_allow_html=True)

# ---------------------------------------------------------------- faq
st.markdown('<div class="section" id="faq"><div class="eyebrow">FAQ</div><div class="sec-h">Common questions</div></div>', unsafe_allow_html=True)
FAQ = [("Does it check whether a claim is actually true?",
        "No. It recognises how real and fake news are usually written, based on 56,340 labelled statements. It cannot look up facts, so treat the result as a first screen."),
       ("How accurate is it?",
        "It was correct on about 97% of unseen statements from the same dataset. On news from other sources the accuracy will likely be lower."),
       ("Which languages work?", "English headlines and short claims work best. Full articles and other languages are not supported."),
       ("Is my text saved?", "No. Statements are only used to produce the result in your current session.")]
for q, a_ in FAQ:
    with st.expander(q):
        st.write(a_)

# ---------------------------------------------------------------- footer
st.markdown(
    '<div class="footer" style="justify-content:center;">'
    'TruthLens may make mistakes. Verify important claims with trusted sources.'
    '</div>',
    unsafe_allow_html=True,
)
