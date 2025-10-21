import streamlit as st
import spacy
import subprocess
import sys
from spacy import displacy

# =====================================
# ✅ Ensure model is installed at startup
# =====================================
def ensure_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy English model (en_core_web_sm)...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        return spacy.load("en_core_web_sm")

nlp = ensure_spacy_model()

# =====================================
# 🧠 App Content
# =====================================
st.title("🧠 NLP Product Review Analyzer")

reviews = [
    "I love my new Samsung Galaxy phone. The camera quality is amazing!",
    "This Apple MacBook Pro is overpriced and the battery life is disappointing.",
    "Sony headphones deliver incredible sound for the price.",
    "The Dell laptop runs smoothly and is perfect for work.",
    "I'm unhappy with this Lenovo tablet — it's very slow."
]

st.header("🔍 Named Entity Recognition (NER) Results")

for review in reviews:
    st.subheader(f"Review: {review}")
    doc = nlp(review)
    if doc.ents:
        for ent in doc.ents:
            st.write(f"- **{ent.text}** ({ent.label_})")
    else:
        st.write("No entities found.")
    st.write("---")

# =====================================
# 💬 Simple Sentiment Analysis
# =====================================
st.header("💬 Sentiment Analysis")

positive_words = ["love", "amazing", "incredible", "perfect", "smoothly"]
negative_words = ["disappointing", "slow", "overpriced", "unhappy", "bad"]

def analyze_sentiment(text):
    text_lower = text.lower()
    pos = sum(word in text_lower for word in positive_words)
    neg = sum(word in text_lower for word in negative_words)
    if pos > neg:
        return "Positive 😊"
    elif neg > pos:
        return "Negative 😞"
    else:
        return "Neutral 😐"

for review in reviews:
    sentiment = analyze_sentiment(review)
    st.write(f"**{review}** → {sentiment}")

# =====================================
# 🖼️ Visualize Entities (Optional)
# =====================================
st.header("🖼️ Entity Visualization (Sample)")
doc = nlp(reviews[0])
html = displacy.render(doc, style="ent")
st.markdown(html, unsafe_allow_html=True)
