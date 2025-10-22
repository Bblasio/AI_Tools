import streamlit as st
import spacy
from spacy import displacy
import spacy.cli

# =============================
# ✅ Ensure spaCy model is installed
# =============================
@st.cache_resource
def load_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        # If model not found, download it using spaCy’s built-in CLI
        st.warning(f"Downloading spaCy model '{model_name}' ... Please wait ⏳")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

nlp = load_model()

# =============================
# 🧠 NLP App — Amazon-style Product Reviews
# =============================
st.title("🧠 NLP Product Review Analyzer")

reviews = [
    "I love my new Samsung Galaxy phone. The camera quality is amazing!",
    "This Apple MacBook Pro is overpriced and the battery life is disappointing.",
    "Sony headphones deliver incredible sound for the price.",
    "The Dell laptop runs smoothly and is perfect for work.",
    "I'm unhappy with this Lenovo tablet — it's very slow."
]

# =============================
# 🔍 Named Entity Recognition (NER)
# =============================
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

# =============================
# 💬 Sentiment Analysis
# =============================
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

# =============================
# 🖼️ Entity Visualization (Optional)
# =============================
st.header("🖼️ Entity Visualization (Sample)")
doc = nlp(reviews[0])
html = displacy.render(doc, style="ent")
st.markdown(html, unsafe_allow_html=True)
