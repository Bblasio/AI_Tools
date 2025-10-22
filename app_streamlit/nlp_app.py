import streamlit as st
import spacy
from spacy import displacy
import os

# =====================================
# ✅ Safe model loading with fallback
# =====================================
def load_spacy_model():
    try:
        # Try to load the model normally
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("⚠️ spaCy model not found. Attempting to load a fallback...")
        try:
            # Try using the package directly if preinstalled
            import en_core_web_sm
            return en_core_web_sm.load()
        except ImportError:
            # Last resort: create a blank English model (tokenization only)
            st.warning("⚙️ Using a lightweight fallback NLP model (no NER).")
            nlp_blank = spacy.blank("en")
            return nlp_blank

nlp = load_spacy_model()

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

# =====================================
# 🔍 Named Entity Recognition
# =====================================
st.header("🔍 Named Entity Recognition (NER) Results")

for review in reviews:
    st.subheader(f"Review: {review}")
    doc = nlp(review)
    if "ner" in nlp.pipe_names and doc.ents:
        for ent in doc.ents:
            st.write(f"- **{ent.text}** ({ent.label_})")
    else:
        st.write("No named entities available (fallback model in use).")
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
# 🖼️ Entity Visualization (Optional)
# =====================================
if "ner" in nlp.pipe_names:
    st.header("🖼️ Entity Visualization (Sample)")
    doc = nlp(reviews[0])
    html = displacy.render(doc, style="ent")
    st.markdown(html, unsafe_allow_html=True)
else:
    st.info("Entity visualization unavailable (fallback model in use).")
