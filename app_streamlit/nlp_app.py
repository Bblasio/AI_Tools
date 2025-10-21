# ============================================
# 🧠 TASK 3: NLP with spaCy
# ============================================

import streamlit as st
import spacy
from spacy import displacy

# Load preinstalled model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()


# --------------------------------------------
# Ensure spaCy model is available
# --------------------------------------------
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        with st.spinner("Downloading spaCy English model... ⏳"):
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_model()

# --------------------------------------------
# App Title
# --------------------------------------------
st.title("💬 NLP App – Amazon Product Review Analyzer")

# Step 1: Sample Reviews
default_reviews = [
    "I love my new Samsung Galaxy phone. The camera quality is amazing!",
    "This Apple MacBook Pro is overpriced and the battery life is disappointing.",
    "Sony headphones deliver incredible sound for the price.",
    "The Dell laptop runs smoothly and is perfect for work.",
    "I'm unhappy with this Lenovo tablet — it's very slow."
]

st.write("### Example Reviews:")
for r in default_reviews:
    st.markdown(f"- {r}")

# Allow user input
st.write("---")
user_review = st.text_area("✍️ Enter your own review to analyze:", "")
reviews = default_reviews + ([user_review] if user_review else [])

# Step 2: Rule-Based Sentiment Setup
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

# Step 3: Run Analysis
if st.button("🔍 Analyze Reviews"):
    for review in reviews:
        st.markdown(f"**Review:** {review}")
        doc = nlp(review)

        # Named Entities
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        if ents:
            st.write("**Named Entities:**")
            for text, label in ents:
                st.write(f"• {text} ({label})")
        else:
            st.write("_No named entities found._")

        # Sentiment
        sentiment = analyze_sentiment(review)
        st.write(f"**Sentiment:** {sentiment}")
        st.write("---")

# Step 4: Visualization Example
if st.checkbox("🧠 Show Entity Visualization (First Review)"):
    doc = nlp(reviews[0])
    html = displacy.render(doc, style="ent")
    st.components.v1.html(html, height=200, scrolling=True)
