# ============================================
# 🧠 Streamlit App - NLP with spaCy
# ============================================

import streamlit as st
import spacy
from spacy import displacy

# --------------------------------------------
# Load English NLP Model
# --------------------------------------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# --------------------------------------------
# App Title
# --------------------------------------------
st.title("🧠 NLP Review Analyzer (spaCy)")
st.write("Analyze **Amazon-style product reviews** for named entities and sentiment polarity.")

st.sidebar.header("⚙️ About the App")
st.sidebar.info(
    """
    - **Library:** spaCy  
    - **Features:**  
      • Named Entity Recognition (NER)  
      • Simple Sentiment Analysis  
    - **Dataset:** Sample product reviews
    """
)

# --------------------------------------------
# Sample Reviews (Default Data)
# --------------------------------------------
default_reviews = [
    "I love my new Samsung Galaxy phone. The camera quality is amazing!",
    "This Apple MacBook Pro is overpriced and the battery life is disappointing.",
    "Sony headphones deliver incredible sound for the price.",
    "The Dell laptop runs smoothly and is perfect for work.",
    "I'm unhappy with this Lenovo tablet — it's very slow."
]

# --------------------------------------------
# Input Section
# --------------------------------------------
st.subheader("💬 Enter or Select a Review")

user_input = st.text_area(
    "Type or paste a product review below:",
    value=default_reviews[0],
    height=100
)

# Optional: Quick selection from sample reviews
if st.checkbox("Use sample reviews"):
    selected = st.selectbox("Choose a sample review:", default_reviews)
    user_input = selected

# --------------------------------------------
# Sentiment Analysis (Rule-based)
# --------------------------------------------
positive_words = ["love", "amazing", "incredible", "perfect", "smoothly", "great", "excellent"]
negative_words = ["disappointing", "slow", "overpriced", "unhappy", "bad", "poor"]

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

# --------------------------------------------
# Run NLP Processing
# --------------------------------------------
if st.button("🔍 Analyze"):
    if user_input.strip():
        doc = nlp(user_input)
        st.subheader("📘 Named Entities Found:")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"- **{ent.text}** ({ent.label_})")
        else:
            st.write("No named entities detected.")

        sentiment = analyze_sentiment(user_input)
        st.subheader("🧭 Sentiment Analysis Result:")
        st.success(f"The review sentiment is: **{sentiment}**")

        # Visualize Entities with displaCy
        st.subheader("🧩 Entity Visualization:")
        html = displacy.render(doc, style="ent")
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("Please enter a review first.")
else:
    st.info("Click **Analyze** to process the review.")

# --------------------------------------------
# Footer
# --------------------------------------------
st.caption("NLP Task 3 | spaCy Named Entity & Sentiment Analysis | © Your Team")
