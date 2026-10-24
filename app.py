import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Spam Classifier", page_icon="🛡️", layout="centered")

# --- MODEL TRAINING (Cached for performance) ---
@st.cache_resource
def train_ai_model():
    # Attempt to load a real dataset (spam.csv)
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['Category', 'Message']
        using_real_data = True
    except FileNotFoundError:
        # Fallback dataset if spam.csv is missing
        data = {
            'Category': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
            'Message': [
                'Hey, are we still meeting for the project tomorrow?', 
                'URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010', 
                'I will be late to the office today. Traffic is bad.', 
                'Get cheap loans instantly! Click here to apply now with 0% interest rate.',
                'Can you review the attached document before the meeting?',
                'Your account has been compromised. Please update your billing details immediately via this link.',
                'Happy Birthday! Hope you have a fantastic day!',
                'Buy 1 Get 1 Free on all electronics! Limited time offer. Visit www.scam-electronics.com'
            ]
        }
        df = pd.DataFrame(data)
        using_real_data = False

    # Convert labels to binary (1 = Spam, 0 = Ham)
    df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

    # Split data to calculate accuracy (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Spam'], test_size=0.2, random_state=42)

    # Create the Machine Learning Pipeline
    # 1. CountVectorizer: Converts text to word counts
    # 2. TfidfTransformer: Penalizes overly common words (like 'the', 'is')
    # 3. MultinomialNB: The Naive Bayes algorithm for classification
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return pipeline, accuracy, using_real_data, len(df)

# Load model and stats
model, model_accuracy, real_data_loaded, dataset_size = train_ai_model()

# --- SIDEBAR: PROJECT DETAILS ---
with st.sidebar:
    st.header("Project Details")
    st.write("**Algorithm:** Naive Bayes")
    st.write("**NLP Techniques:** Bag of Words & TF-IDF")
    
    st.divider()
    
    st.subheader("Model Statistics")
    if real_data_loaded:
        st.success("✅ Real Dataset Loaded")
    else:
        st.warning("⚠️ Using Mock Dataset (spam.csv missing)")
        
    st.metric("Total Emails Analyzed", dataset_size)
    st.metric("Model Accuracy", f"{model_accuracy * 100:.2f}%")

# --- MAIN INTERFACE ---
st.title("🛡️ AI Spam Email Classifier")
st.write("This application uses Natural Language Processing (NLP) to analyze text and predict whether a message is safe or spam.")

st.divider()

st.subheader("Test the AI")
user_input = st.text_area(
    "Paste an email, SMS, or text message here:", 
    height=150, 
    placeholder="e.g., Congratulations! You have been selected to win a free iPhone..."
)

if st.button("Classify Message", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.error("Please enter some text to classify.")
    else:
        # Make Prediction
        prediction = model.predict([user_input])[0]
        
        # Get prediction probabilities (confidence score)
        probabilities = model.predict_proba([user_input])[0]
        confidence = probabilities[prediction] * 100
        
        st.divider()
        st.subheader("AI Analysis Result:")
        
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED** (Confidence: {confidence:.1f}%)")
            st.write("This message exhibits strong linguistic patterns commonly associated with unsolicited spam, phishing attempts, or promotional junk.")
        else:
            st.success(f"✅ **SAFE MESSAGE** (Confidence: {confidence:.1f}%)")
            st.write("This message appears legitimate and does not match the typical patterns of spam or phishing.")