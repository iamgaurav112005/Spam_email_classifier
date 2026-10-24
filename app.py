import streamlit as st
import pandas as pd
import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI Spam Classifier", page_icon="📧", layout="wide")

# --- 1. TRAIN MODEL ON STARTUP ---
@st.cache_resource
def train_ai_model():
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
        df.columns = ['Category', 'Message']
    except FileNotFoundError:
        df = pd.DataFrame({
            'Category': ['ham', 'spam', 'ham', 'spam'],
            'Message': ['Hey, are we still meeting?', 'URGENT! You won a prize!', 'Please review the document.', 'Cheap loans instantly!']
        })
    df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    pipeline = Pipeline([('vectorizer', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])
    pipeline.fit(df['Message'], df['Spam'])
    return pipeline

model = train_ai_model()

# --- 2. NATURAL LANGUAGE EXPLANATION ENGINE ---
def generate_natural_explanation(text, prediction):
    """Generates a human-readable paragraph explaining the AI's decision."""
    text_lower = text.lower()
    reasons = []

    if prediction == 1:
        # Check for common spam/phishing heuristics
        if any(word in text_lower for word in ["dear", "customer", "user", "sir", "madam"]):
            reasons.append("a generic greeting")
        if any(word in text_lower for word in ["win", "prize", "cash", "lottery", "reward", "$"]):
            reasons.append("an unsolicited claim of a financial prize")
        if any(word in text_lower for word in ["urgent", "immediately", "act now", "expire", "limited time"]):
            reasons.append("the use of artificial urgency to pressure the recipient")
        if any(word in text_lower for word in ["password", "account", "verify", "ssn", "login", "bank"]):
            reasons.append("requests for personal or account information")
        if any(word in text_lower for word in ["click", "link", "http", "www", "here"]):
            reasons.append("suspicious interactive links")

        if not reasons:
            return "The message contains irregular vocabulary and statistical text patterns commonly associated with unsolicited promotional junk."

        # Build the flowing paragraph
        explanation = "The message exhibits multiple classic phishing indicators, including "
        if len(reasons) > 1:
            explanation += ", ".join(reasons[:-1]) + ", and " + reasons[-1] + "."
        else:
            explanation += reasons[0] + "."
            
        if "suspicious interactive links" in reasons and "requests for personal or account information" in reasons:
            explanation += " It explicitly requests data via these links, which is a hallmark of identity theft attempts."
            
        return explanation
    else:
        return "The message utilizes standard conversational vocabulary, lacks manipulative urgency or suspicious links, and matches the statistical patterns of legitimate communication."

# --- 3. CUSTOM UI COMPONENTS ---
def display_custom_alert(explanation, is_spam):
    """Renders a custom HTML block matching the requested UI design."""
    if is_spam:
        # Matches the red warning box design
        html = f"""
        <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 8px; padding: 16px; display: flex; font-family: sans-serif; margin-top: 20px;">
            <div style="margin-right: 16px; margin-top: 2px;">
                <div style="background-color: #FEE2E2; color: #DC2626; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 16px;">
                    ⚠️
                </div>
            </div>
            <div>
                <h4 style="color: #B91C1C; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">Warning: Spam Detected</h4>
                <p style="color: #4B5563; margin: 0; font-size: 15px; line-height: 1.6;">
                    {explanation}
                </p>
            </div>
        </div>
        """
    else:
        # Green safe box equivalent
        html = f"""
        <div style="background-color: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 16px; display: flex; font-family: sans-serif; margin-top: 20px;">
            <div style="margin-right: 16px; margin-top: 2px;">
                <div style="background-color: #DCFCE7; color: #16A34A; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 16px;">
                    ✅
                </div>
            </div>
            <div>
                <h4 style="color: #15803D; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">Safe Message</h4>
                <p style="color: #4B5563; margin: 0; font-size: 15px; line-height: 1.6;">
                    {explanation}
                </p>
            </div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


# --- 4. EMAIL PARSING LOGIC ---
def extract_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode('utf-8', errors='ignore')
    else:
        return msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    return ""

def scan_inbox(username, app_password, num_emails=10):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, app_password)
        mail.select("inbox")

        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()[-num_emails:] 
        
        scanned_data = []

        for e_id in reversed(email_ids):
            res, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    subject_header = decode_header(msg.get("Subject", "No Subject"))[0]
                    subject = subject_header[0]
                    encoding = subject_header[1]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8", errors='ignore')
                    
                    body = extract_email_body(msg)
                    if not body.strip():
                        body = "[No readable text found]"
                        
                    prediction = model.predict([body])[0]
                    label = "🚨 Spam" if prediction == 1 else "✅ Safe"
                    explanation = generate_natural_explanation(body, prediction)
                    
                    scanned_data.append({
                        "Classification": label,
                        "Subject": subject,
                        "AI Reasoning": explanation
                    })
        mail.logout()
        return scanned_data
    except Exception as e:
        return str(e)

# --- 5. DASHBOARD INTERFACE ---
st.title("📧 Comprehensive AI Spam Classifier")

with st.sidebar:
    st.header("IMAP Connection (For Live Scanner)")
    email_user = st.text_input("Gmail Address", placeholder="you@gmail.com")
    email_pass = st.text_input("App Password", type="password", placeholder="16-character password")
    scan_limit = st.slider("Emails to Scan", min_value=5, max_value=50, value=10)

tab1, tab2 = st.tabs(["✍️ Manual Text Entry", "📥 Live Inbox Scanner"])

# --- TAB 1: MANUAL ENTRY ---
with tab1:
    st.subheader("Paste an email or text message to analyze it")
    user_input = st.text_area(
        "Enter text here:", 
        height=150, 
        placeholder="Dear user, you have won a $1,000 cash prize! Click here immediately to claim..."
    )
    
    if st.button("Analyze Text"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # Predict and Generate Explanation
            prediction = model.predict([user_input])[0]
            explanation = generate_natural_explanation(user_input, prediction)
            
            # Display Custom UI Component
            display_custom_alert(explanation, is_spam=(prediction == 1))

# --- TAB 2: LIVE SCANNER ---
with tab2:
    st.subheader("Automatically read and classify your latest emails")
    if st.button("Scan My Inbox", type="primary"):
        if not email_user or not email_pass:
            st.error("Please enter your email and App Password in the sidebar.")
        else:
            with st.spinner("Connecting to Gmail and analyzing messages..."):
                results = scan_inbox(email_user, email_pass, scan_limit)
                
                if isinstance(results, str):
                    st.error(f"Failed to connect: {results}")
                elif len(results) == 0:
                    st.warning("No emails found.")
                else:
                    st.success(f"Successfully analyzed {len(results)} emails!")
                    df_results = pd.DataFrame(results)
                    
                    def color_rows(row):
                        if "Spam" in row['Classification']:
                            return ['background-color: #ffcccc; color: black'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(df_results.style.apply(color_rows, axis=1), use_container_width=True)
