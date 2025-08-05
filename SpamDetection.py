# 🎮️ Enhanced Spam Detection Streamlit App

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from joblib import load
from wordcloud import WordCloud
import re
import string
import numpy as np
from io import BytesIO

# 🍿️ Centered Heading Above Image
st.markdown(
    """
    <div style='text-align: center; padding-top: 20px;'>
        <h1 style='font-size: 52px; color: #003566;'>
            📨 SPAM DETECTION APP 📄
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

#============================================================================

# -----------------------------
# 🌙 Theme Toggle
st.set_page_config(page_title="SPAM DETECTION APP", layout="wide" )
# 🌄 Full Background Image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.istockphoto.com/id/1205872379/vector/abstract-background-of-smooth-curves.jpg?s=612x612&w=0&k=20&c=iSVtjNn-UVardlSWofafiwyoci93HEzvB647mz119EI=");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
<div style="text-align: center;">
 <img src="https://img.freepik.com/free-photo/cybersecurity-concept-collage-design_23-2151877155.jpg?semt=ais_hybrid&w=740&q=80" style="width: 700px;" />
</div>
 """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: bold;
        color: #222 !important;
        border: 2px solid #ccc;
        border-radius: 12px;
        padding: 10px 20px;
        margin-right: 5px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f0f0 !important;
    }
    body[data-theme="dark"] .block-container {
        background: rgba(0, 0, 0, 0.65);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

if "theme" not in st.session_state:
    st.session_state.theme = "light"

with st.sidebar:
    st.image("https://media.contra.com/image/upload/w_800,q_auto/hrrymjceqwr3noubnslk", width= 200)
    st.title("Spam Classifier📄")
    theme_toggle = st.radio("Select Theme", ["light", "dark"])
    if theme_toggle != st.session_state.theme:
        st.session_state.theme = theme_toggle
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About This App"):
    st.markdown("""
    ### 📨 Spam Detection App
    - Built using **TF-IDF** + Machine Learning
    - Models included: Logistic Regression, Random Forest, XGBoost
    - Supports **batch uploads** and **custom model testing**

    ### 🤔 What is Spam?
    Unwanted or unsolicited messages — often promotional or fraudulent.

    ### 📚 Dataset Source
    - Public dataset from UCI: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

    ### 🧠 How It Works
    - **TF-IDF Vectorizer**: Converts text into numerical form
    - **ML Model**: Classifies messages as Ham or Spam

    ### 📌 Author
    - Kaushal Chaurasia
    """)

st.sidebar.markdown("---")   
with st.sidebar:
    show_charts = st.checkbox("📊 Show Visualizations", value=True)
    show_wordcloud = st.checkbox("☁️ WordClouds", value=True)
    show_top_words = st.checkbox("🔝 Top Spam Words", value=True)
    show_tfidf = st.checkbox("🧠 Show Top TF-IDF Features", value=True)
    st.markdown("---")
    st.markdown("Developed by **Kaushal Chaurasia**")
    st.image("https://png.pngtree.com/png-vector/20220704/ourlarge/pngtree-vector-no-spam-icon-advertising-safety-security-vector-png-image_14056372.jpg", width=200)
        
    
if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .stButton>button {
        color: white;
        background-color: #0096c7;
        border-radius: 8px;
    }
    .stTextArea>div>textarea {
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# 🔤 Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

# Load Data
raw_df = pd.read_csv("spam.csv", encoding='latin1')[['v1', 'v2']]
raw_df.columns = ['label', 'message']
raw_df['label'] = raw_df['label'].map({'ham': 0, 'spam': 1})
raw_df.dropna(subset=['label'], inplace=True)
raw_df.drop_duplicates(inplace=True)
raw_df['message'] = raw_df['message'].apply(clean_text)

X = raw_df['message']
y = raw_df['label']

# Load Vectorizer
vectorizer = load("tfidf_vectorizer.joblib")
X_vectorized = vectorizer.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 🔢 Model selection
model_option = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

# Load models
log_model = load("spam_model.joblib")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Model picker
if model_option == "Logistic Regression":
    model = log_model
elif model_option == "Random Forest":
    model = rf_model
else:
    model = xgb_model
# Evaluate the selected model
st.markdown("### 📊 Model Evaluation")

# Predict on the test set
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**✅ Accuracy of {model_option}:** {accuracy:.2%}")

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

col1= st.columns(2)[0]
if col1.button("Show Confusion Matrix"):
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))  # 👈 smaller figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax, annot_kws={"size": 12})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("📊 Confusion Matrix")
    st.pyplot(fig)
# 📊 Model Comparison Section
with st.expander("📈 Compare All Models"):
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve, auc
    import time

    st.markdown("### 📌 Accuracy & Evaluation of Models")
    models = {
        "Logistic Regression": log_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    results = []

    for name, m in models.items():
        start = time.time()
        preds = m.predict(X_test)
        end = time.time()
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": report['1']['precision'],
            "Recall": report['1']['recall'],
            "F1-Score": report['1']['f1-score'],
            "Time (s)": end - start
        })

    res_df = pd.DataFrame(results)
    st.dataframe(res_df.style.highlight_max(axis=0, color='lightgreen'))

    # Plot Accuracy Comparison
    st.markdown("### 📊 Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=res_df, x='Model', y='Accuracy', palette="viridis", ax=ax)
    ax.set_ylim(0.8, 1.0)
    st.pyplot(fig)

    # ROC Curve for each model
    st.markdown("### 📈 ROC-AUC Curves")
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in models.items():
        probs = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC Curve")
    ax.legend()
    st.pyplot(fig)
import time
start = time.time()
rf_model.fit(X_train, y_train)
end = time.time()
st.write(f"Random Forest Training Time: {end - start:.2f} seconds")


# Classification report
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True)
st.write("**📋 Classification Report:**")
st.dataframe(pd.DataFrame(report).transpose())


# The rest of your prediction and app logic goes here...
st.markdown("### 📥 Upload Your Custom Model (.joblib)")
custom_model_file = st.file_uploader("Upload a trained model file", type=["joblib"], key="custom_model")

if custom_model_file is not None:
    try:
        from joblib import load
        import tempfile

        # Save and load temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(custom_model_file.read())
            tmp_path = tmp.name

        custom_model = load(tmp_path)
        model = custom_model  # overwrite current selected model
        st.success("✅ Custom model loaded successfully!")

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")


# ======================== Main App ===========================
tabs = st.tabs(["📩 Single Prediction", "📂 Batch Upload", "📊 Visuals"])

# Single Message Prediction
with tabs[0]:
    st.subheader("🔍 Check If A Message Is Spam")
    user_input = st.text_area("Enter a Message:")
    if st.button("Predict"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            confidence = model.predict_proba(vector)[0][prediction]
            label = "📬 Ham" if prediction == 0 else "❌ Spam"
            st.success(f"Prediction: {label} (Confidence: {confidence:.2%})")
            st.progress(int(confidence * 100))
        else:
            st.warning("Please enter a message.")
    st.markdown("---")       

    if show_wordcloud:
        st.subheader("☁️ WordClouds")
        spam_words = ' '.join(raw_df[raw_df['label'] == 1]['message'])
        ham_words = ' '.join(raw_df[raw_df['label'] == 0]['message'])
        col1, col2 = st.columns(2)
        with col1:
            wc1 = WordCloud(width=500, height=300).generate(spam_words)
            st.image(wc1.to_array())
        with col2:
            wc2 = WordCloud(width=500, height=300).generate(ham_words)
            st.image(wc2.to_array())

# Batch Prediction
with tabs[1]:
    st.subheader("📂 Batch Prediction")
    uploaded_file = st.file_uploader("📂 Upload a CSV file")
    if uploaded_file:
        try:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            st.write("📌 Columns found in file:", df.columns.tolist())
            selected_column = st.selectbox("📂 Select the column that contains messages", df.columns)
            df['cleaned'] = df[selected_column].astype(str).apply(clean_text)
            X_batch = vectorizer.transform(df['cleaned'])
            df['prediction'] = model.predict(X_batch)
            df['prediction'] = df['prediction'].map({0: 'Ham', 1: 'Spam'})
            st.success("✅ Prediction complete!")
            st.write(df[[selected_column, 'prediction']])
            st.download_button("📅 Download Results", df.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

with tabs[2]:
    if show_charts:
        st.subheader("📊 Label Distribution")
        label_counts = raw_df['label'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie(label_counts, labels=['Ham', 'Spam'], autopct='%1.1f%%', colors=['green', 'red'], explode=(0.05, 0.05), shadow=True)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='label', data=raw_df, ax=ax2, palette=['green', 'red'])
        ax2.set_xticklabels(['Ham', 'Spam'])
        st.pyplot(fig2)

    if show_top_words:
        st.subheader("🔝 Top Frequent Spam Words")
        spam_messages = raw_df[raw_df['label'] == 1]['message']
        words = ' '.join(spam_messages).split()
        freq_dist = pd.Series(words).value_counts()[:10]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=freq_dist.values, y=freq_dist.index, palette="Reds_r", ax=ax)
        ax.set_title("Top 10 Spam Words")
        st.pyplot(fig)

    if show_tfidf:
        st.subheader("🧠 Top Features Indicative of Spam")
        feature_names = vectorizer.get_feature_names_out()
        top_n = 15
        if model_option == "Logistic Regression":
            coef = model.coef_[0]
            top_indices = np.argsort(coef)[-top_n:]
            top_features = feature_names[top_indices]
            top_values = coef[top_indices]
        else:
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:]
            top_features = feature_names[top_indices]
            top_values = importances[top_indices]

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_values, y=top_features, palette="coolwarm", ax=ax)
        ax.set_title(f"Top {top_n} Features - {model_option}")
        st.pyplot(fig)
st.markdown("---")
st.header("How to prevent us from Spam Messages")        
col1, col2=st.columns(2)
with col1 :
    st.header("✅ For Personal Use / Devices")
    st.image("https://framerusercontent.com/images/eu8wclLvMovvgLVZ5Ihpv4hM.jpg",width=500)
    vote1=st.button("Selected for Personal Use/Devices")
    with st.expander("To prevent spam messages, follow these steps:"):
      st.write("""
               1. Don’t share your number/email publicly
Avoid putting your contact info on public forums, social media, or untrusted websites.

2. Enable spam filters
Use Gmail/Outlook/Yahoo's built-in spam filters.

Enable SMS spam filtering in phones:

Android: Settings → Messages → Spam Protection

iPhone: Settings → Messages → Filter Unknown Senders

3. Use a secondary email/phone number
Create a separate email for sign-ups or promotions.

Use services like Google Voice, Burner, or TempMail for temporary contact info.

4. Don’t click suspicious links
Even if the message seems legitimate, avoid clicking links or downloading attachments unless you're sure.

5. Report & block spam
Always mark emails/SMS as spam so systems can learn.

Block numbers directly in your messaging app.

""")

with col2:
    st.header("🧠 For Developers/Businesses")
    st.image("https://xperteria.com/wp-content/uploads/2021/10/cover-1.png",width=480)
    vote2=st.button("Selected for Developers/Businesses") 
    with st.expander("To prevent spam messages, follow these steps:"):
      st.write("""1. Use CAPTCHA / reCAPTCHA
Prevent bots from submitting spam via forms.

2. Validate and sanitize user input
Use proper validation (email formats, phone formats).

Use server-side sanitization to prevent injection/spammy content.

3. Use spam detection ML models
Like the one you've built using TF-IDF + Logistic Regression.

Or use external APIs like:

Akismet (for comments)

Google Perspective API (for toxicity)

SpamAssassin (for email)

4. Rate limiting
Throttle how many times a user can submit a form/message to prevent flooding.

5. Email Verification
Send verification links to ensure the email is real.""")

if vote1:
    st.success("You voted for Personal Use!")
elif vote2:
    st.success("You voted for Developers/Businesses!")

st.markdown("---")
st.header(" Tools You Can Use")
st.markdown('>Blockquote: To make a perfect spam filter, follow these steps:\n>1. 🔐 Spam filters: SpamTitan, Barracuda, Mailwasher\n>2. 📱 Apps for SMS spam: Truecaller, Hiya, RoboKiller.\n>3. 📧 Disposable email: TempMail, Mailinator \n> 4. 📬 Email verification: Never use your primary email for sign-ups.\n>5. 🛡️ Use a VPN: Protect your IP and location.')
st.markdown("""----""")
import streamlit as st
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ✅ Function to send email via SendGrid
def send_email(name, email, feedback, feedback_type, rating):
    try:
        message = Mail(
             from_email=st.secrets["sendgrid"]["from_email"],
            to_emails=st.secrets["sendgrid"]["to_email"],
            subject=f"📬 Feedback from {name} - {feedback_type}",
            html_content=f"""
                <strong>Name:</strong> {name}<br>
                <strong>Email:</strong> {email or 'Not Provided'}<br>
                <strong>Rating:</strong> {rating}/5<br>
                <strong>Feedback Type:</strong> {feedback_type}<br><br>
                <strong>Message:</strong><br>{feedback}
            """
        )
        sg = SendGridAPIClient(st.secrets["sendgrid"]["api_key"])
        sg.send(message)
        return True
    except Exception as e:
        st.error(f"Email sending error: {e}")
        return False

# ✅ Feedback Form UI
st.subheader("📬 Feedback")
with st.form("feedback_form"):
    name = st.text_input("Your Name")
    rating = st.slider("Rate your experience (1 = Bad, 5 = Excellent)", 1, 5, 3)
    feedback_type = st.selectbox("Type of Feedback", ["Bug Report", "UI Suggestion", "Model Accuracy", "Other"])
    screenshot = st.file_uploader("Upload Screenshot (optional)", type=["png", "jpg", "jpeg"])
    if screenshot:
        st.image(screenshot, caption="Uploaded Screenshot", use_container_width=True)

    feedback = st.text_area("Share your feedback")
    email = st.text_input("Your Email (optional)")

    submitted = st.form_submit_button("Submit")

    if submitted:
        success = send_email(name, email, feedback, feedback_type, rating)
        if success:
            st.success(f"✅ Thanks {name}, we appreciate your feedback! Email sent successfully.")
        else:
            st.warning("⚠️ Feedback saved, but email notification failed. Check SendGrid secrets.")

