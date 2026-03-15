import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

.big-title{
font-size:40px;
font-weight:bold;
text-align:center;
color:#4CAF50;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
}

.result-spam{
background-color:#ff4b4b;
padding:20px;
border-radius:10px;
color:white;
font-size:22px;
text-align:center;
}

.result-ham{
background-color:#2ecc71;
padding:20px;
border-radius:10px;
color:white;
font-size:22px;
text-align:center;
}

.stButton>button{
width:100%;
border-radius:10px;
height:3em;
background-color:#4CAF50;
color:white;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("mail_data.csv")
    df = df.where(pd.notnull(df), "")

    df.loc[df["Category"] == "spam", "Category"] = 0
    df.loc[df["Category"] == "ham", "Category"] = 1

    X = df["Message"]
    Y = df["Category"].astype(int)

    return train_test_split(X, Y, test_size=0.2, random_state=3), df


# -------------------- TRAIN MODEL --------------------
@st.cache_resource
def train_model(X_train, Y_train):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1,2),
        max_features=3000
    )

    X_train_features = vectorizer.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_features, Y_train)

    model = LogisticRegression()
    model.fit(X_train_balanced, Y_train_balanced)

    return model, vectorizer


# -------------------- MAIN APP --------------------
def main():

    # Sidebar
    with st.sidebar:
        st.title("📧 Spam Mail Detector")
        st.write("Machine Learning based email spam classifier")

        st.markdown("---")

        st.write("### How it works")
        st.write("1️⃣ Enter message")
        st.write("2️⃣ Click Predict")
        st.write("3️⃣ Get spam prediction")

        st.markdown("---")
        st.info("Made by **Mohd Shami**")

    # Load data
    (X_train, X_test, Y_train, Y_test), raw_df = load_data()
    model, vectorizer = train_model(X_train, Y_train)

    # Title
    st.markdown('<p class="big-title">Spam Mail Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Detect whether an email is Spam or Ham using Machine Learning</p>', unsafe_allow_html=True)

    st.write("")

    # Input box
    user_input = st.text_area("✉ Enter Email / Message")

    # Predict button
    if st.button("🔍 Predict Message"):

        if user_input.strip()=="":
            st.warning("Please enter a message")

        else:
            input_data = vectorizer.transform([user_input])
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.markdown('<div class="result-ham">✅ This message is HAM (Not Spam)</div>', unsafe_allow_html=True)

            else:
                st.markdown('<div class="result-spam">🚨 This message is SPAM</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    # ---------------- PERFORMANCE ----------------
    with st.expander("📊 Model Performance"):

        X_test_features = vectorizer.transform(X_test)
        predictions = model.predict(X_test_features)

        acc = accuracy_score(Y_test, predictions)

        col1,col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{acc*100:.2f}%")

        with col2:
            st.write("Confusion Matrix")
            cm = confusion_matrix(Y_test, predictions)
            st.dataframe(pd.DataFrame(cm))

        st.write("Classification Report")
        report = classification_report(Y_test, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # ---------------- DATASET CHART ----------------
    with st.expander("📊 Dataset Distribution"):

        counts = raw_df["Category"].value_counts()

        st.bar_chart(counts)


if __name__ == "__main__":
    main()
