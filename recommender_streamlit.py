import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df = pd.read_csv("Price.csv")
df.columns = df.columns.str.replace(" ", "_")  # Fix column names

df["Content"] = df[["Commodity", "Variety", "Grade"]].apply(lambda x: " ".join(x.dropna().astype(str)).lower(), axis=1)

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
commodity_matrix = vectorizer.fit_transform(df["Content"])

def get_market_recommendations(commodity, variety, grade, top_k=5):
    input_query = f"{commodity} {variety} {grade}".lower()
    query_vector = vectorizer.transform([input_query])
    sim_scores = linear_kernel(query_vector, commodity_matrix).flatten()
    top_indices = sim_scores.argsort()[-(top_k + 1):-1][::-1]
    return df.iloc[top_indices][["Market", "State", "District"]]

# Streamlit UI
st.title("Commodity Market Recommender")
st.write("Select the details below to get recommended markets.")

# Dropdown options
commodity_options = df["Commodity"].dropna().unique().tolist()
variety_options = df["Variety"].dropna().unique().tolist()
grade_options = df["Grade"].dropna().unique().tolist()

# User selection
commodity = st.selectbox("Select Commodity", commodity_options)
variety = st.selectbox("Select Variety", variety_options)
grade = st.selectbox("Select Grade", grade_options)

if st.button("Get Recommendations"):
    recommendations = get_market_recommendations(commodity, variety, grade, top_k=5)
    if not recommendations.empty:
        st.write("### Recommended Markets:")
        st.dataframe(recommendations)
    else:
        st.warning("No recommendations found. Try different inputs.")
