import streamlit as st
import pandas as pd
from recommender import build_recommender, recommend_jobs

st.set_page_config(page_title="Smart Job Recommendation Engine")

st.title("💼 Smart Job Recommendation Engine")
st.write("Enter your skills or profile to get relevant job recommendations.")

# Load data
df = pd.read_csv("data/job_title_des.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Build model
tfidf, tfidf_matrix = build_recommender(df)

# User input
user_input = st.text_area(
    "Enter your skills / experience",
    placeholder="e.g. python django backend developer with REST API experience"
)

if st.button("Recommend Jobs"):
    if user_input.strip() == "":
        st.warning("Please enter your skills.")
    else:
        results = recommend_jobs(user_input, df, tfidf, tfidf_matrix)

        st.subheader("🔍 Recommended Jobs")
        for i, row in results.iterrows():
            st.markdown(f"### {row['Job Title']}")
            st.write(row['Job Description'][:500] + "...")
            st.markdown("---")
