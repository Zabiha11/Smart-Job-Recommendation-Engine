# 💼 Smart Job Recommendation Engine using Machine Learning

## 📌 Overview
The **Smart Job Recommendation Engine** is a content-based machine learning system that recommends relevant job roles to users based on their skills and experience.  
It uses **Natural Language Processing (NLP)** and **TF-IDF with cosine similarity** to match user profiles with job descriptions in real time.

This project simulates how platforms like **LinkedIn, Indeed, and Naukri** recommend jobs to candidates.

---

## 🎯 Problem Statement
Job seekers often apply to roles that do not align with their skill sets, resulting in low response rates and inefficient job searches.  
This project aims to solve that problem by automatically recommending the **most relevant job opportunities** based on textual similarity between user skills and job descriptions.

---

## 🚀 Features
- Content-based job recommendation
- NLP-based text preprocessing
- TF-IDF vectorization with bi-grams
- Cosine similarity ranking
- Real-time recommendations using Streamlit
- Clean and interpretable ML approach (no black-box models)

---

## 🧠 Machine Learning Approach

### 🔹 Text Preprocessing
- Lowercasing
- Removing punctuation and numbers
- Stopword removal using NLTK

### 🔹 Feature Extraction
- TF-IDF Vectorizer
- Uni-grams and bi-grams for better context capture

### 🔹 Similarity Measure
- Cosine Similarity to rank jobs based on relevance to user input

---

## 🏗️ System Architecture



User Skills / Profile
↓
Text Preprocessing (NLP)
↓
TF-IDF Vectorization
↓
Cosine Similarity Computation
↓
Top Job Recommendations


---

## 🗂️ Project Structure



Smart Job Recommendation Engine/
│
├── app.py # Streamlit web application
├── recommender.py # ML & recommendation logic
├── data/
│ └── jobs.csv # Job title & description dataset
├── requirements.txt
└── README.md


---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit
- Matplotlib, Seaborn (EDA)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

git clone <your-repo-link>
cd Smart-Job-Recommendation-Engine

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download NLTK Stopwords
import nltk
nltk.download('stopwords')

4️⃣ Run the Streamlit App
streamlit run app.py

🧪 Example Input
python django backend developer with REST API experience

🔍 Output

Recommended job titles

Job descriptions ranked by relevance

📊 Dataset

The dataset consists of:

Job Title

Job Description

It represents real-world job postings used to train and evaluate the recommendation engine.

📈 Future Enhancements

Skill gap analysis

Similarity score (%) display

Resume (PDF) upload and parsing

Job category filtering

Deployment on cloud platforms

📄 Resume Highlights

Built a Smart Job Recommendation Engine using NLP and TF-IDF

Implemented cosine similarity–based ranking

Developed a real-time Streamlit web application

Designed an end-to-end ML pipeline from preprocessing to deployment

👤 Author

Zabi
Machine Learning & Software Engineering Enthusiast

📜 License

This project is for educational and portfolio purposes.
