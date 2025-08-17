# Brainwave_Matrix_Intern
Fake News Detection Using Machine Learning

This project implements a machine learning pipeline to classify news articles as real or fake using natural language processing (NLP) and multiple classification algorithms. It uses cleaned and labeled news datasets and evaluates models like Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

📌 Content:
    Overview 
    Dataset
    Technologies Used
    Project Structure
    Model Performance
    How it Works
    Usage
    


🔍 Overview:
Fake news poses a serious threat to public opinion and democracy. This project:
  Preprocesses real-world news articles
  Applies NLP techniques to clean and vectorize text
  Trains and evaluates multiple ML models
  Provides a manual testing function to test any news content

📁 Dataset:
    This project uses the Fake and Real News Dataset from Kaggle, which includes:
    Fake.csv – News labeled as fake
    True.csv – News labeled as real
    Total samples: ~44,000+
    Each dataset contains: title, text, subject, date

⚙️ Technologies Used:

Python,Pandas, NumPy,Matplotlib, Seaborn,Scikit-learn,Natural Language Processing (Regex, TF-IDF)

📂 Project Structure:
Fake_News_Detection/
├── data                   
├── fake_news_detection.py  
├── README.md
├── requirements.txt

📊 Model Performance:
Model	Accuracy (Approx)
Logistic Regression	✅ High
Decision Tree	✅ Moderate
Random Forest	✅ High
Gradient Boosting	✅ High
Each model is evaluated using:
Accuracy Score
Classification Report (Precision, Recall, F1-score)

🚀 How It Works:
1) Load & Label Data
    Fake.csv → label 1 (Fake)
    True.csv → label 0 (Real)
2) Preprocessing
    Remove HTML tags, URLs, punctuation, numbers
    Lowercase and normalize text
3) Text Vectorization
    TF-IDF Vectorizer converts text to numeric features
4) Model Training
    Logistic Regression
    Decision Tree Classifier
    Gradient Boosting Classifier
    Random Forest Classifier
5) Evaluation
    Accuracy, Precision, Recall, F1-score
6) Manual Testing Function
    You can input a news article and get predictions from all four models.

🧪 Usage:
  To manually test a piece of news:
      python fake_news_detection.py
  Then input a news article when prompted:
      Enter news text: <paste your article here>
  
  OUT PUT:
  LR Prediction: Fake News
  DT Prediction: Fake News
  GBC Prediction: Not A Fake News
  RFC Prediction: Fake News
