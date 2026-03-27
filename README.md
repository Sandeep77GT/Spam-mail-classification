# Spam Detection Web Application using Machine Learning

## Overview
This project presents a web-based spam detection system built using Machine Learning and deployed with Streamlit. The application allows users to upload a dataset, train a model, and classify email or message text as spam or ham in real time.

The system uses TF-IDF vectorization and a Multinomial Naive Bayes classifier to perform text classification.

---

## Problem Statement
Spam messages are a common issue in communication systems. This project aims to build an automated system that can accurately classify messages into:

- Spam
- Ham (Not Spam)

---

## Features
- Upload dataset in ZIP format
- Automatic dataset preprocessing
- TF-IDF based feature extraction
- Model training using Multinomial Naive Bayes
- Model evaluation (accuracy and classification report)
- Real-time text classification through a web interface

---

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

## Dataset Format
The dataset should contain at least two columns:

- `text`: The message content
- `label`: The class label (spam or ham)

Example:

| text | label |
|------|------|
| Congratulations! You won a prize | spam |
| Let's meet tomorrow | ham |

---

## Methodology

### 1. Data Loading
- Dataset is uploaded as a ZIP file
- First CSV file inside the ZIP is extracted and loaded

### 2. Data Preprocessing
- Labels are cleaned and standardized
- Only valid labels (spam, ham) are retained
- Missing values are removed

### 3. Feature Extraction
- TF-IDF vectorization is applied
- Maximum features limited to 5000
- English stopwords removed

### 4. Model Training
- Train-test split (70% training, 30% testing)
- Multinomial Naive Bayes classifier is used

### 5. Evaluation
- Accuracy score
- Classification report (precision, recall, F1-score)

### 6. Prediction
- User input text is transformed using TF-IDF
- Model predicts whether the message is spam or ham

---
├── app.py
├── requirements.txt
├── README.md


---

## Installation

### 1. Clone the Repository

git clone https://github.com/your-username/spam-detector-app.git

cd spam-detector-app


### 2. Install Dependencies

pip install -r requirements.txt

---

## Running the Application

streamlit run app.py

---

## Example Workflow
1. Upload ZIP file containing dataset  
2. Model is trained automatically  
3. View accuracy and classification report  
4. Enter custom email text  
5. Get prediction (Spam or Ham)  

---

## Model Details

- Algorithm: Multinomial Naive Bayes
- Feature Engineering: TF-IDF
- Train/Test Split: 70/30
- Max Features: 5000

---

## Results
The model typically achieves good accuracy depending on dataset quality. Performance is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-score

---

## Limitations
- Performance depends on dataset quality
- Limited to binary classification (spam/ham)
- Does not handle multilingual text
- No hyperparameter tuning implemented

---

## Future Improvements
- Add support for multiple datasets
- Implement advanced models (Logistic Regression, SVM, Transformers)
- Deploy on cloud platforms (AWS, GCP, Azure)
- Add model persistence (save/load trained model)
- Improve UI with visualizations

---

## Author
Sandeep S L  
MSc Data Analytics (Computational Science)  
Digital University of Kerala  

---

## License
This project is open-source and available under the MIT License.



## Project Structure
