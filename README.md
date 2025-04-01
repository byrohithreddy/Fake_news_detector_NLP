# Fake News Detector (NLP)

## 🚀 Project Overview
This is a **Fake News Detection** system built using **Natural Language Processing (NLP)** and **Machine Learning** models. It classifies news articles as **FAKE or GENUINE** based on their text content using **Logistic Regression & Random Forest Classifier**.

## 📌 Features
- 📊 **Uses Machine Learning models** (Logistic Regression, Random Forest) for classification.
- 📝 **Preprocesses text** using tokenization, stopword removal, and TF-IDF Vectorization.
- 🖥️ **Provides an Interactive UI** using `Gradio`.
- ✅ **Simple & Fast Execution** for fake news detection.

## 📂 Dataset
- **`True.csv`** → Contains real news articles.
- **`Fake.csv`** → Contains fake news articles.

## 🛠️ Technologies Used
- **Python** 🐍
- **Pandas, NumPy** → Data Preprocessing
- **Scikit-Learn** → Machine Learning
- **Gradio** → Web-based UI
- **Regex** → Text Cleaning

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/byrohithreddy/Fake_news_detector_NLP.git
cd Fake_news_detector_NLP
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python main.py
```

This will launch the Gradio UI in your browser where you can enter news articles and get predictions.

---

## 🔬 How It Works
1. The model is trained on **fake & real news datasets**.
2. **Text preprocessing** is applied (lowercasing, removing punctuation, stopwords, etc.).
3. **TF-IDF Vectorization** converts text into numerical features.
4. **Machine Learning models** (Logistic Regression & Random Forest) classify the news.
5. The **Gradio interface** allows users to input news and check results.

---

## 🎯 Usage
1. **Run the script** using `python main.py`.
2. **Enter a news article** in the Gradio interface.
3. **Check the Prediction:**
   - ✅ **Genuine** → If both models predict as real news.
   - ❌ **Fake** → If both models predict as fake news.
   - ⚠️ **Uncertain** → If models have conflicting outputs.

---

## 📜 Example Output
```bash
LR Prediction: GENUINE, RFC Prediction: FAKE
Overall: UNCERTAIN
```

---

## 📌 Future Improvements
- 🌍 Deploy the model on **Streamlit / Flask**.
- 📊 Improve accuracy using **Deep Learning (LSTM, Transformers)**.
- 📡 API Integration for real-time news classification.

---

## 🤝 Contributing
Feel free to fork this repository, create a branch, and submit a pull request for improvements.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## ✨ Author
👨‍💻 **Rohith Reddy**  
GitHub: [byrohithreddy](https://github.com/byrohithreddy)  
LinkedIn: [MUSHKE ROHITH REDDY](www.linkedin.com/in/mushke-rohith-reddy-915945306)  

---

⭐ **If you found this useful, please give it a star!** ⭐

