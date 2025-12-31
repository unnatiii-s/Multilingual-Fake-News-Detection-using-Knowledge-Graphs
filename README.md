# Multilingual-Fake-News-Detection-using-Knowledge-Graphs

This project is a Streamlit-based web application for detecting fake news across multiple languages.  
It uses transformer-based contextual embeddings combined with lightweight knowledge graph features to perform supervised fake news classification.

---

## 🔍 Key Features
- Multilingual fake news detection
- Subword tokenization using transformer tokenizer
- Contextual embeddings (DistilBERT / XLM-R)
- Supervised learning using FakeNewsNet dataset
- Knowledge Graph construction using Named Entity Recognition
- Confidence-based prediction with UNCERTAIN handling
- Interactive UI built with Streamlit

---

## 🧠 Model & Techniques
- **Tokenizer**: Multilingual subword tokenizer
- **Embeddings**: Contextual embeddings from transformer models
- **Classifier**: Neural network with KG feature fusion
- **Dataset**: FakeNewsNet (BuzzFeed & PolitiFact)
- **KG Features**: Number of nodes, edges, and graph density

---

🚀 How to Run the App

1️⃣ Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

2️⃣ Train the model
python train.py

3️⃣ Run the Streamlit app
streamlit run app.py
