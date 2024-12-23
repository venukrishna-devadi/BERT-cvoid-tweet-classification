# BERT COVID Tweet Classification  
A deep learning project to classify tweets as related or unrelated to COVID-19, leveraging pre-trained BERT models for robust and accurate text classification.  

---

## 📜 Project Overview  
This repository contains the code and methodology for a binary classification task aimed at determining whether a tweet is related to COVID-19. By fine-tuning BERT, the project demonstrates the effectiveness of state-of-the-art NLP techniques in analyzing large-scale text data.  

---

## ✨ Key Features  
- **Model**: Fine-tuned pre-trained BERT to classify tweets with high accuracy and reliability.  
- **Performance**: Achieved:  
  - **Accuracy**: 94%  
  - **F1-Score**: 0.946  
  - **Precision**: 0.941  
  - **Recall**: 0.950  
- **Optimization**: Hyperparameter tuning for learning rate, batch size, and number of epochs to enhance classification performance.  
- **Robust Evaluation**: Ensured stability and consistency across multiple epochs using standard metrics like accuracy, precision, recall, and F1-score.  

---

## 🛠️ Methodology  
1. **Data Preparation**:  
   - Used a labeled dataset of tweets for training and testing.  
2. **Model Training**:  
   - Fine-tuned the pre-trained BERT model using Hugging Face's Transformers library.  
   - Optimized hyperparameters to achieve optimal performance.  
3. **Evaluation**:  
   - Evaluated the model using metrics such as accuracy, precision, recall, and F1-score to validate its effectiveness.  

---

## 📊 Results  
- **Accuracy**: 94%  
- **F1-Score**: 0.946  
- **Precision**: 0.941  
- **Recall**: 0.950  

---

## 📂 Repository Structure  
- `data/`: Contains the labeled dataset used for training and testing.  
- `notebooks/`: Jupyter notebooks for training, evaluation, and analysis.  
- `scripts/`: Python scripts for preprocessing, training, and testing the model.  
- `results/`: Contains performance metrics and visualizations of the model's outcomes.  

---

## 🔑 Key Insights  
- Pre-trained BERT models are highly effective for binary text classification tasks, particularly in domains requiring nuanced understanding like public health communication.  
- Hyperparameter tuning significantly improves model performance, ensuring consistent results across diverse data splits.  

---

## 🔗 How to Use  
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/venukrishna-devadi/BERT-cvoid-tweet-classification.git
   cd BERT-cvoid-tweet-classification

2. **Install Requirements**:
   pip install -r requirements.txt

3. **Run Training**:
  Open the relevant notebook or script and execute the training pipeline.

4. **Evaluate Performance**:
   Use the evaluation scripts to analyze accuracy, precision, recall, and F1-score.

## 🚀 Future Work
Extend the classification task to multi-label classification for categorizing tweets into themes like prevention, outbreak, and symptoms.
Incorporate additional pre-trained models like RoBERTa or DistilBERT for comparative analysis.

## 📖 About the Author  
I’m **Venu Gopal Krishna Devadi**, a graduate student specializing in **Data Science** at Saint Peter’s University. My interests lie in **Natural Language Processing**, **Transformer Models**, and applying **Deep Learning** to solve real-world challenges.  

### Other Relevant Projects  
- **Multilingual Language Translator**: Transformer-based model translating English to Hindi and Telugu.  
- **Anime Character Image Classification**: Using EfficientNet B2 for advanced image recognition.  
- **Machine Learning Doubt Clarifier Chatbot**: Leveraging GPT and T5 models for ML query resolution.  

### Publications  
- **"Infodemic Management using Natural Language Processing: A COVID-19 Case Study"**  
  Presented at AMIA 2024 Annual Symposium – P157, San Francisco, CA.  

- **"Information Management using Natural Language Processing: A COVID-19 Case Study"**  
  Accepted in *Advances in Healthcare using Machine Learning*, Taylor and Francis.  

---

## 🔗 Connect with Me  
- **Email**: venukrishnadevadi@gmail.com  
- **LinkedIn**: [linkedin.com/in/venu-devadi-2350b3252](https://linkedin.com/in/venu-devadi-2350b3252/)  

---

## 🛠️ Getting Started  

### Prerequisites  
- Python 3.7+  
- Libraries: `pandas`, `numpy`, `tensorflow`, `torch`, `sklearn`, `tweepy`, `matplotlib`  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/COVID-Twitter-MultiLabel-Classification.git
   cd COVID-Twitter-MultiLabel-Classification

2. Install required libraries:  
   pip install -r requirements.txt
   
## 📂 Repository Files
Notebooks: Code for binary and multi-label classification.
Data: Scripts to preprocess and load data.
Results: Visualization of model performance metrics.

## 🚀 Future Work
Extend to a broader dataset, including unverified accounts.
Experiment with zero-shot classification techniques.
Explore explainability techniques to enhance model transparency.

Feel free to fork, star ⭐, and contribute to this repository! 😊
