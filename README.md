# Medical Diagnosis Chatbot with BERT

A BERT-based intelligent chatbot to assist with preliminary disease prediction based on user-described symptoms. The model is fine-tuned on a multi-label symptom-disease dataset and supports natural language inputs such as:

> "I have chest pain and shortness of breath."

---

## Features

- BERT-based multi-class classification model  
- Predicts one of 772 disease labels based on symptoms  
- Accepts free-form symptom descriptions in English  
- Designed for triage assistance and educational use  

---

## Dataset

This project uses the [Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset) from Kaggle, containing:

- Approximately 250,000 entries  
- 772 disease classes  
- 377 binary symptom features  

**Dataset is not included in this repository** due to size and licensing.  
To use this project, please download the dataset manually from Kaggle.

---

The chatbot aims to:

Collect symptom descriptions from patients.

Predict possible medical conditions.

Estimate urgency levels to support healthcare professionals.

---

It's intended as part of a larger hospital triage system but can function independently for early testing and development.

This is a preliminary prototype. I'm trying different datasets and models for best performance. 

---

## License

The dataset is published under the World Bank Dataset Terms of Use.  
See [World Bank Terms](https://datacatalog.worldbank.org/public-licenses#cc-by) for details.

This project is for educational and research purposes only.
