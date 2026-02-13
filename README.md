# 📚 Assignment Submission Authenticity Detection System

This project is an AI-powered web application built to analyze assignment submissions and determine whether the content is AI-generated or written by a human. The system also includes plagiarism support and an AI text humanization feature, making it a complete solution for academic submission authenticity checking.

The main goal of this project is to help educators and institutions evaluate the originality of submitted work in a practical and automated way.

---

## 🚀 What This Project Does

When a user uploads a file (PDF, DOCX, or even a ZIP containing multiple files), the system:

- Extracts the text from the file
- Processes and cleans the content
- Converts the text into numerical format using vectorization
- Uses a trained Machine Learning model (MLP Classifier) to predict
- Displays the percentage of AI-generated vs Human-written content
- Optionally rewrites content using a humanization feature

It provides a simple web interface where users can upload documents and instantly see results.

---

## 🛠 Technologies Used

This project is built using:

- **Python** for backend logic
- **Flask** for the web framework
- **Scikit-learn** for the Machine Learning model
- **MLP Classifier** for AI detection
- **CountVectorizer** for text vectorization
- **Transformers (BERT) & Torch** for deeper NLP processing
- **PyPDF2 & docx2txt** for extracting text from files
- **Joblib** for loading trained models

---

## 📁 Project Structure

The main application logic is inside the `ai-detect-remove` folder, which includes:

- `app.py` – Main Flask application
- `ai_detector.py` – AI detection logic
- `ai_remover.py` – Humanization/rewrite feature
- `file_utils.py` – File extraction utilities
- `templates/` – HTML frontend files
- `requirements.txt` – All required Python packages

Additionally, trained model files (`.joblib`) are included for prediction.

---

👨‍💻 Author

Mukesh Ahirwar
B.Tech in Computer Science (AI & ML)
Indian Institute of Information Technology Nagpur

GitHub: https://github.com/mukeshahirwar9644

LinkedIn: https://www.linkedin.com/in/MukeshAhirwar9644
