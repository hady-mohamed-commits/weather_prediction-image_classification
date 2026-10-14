🌦️ Weather Image Classification Using CNN

A Computer Vision project that uses a Convolutional Neural Network (CNN) to classify weather images into 11 different categories using PyTorch and Deep Learning.

🚀 Live Demo



🔗 Try the App: Weather Classification Demo

📌 Project Overview

This project classifies weather images into 11 categories, helping demonstrate how deep learning can be applied to image recognition tasks.

The model is trained on a labeled weather image dataset and can predict the weather category of a new image through an interactive Streamlit web application.

🌤️ Weather Categories

The model recognizes the following classes:

* Hail
* Rainbow
* Frost
* Rime
* Fog/Smog
* Snow
* Rain
* Glaze
* Lightning
* Sandstorm
* Dew

🛠️ Technologies Used

* Python — Main programming language
* PyTorch — Deep learning model development and training
* CNN — Image feature extraction and classification
* Torchvision — Image transformations and preprocessing
* Pillow (PIL) — Image loading and processing
* Matplotlib — Data visualization
* Scikit-learn — Model evaluation
* Streamlit — Interactive web application

⚙️ Project Workflow

1. 📂 Load and explore the weather image dataset.
2. 🔍 Inspect image dimensions, class distribution, and data quality.
3. 🧹 Clean and preprocess the images.
4. ✂️ Split the dataset into training, validation, and testing sets.
5. 🧠 Build a CNN model using PyTorch.
6. 🎯 Train the model and monitor validation performance.
7. 📊 Evaluate the model using classification metrics and a confusion matrix.
8. 💾 Save the trained model and class names.
9. 🌐 Deploy an interactive application using Streamlit.

📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Training and validation loss

Add your actual evaluation results here after training.

💻 Run the Project Locally

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_LINK
cd weather_classification

2. Install the dependencies

pip install -r requirements.txt

3. Run the Streamlit application

streamlit run app.py

Open the local URL displayed in your terminal to use the application.

📁 Project Structure

weather_classification/
├── app.py
├── weather_cnn.pth
├── classes.json
├── requirements.txt
└── README.md

* app.py: Streamlit application and prediction logic.
* weather_cnn.pth: Saved trained model weights.
* classes.json: Weather class names.
* requirements.txt: Required Python libraries.

🎯 Project Objectives

* Understand CNN architectures and image classification.
* Apply image preprocessing and data augmentation.
* Train and evaluate a deep learning model.
* Build and deploy a practical AI application.

👨‍💻 Author

Hady — Aspiring AI & Machine Learning Developer

🔗 GitHub: Your GitHub Profile

🔗 Live Demo: Weather Image Classification

⸻

⭐ If you find this project useful, feel free to star the repository!
