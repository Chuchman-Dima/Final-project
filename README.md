# 📈 AI Fundamentals Final Project: Linear Regression

This project is a full-cycle machine learning application built with **Python**, **Scikit-learn**, and **Streamlit**. It covers everything from synthetic data generation to interactive model deployment and containerization.

---

## 🚀 Live Demo
You can try the live interactive web application here:
👉 **[Link to Streamlit App](https://final-project-ayx7ntn2okqh2ece7ykm2x.streamlit.app/)**

## 📋 Project Overview & Purpose
The core objective is to build a simple linear regression model that predicts target values based on input features, and to make this model easily accessible via a web interface. The project is split into two main phases:
1. **Model Development:** Training, evaluating, and serializing the regression model.
2. **Web Deployment:** Serving the model via an interactive Streamlit application and containerizing it with Docker.

### 🗂 Project Structure
* `model.py`: Script for generating synthetic data, training the Linear Regression model, and saving artifacts (`.joblib`).
* `app.py`: The Streamlit web interface for real-time predictions and Matplotlib visualizations.
* `Dockerfile`: Instructions for containerizing the application using a lightweight Python image.
* `requirements.txt`: Project dependencies.

## 🛠 Tech Stack
* **Language:** Python 3.11
* **Machine Learning:** Scikit-learn (Linear Regression), Numpy
* **Web Framework:** Streamlit
* **Visualization:** Matplotlib
* **Serialization:** Joblib
* **DevOps & Deployment:** Docker, Docker Hub, Streamlit Community Cloud

## ✨ Key Features
### 1. Model Training & Evaluation
* **Synthetic Data:** Generation of a custom dataset for training and testing.
* **Model Serialization:** Saving the trained model as a `.joblib` file for efficient loading.
* **Performance Metrics:** Calculation of **Mean Squared Error (MSE)** to evaluate accuracy.

### 2. Interactive Web Application
* **Feature Input:** A slider to select feature values for real-time prediction.
* **Visual Comparison:** A scatter plot displaying the dataset, the actual target, and the predicted value.
* **Difference Annotation:** Automatic calculation and visualization of the error (difference) on the chart.

## 🚀 Getting Started

### 🐳 Option 1: Running with Docker (Recommended)
This application is fully containerized. You can run it on any machine without installing Python or libraries.

1. **Pull the image from Docker Hub:**
   ```bash
   docker pull dmytrochuchman/ai_fundamentals:streamlit-v1
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 dmytrochuchman/ai_fundamentals:streamlit-v1
   ```

3. **View the app:** Open your browser and go to [http://localhost:8501](http://localhost:8501)

---

### 💻 Option 2: Local Installation (Console)
If you prefer to run the code directly on your machine, follow these steps:

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Chuchman-Dima/Final-project.git
   ```

2. **Navigate to the app folder:**
   ```bash
   cd Final-project/src/task-initial
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Train the model (if needed):**
   ```bash
   python model.py
   ```

5. **Run the Streamlit app:**
   ```bash
   python -m streamlit run app.py
   ```

## 🎓 What I Learned & Improved
Throughout this project, I significantly improved my practical engineering skills:

* **End-to-End ML Pipeline:** Transitioned from a simple local script to a fully deployed web application.
* **Docker Containerization:** Applied and refined my existing knowledge by writing an efficient `Dockerfile`, optimizing image size using `python:3.11-slim`, mapping ports, and publishing the finalized image to Docker Hub.
* **Data Visualization:** Enhanced my ability to dynamically plot custom data points and annotations using Matplotlib within a Streamlit app.
