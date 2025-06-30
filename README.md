# MindPulse
EEG-based Emotion Detection system using XGBoost and Flask. Utilizes brainwave signals to classify emotional states (Positive, Neutral, Negative) and deploys a real-time web interface for user interaction.
# Emotion Detection Using EEG Signals 

This project implements a machine learning-based system to detect human emotions using EEG (Electroencephalography) signals. By leveraging the XGBoost algorithm and deploying the model via a Flask web application, the system can classify emotional states **Positive**, **Neutral**, and **Negative** based on brainwave data.

🚀 Project Motivation
Emotion-aware systems are increasingly vital in healthcare, e-learning, gaming, and human-computer interaction (HCI). EEG signals offer real-time and non-invasive insights into a user's emotional state. This project aims to make emotion recognition accessible, accurate, and deployable in practical environments.

🧠 Dataset
- **Name**: EEG Brainwave Dataset: Feeling Emotions
- **Source**: [Kaggle]((https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions))
- **Signals Captured**: TP9, TP10, AF7, AF8 (Muse Headband)
- **Emotion Labels**: POSITIVE, NEGATIVE, NEUTRAL
- **Format**: Preprocessed CSV with FFT features

🔧 Technologies Used

| Layer | Tools |
|-------|-------|
| ML Model | XGBoost (Gradient Boosted Trees) |
| Preprocessing | NumPy, Pandas, Scikit-learn |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Hosting (Local) | Flask Server |

 System Architecture

1. **Data Collection**: EEG signal acquisition via Muse headband.
2. **Preprocessing**: Noise removal, label encoding, and normalization.
3. **Feature Extraction**: Band powers (Alpha, Beta, Theta, Delta).
4. **Model Training**: XGBoost with cross-validation and hyperparameter tuning.
5. **Deployment**: Flask API with a user-friendly web interface.

  Results

- **Model Accuracy**: ~100% on the test set (confusion matrix shows perfect classification).
- **Distribution**:
  - POSITIVE: 39.5%
  - NEUTRAL: 34.9%
  - NEGATIVE: 25.6%

🖥 Demo
Users can enter EEG band values via a web form and get real-time emotion predictions through a Flask API.








📄 License: *Academic Use Only*

