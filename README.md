# ECG Heartbeat Classification Project

---

## 🔬 Introduction

This project details the development of a classification model for **electrocardiogram (ECG)** signals, aiming to accurately differentiate between normal and abnormal cardiac rhythms. Leveraging deep learning methodologies, specifically **Long Short-Term Memory (LSTM) networks**, the objective is to construct a robust system capable of identifying anomalies within sequential ECG data. The methodology encompasses rigorous data preprocessing, model training with strategies for imbalanced datasets, and comprehensive performance evaluation.

---

## 📊 Dataset

The model's training and validation are performed using the "Heartbeat Sounds" dataset, obtained from Kaggle. Specifically, the `ptbdb_abnormal.csv` file, containing preprocessed ECG time series data, is utilized to facilitate the development of a binary classification model for cardiac anomaly detection.

* **Dataset Source:** [Heartbeat Sounds Dataset on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data?select=ptbdb_abnormal.csv)

---

## 📦 System Requirements

Successful execution and replication of this project necessitate the installation of the following Python libraries:

* **`pandas`**: For data manipulation and tabular data operations.
* **`numpy`**: Essential for numerical computations and array manipulations.
* **`scikit-learn` (`sklearn`)**: Provides tools for data splitting, preprocessing (e.g., `StandardScaler`), and various performance metrics.
* **`tensorflow`**: The foundational library for building and training deep neural networks, utilizing its Keras API.
* **`matplotlib`**: For static, animated, and interactive visualizations in Python.
* **`imbalanced-learn` (`imblearn`)**: Specifically for `SMOTE` (Synthetic Minority Over-sampling Technique), addressing class imbalance in the training data.

---

## 🧠 Model Architecture

The core of the classification system is a **Sequential Keras model**, designed to effectively process the temporal dependencies inherent in ECG signals. The architecture comprises:

* A **`Masking` layer**: This layer is implemented to disregard padded values in input sequences, ensuring that subsequent layers process only meaningful data points.
* **`LSTM` layers**: A hierarchical arrangement of three LSTM layers with 64, 32, and 16 units, respectively, are employed to capture complex long-range dependencies within the time-series ECG data.
* A **`Dense` output layer**: Consisting of a single neuron with a `sigmoid` activation function, this layer outputs the predicted probability of the input ECG signal representing an abnormal heartbeat.
* **Compilation**: The model is compiled using the `Adam` optimizer, an adaptive learning rate optimization algorithm, and `binary_crossentropy` as the loss function, suitable for binary classification tasks.

---

## 🧪 Methodology

The project's methodological framework is structured to ensure robust model training and evaluation:

1.  **Data Partitioning**: The dataset is systematically divided into training, validation, and test sets. This rigorous partitioning strategy is critical for training the model, fine-tuning hyperparameters, and providing an unbiased evaluation of generalization performance.
2.  **Class Imbalance Mitigation**: To counteract the typical imbalance between normal and abnormal heartbeat samples, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied exclusively to the training dataset. This technique synthesizes new examples of the minority class, thereby balancing the class distribution and enhancing the model's ability to learn from and classify abnormal patterns.
3.  **Model Training**: The LSTM model is trained using the resampled training data. Training incorporates several strategies to optimize performance and prevent overfitting:
    * **Early Stopping**: A callback mechanism that monitors the validation loss and halts training when no significant improvement is observed over a specified number of epochs, restoring the best weights.
    * **Class Weighting**: Class weights are computed and applied during training to further adjust for the inherent imbalance, assigning higher penalties for misclassifications of the minority class.
    * **Optimizer and Learning Rate Scheduling**: The Adam optimizer is utilized, with a `ReduceLROnPlateau` callback implicitly suggested by the imports, which reduces the learning rate when a metric has stopped improving, aiding convergence.
4.  **Cross-Validation**: The overall model performance is assessed through a **5-fold cross-validation** approach (implied by the "5 split run" and use of `StratifiedKFold` in imports). This technique provides a more reliable estimate of the model's performance on unseen data by averaging results across multiple train/test splits.

---

## 📈 Results

The performance of the classification model was rigorously evaluated across 5 distinct data splits using standard classification metrics. The aggregated results are presented as mean $\pm$ standard deviation, demonstrating the model's efficacy:

* **Overall Accuracy:** $0.9599 \pm 0.0212$
* **Overall Precision:** $0.9454 \pm 0.0257$
* **Overall Recall:** $0.9573 \pm 0.0232$
* **Overall F1-Score:** $0.9511 \pm 0.0246$

These metrics collectively indicate the model's strong capability in accurately distinguishing between normal and abnormal ECG signals, highlighting its potential for robust cardiac anomaly detection.
