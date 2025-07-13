# 🧠 Self-Organizing Map: Credit Card Fraud Detection

This is my first Unsupervised Deep Learning project using a **Self-Organizing Map (SOM)** to detect potential fraud in credit card applications. I built this project while learning from the **SuperDataScience Deep Learning A-Z™** course on **Udemy**.

---

## 📚 What I Learned

- How to build and train a Self-Organizing Map (SOM)
- Preprocessing data using MinMaxScaler
- Visualizing high-dimensional data on a 2D grid
- Detecting outliers based on **MID (Mean Interneuron Distance)**
- Using the `MiniSom` library in Python

---

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- MiniSom (`minisom` library)
- Scikit-learn

---

## 📁 Dataset

The dataset used is `Credit_Card_Application.csv`, included in the Udemy course materials. It contains various customer features along with a final column indicating whether the credit card application was approved (`1`) or not (`0`).

> ⚠️ Dataset is not publicly shared due to course content licensing.

---

## 🔍 How It Works

1. **Load the dataset**
2. **Preprocess and scale the features**
3. **Train a SOM on the data (10x10 grid)**
4. **Visualize the MID (Mean Interneuron Distance) to identify outliers**
5. **Highlight customers mapped to high MID neurons (potential frauds)**

---

## 📈 Visualization

- White squares on the SOM grid indicate **higher MID**, suggesting anomaly.
- Red (`o`) and green (`s`) markers represent whether the application was rejected or approved.
- Customers linked to high-MID neurons are considered **potential fraud cases**.

---

## 📦 Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/som-fraud-detector.git
   cd som-fraud-detector
