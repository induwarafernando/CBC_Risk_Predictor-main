# @title Default title text
# Install necessary libraries (only needed in Google Colab)
# !pip install pandas numpy scikit-learn tensorflow shap lime openpyxl matplotlib seaborn

# Import required libraries
import numpy as np
import pandas as pd
# import shap
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



# Load dataset in chunks to handle large files efficiently
file_path = "cbc_balanced_dataset.csv"
chunk_size = 5000  # Process data in smaller chunks
chunks = []  # List to store chunked data

# Read CSV in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunks.append(chunk)

# Combine all chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)

# Display dataset shape
print("✅ Data successfully loaded in chunks!")
print("Dataset Shape:", df.shape)

# Correcting column renaming
df.rename(columns={
    "WBC": "WBC",
    "Lymphocytes": "Lymphocytes",
    "Neutrophils": "Neutrophils",
    "RBC": "RBC",
    "Hemoglobin": "Hemoglobin",  # Fixed from HGB
    "Hematocrit": "Hematocrit",
    "Platelets": "Platelets",  # Fixed from PLT
}, inplace=True)

# Ensure required columns exist before proceeding
required_columns = ["WBC", "Lymphocytes", "Neutrophils", "RBC", "Hemoglobin", "Hematocrit", "Platelets"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("✅ Column names verified. Proceeding with data processing...")

# Calculate Neutrophil-to-Lymphocyte Ratio (NLR)
df["NLR"] = df["Neutrophils"] / df["Lymphocytes"]

# Function: Label Disease Conditions Based on Medical Thresholds
def label_disease(row):
    conditions = []
    if row["RBC"] < 4.0 or row["Hemoglobin"] < 12.0 or row["Hematocrit"] < 36:
        conditions.append("Anemia")
    if row["WBC"] > 11 or row["Neutrophils"] > 70:
        conditions.append("Infection Risk")
    if row["NLR"] > 3:
        conditions.append("Cardiovascular Risk")
    if row["WBC"] > 18:
        conditions.append("Leukemia Risk")
    return ", ".join(conditions) if conditions else "Normal"

df["DiseaseLabel"] = df.apply(label_disease, axis=1)  # Apply labeling function

# Convert categorical labels to numeric
df["DiseaseLabel"] = df["DiseaseLabel"].astype("category").cat.codes

# Handle Missing Values
df.fillna(df.median(), inplace=True)

# Ensure Each Class Has At Least 2 Samples
df = df[df["DiseaseLabel"].map(df["DiseaseLabel"].value_counts()) > 1]

# Define Features and Labels
X = df.drop(columns=["DiseaseLabel"])
y = df["DiseaseLabel"]

# Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display train-test split details
print("✅ Training Data Shape:", X_train.shape)
print("✅ Testing Data Shape:", X_test.shape)

# Build a Neural Network for EPOCH visualization
model_nn = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(y.unique()), activation='softmax')
])

# Compile Model
model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model & Track History
history = model_nn.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)

# Plot Training vs Validation Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# Evaluate Model Performance
y_pred = model_nn.predict(X_test).argmax(axis=1)

# Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)

# Print Evaluation Metrics
print("\n📊 **Model Evaluation Metrics**")
print("-----------------------------------------")
print(f"✅ **Accuracy:** {accuracy:.2f}")
print(f"✅ **Precision:** {precision:.2f}")
print(f"✅ **Recall:** {recall:.2f}")
print(f"✅ **F1-Score:** {f1:.2f}")
print("\n🔍 **Detailed Classification Report**")
print(classification_report(y_test, y_pred, zero_division=1))

# Visualization (PCA)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train.astype(str), palette="tab10", alpha=0.7, edgecolor="k")
plt.title("PCA Visualization of Disease Grouping")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Disease Distribution Bar Plot
df["DiseaseLabelName"] = df["DiseaseLabel"].map({
    0: "Normal",
    1: "Anemia",
    2: "Infection Risk",
    3: "Cardiovascular Risk",
    4: "Leukemia Risk"
})

plt.figure(figsize=(10, 6))
sns.countplot(y=df["DiseaseLabelName"], palette="coolwarm", order=df["DiseaseLabelName"].value_counts().index)
plt.xlabel("Number of Patients")
plt.ylabel("Disease Type")
plt.title("Distribution of Diseases in Training Data")
plt.show()


# Disease Prediction Function
def predict_disease_risk(cbc_values):
    input_data = pd.DataFrame([cbc_values], columns=X.columns)

    # Predict Disease Probability
    disease_prob = model_nn.predict(input_data)[0] * 100  # Use the correct model

    # Risk Calculations Based on Medical Thresholds - Corrected Indentation
    anemia_risk = max(0, min(100, (15 - input_data["Hemoglobin"].values[0]) * 10))  # Lower Hemoglobin = Higher Risk
    infection_risk = max(0, min(100, (input_data["WBC"].values[0] - 11) * 10))  # Higher WBC = Higher Infection Risk
    cardiovascular_risk = max(0, min(100, input_data["NLR"].values[0] * 10))  # High NLR = Cardiovascular Risk
    leukemia_risk = max(0, min(100, (input_data["WBC"].values[0] - 18) * 20))  # Extreme WBC levels → Leukemia Risk

    # ... (rest of the function remains the same)
    def classify_risk(value):
        if value >= 75:
            return "🔴 High"
        elif value >= 50:
            return "🟡 Moderate"
        else:
            return "🟢 Low"

    # Structured Alerts & Recommendations
    alerts = []
    if anemia_risk > 75:
        alerts.append("⚠️ **Anemia Warning:** Retest Hemoglobin levels.")
    if infection_risk > 70:
        alerts.append("⚠️ **Possible Infection Detected:** Consider bacterial infection screening.")
    if cardiovascular_risk > 80:
        alerts.append("⚠️ **Cardiovascular Risk Alert:** Recommend consulting a cardiologist.")
    if leukemia_risk > 80:
        alerts.append("⚠️ **Leukemia Risk Detected:** Immediate consultation with a hematologist recommended.")

    # Formatted Output for Readability
    output = (
        "\n📌 **Disease Risk Assessment Report**\n"
        "-----------------------------------------\n"
        f"✅ **Anemia Risk:** {anemia_risk:.2f}% ({classify_risk(anemia_risk)})\n"
        f"✅ **Infection Risk:** {infection_risk:.2f}% ({classify_risk(infection_risk)})\n"
        f"✅ **Cardiovascular Disease Risk:** {cardiovascular_risk:.2f}% ({classify_risk(cardiovascular_risk)})\n"
        f"✅ **Leukemia Risk:** {leukemia_risk:.2f}% ({classify_risk(leukemia_risk)})\n"
        "\n🏥 **Clinical Recommendations**\n"
        "-----------------------------------------\n"
        + ("\n".join(alerts) if alerts else "✅ No immediate concerns detected.") +
        "\n\n🔍 **Next Steps:** Consider further clinical evaluation based on risk levels.\n"
    )

    return output



# Sample Predictions
sample_patient = {
    "WBC": 12.5,
    "Lymphocytes": 20,
    "Neutrophils": 75,
    "RBC": 4.2,
    "Hemoglobin": 10.8,
    "Hematocrit": 34,
    "Platelets": 180,
    "NLR": 3.75
}

# Define example CBC values for different disease types
example_patients = {
    "Normal": {
        "WBC": 7.0, "Lymphocytes": 30, "Neutrophils": 55, "RBC": 4.5,
        "Hemoglobin": 14.0, "Hematocrit": 42, "Platelets": 250, "NLR": 1.83
    },
    "Anemia": {
        "WBC": 7.0, "Lymphocytes": 30, "Neutrophils": 55, "RBC": 3.5,
        "Hemoglobin": 9.5, "Hematocrit": 28, "Platelets": 250, "NLR": 1.83
    },
    "Infection Risk": {
        "WBC": 13.5, "Lymphocytes": 18, "Neutrophils": 75, "RBC": 4.8,
        "Hemoglobin": 13.0, "Hematocrit": 39, "Platelets": 210, "NLR": 4.17
    },
    "Cardiovascular Risk": {
        "WBC": 9.0, "Lymphocytes": 15, "Neutrophils": 70, "RBC": 4.6,
        "Hemoglobin": 13.5, "Hematocrit": 41, "Platelets": 240, "NLR": 4.67
    },
    "Leukemia Risk": {
        "WBC": 20.5, "Lymphocytes": 10, "Neutrophils": 80, "RBC": 4.7,
        "Hemoglobin": 12.0, "Hematocrit": 37, "Platelets": 180, "NLR": 8.00
    }
}

# Run predictions on all example patients
for disease, patient_data in example_patients.items():
    print(f"\n🩺 **Prediction for {disease} Patient**")
    print(predict_disease_risk(patient_data))

print("\n🧬 **Sample Prediction**")
print(predict_disease_risk(sample_patient))

# Visualization (PCA & t-SNE)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train.astype(str), palette="tab10", alpha=0.7, edgecolor="k")
plt.title("PCA Visualization of Disease Grouping")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Mapping numeric disease labels back to their original disease names
disease_mapping = {
    0: "Normal",
    1: "Anemia",
    2: "Infection Risk",
    3: "Cardiovascular Risk",
    4: "Leukemia Risk"
}

df["DiseaseLabelName"] = df["DiseaseLabel"].map(disease_mapping)

# Disease Distribution Bar Plot
plt.figure(figsize=(10, 6))
sns.countplot(y=df["DiseaseLabelName"], palette="coolwarm", order=df["DiseaseLabelName"].value_counts().index)
plt.xlabel("Number of Patients")
plt.ylabel("Disease Type")
plt.title("Distribution of Diseases in Training Data")
plt.show()
# Fix countplot issue and add better readability
plt.figure(figsize=(10, 6))
sns.countplot(y=df["DiseaseLabelName"], hue=df["DiseaseLabelName"], palette="coolwarm", legend=False)
plt.xlabel("Number of Patients")
plt.ylabel("Disease Type")
plt.title("Distribution of Diseases in Training Data")
plt.show()

model_nn.save("cbc_disease_model.h5")


