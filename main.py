import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
# You will need to replace this with the actual dataset loading code (e.g., pd.read_csv)
# For now, assuming 'data.csv' is your dataset file with the required columns
data = pd.read_csv('your_dataset.csv')  # Replace with actual path

# 2. Feature Selection (according to report)
selected_features = [
    'Protocol_Type', 'TCP_Flags', 'Header_Length', 'Packet_Magnitude', 'Inter_Arrival_Time'
    # Add the rest of relevant features here (total 45 features)
]
X = data[selected_features]

# 3. Target Variables
# For binary classification: 'Label' column with 'Benign' vs 'Attack'
# For multiclass classification: 'Attack_Type' column with classes like 'Benign', 'DDoS', 'DoS', 'Recon', 'Spoofing', 'MQTT'
# Adjust according to dataset's actual label columns
y_binary = data['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
y_multi = data['Attack_Type']

# 4. Encoding categorical features (Protocol_Type, TCP_Flags may be categorical)
for col in ['Protocol_Type', 'TCP_Flags']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 5. Normalize numerical features with Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split (70%-30%)
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
_, _, y_train_multi, y_test_multi = train_test_split(X_scaled, y_multi, test_size=0.3, random_state=42, stratify=y_multi)

# 7. Classical ML Models function to train and evaluate
def train_evaluate_classical(X_train, y_train, X_test, y_test, model, binary=True):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    if binary:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    else:
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
    return acc, precision, recall, f1, y_pred

# Initialize classical models with given parameters
dt_bin = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, criterion='gini', random_state=42)
rf_bin = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42)
svm_bin = SVC(kernel='rbf', probability=True, random_state=42)

dt_multi = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, criterion='gini', random_state=42)
rf_multi = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42)
svm_multi = SVC(kernel='rbf', probability=True, random_state=42)

# 8. Neural Network Model function
def create_nn_model(input_dim, output_dim, binary=True):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    if binary:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    else:
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    return model

# Prepare NN labels
# For binary classification (already y_train_bin, y_test_bin)
# For multiclass: one-hot encode y_multi
from tensorflow.keras.utils import to_categorical
le_multi = LabelEncoder()
y_train_multi_enc = le_multi.fit_transform(y_train_multi)
y_test_multi_enc = le_multi.transform(y_test_multi)
y_train_multi_ohe = to_categorical(y_train_multi_enc)
y_test_multi_ohe = to_categorical(y_test_multi_enc)
num_classes = y_train_multi_ohe.shape[1]

# Train NN for binary classification
nn_bin = create_nn_model(X_train.shape[1], 1, binary=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

nn_bin.fit(X_train, y_train_bin, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=2)

# Predict and evaluate binary NN
y_pred_bin_nn_prob = nn_bin.predict(X_test).flatten()
y_pred_bin_nn = (y_pred_bin_nn_prob > 0.5).astype(int)

acc_bin_nn = accuracy_score(y_test_bin, y_pred_bin_nn)
precision_bin_nn = precision_score(y_test_bin, y_pred_bin_nn)
recall_bin_nn = recall_score(y_test_bin, y_pred_bin_nn)
f1_bin_nn = f1_score(y_test_bin, y_pred_bin_nn)

# Train NN for multiclass classification
nn_multi = create_nn_model(X_train.shape[1], num_classes, binary=False)
nn_multi.fit(X_train, y_train_multi_ohe, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=2)

# Predict and evaluate multiclass NN
y_pred_multi_nn_prob = nn_multi.predict(X_test)
y_pred_multi_nn = np.argmax(y_pred_multi_nn_prob, axis=1)

acc_multi_nn = accuracy_score(y_test_multi_enc, y_pred_multi_nn)
precision_multi_nn = precision_score(y_test_multi_enc, y_pred_multi_nn, average='macro')
recall_multi_nn = recall_score(y_test_multi_enc, y_pred_multi_nn, average='macro')
f1_multi_nn = f1_score(y_test_multi_enc, y_pred_multi_nn, average='macro')

# Confusion matrix for NN multiclass
cm = confusion_matrix(y_test_multi_enc, y_pred_multi_nn)

# 9. Train and evaluate classical models on binary task
results_bin = {}
for name, model in zip(['Decision Tree', 'Random Forest', 'SVM'], [dt_bin, rf_bin, svm_bin]):
    acc, prec, rec, f1, _ = train_evaluate_classical(X_train, y_train_bin, X_test, y_test_bin, model, binary=True)
    results_bin[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

# Add NN results for binary
results_bin['Neural Network'] = {'Accuracy': acc_bin_nn, 'Precision': precision_bin_nn, 'Recall': recall_bin_nn, 'F1-Score': f1_bin_nn}

# 10. Train and evaluate classical models on multiclass task
results_multi = {}
for name, model in zip(['Decision Tree', 'Random Forest', 'SVM'], [dt_multi, rf_multi, svm_multi]):
    acc, prec, rec, f1, _ = train_evaluate_classical(X_train, y_train_multi_enc, X_test, y_test_multi_enc, model, binary=False)
    results_multi[name] = {'Accuracy': acc, 'Macro Precision': prec, 'Macro Recall': rec, 'Macro F1-Score': f1}

# Add NN results for multiclass
results_multi['Neural Network'] = {'Accuracy': acc_multi_nn, 'Macro Precision': precision_multi_nn, 'Macro Recall': recall_multi_nn, 'Macro F1-Score': f1_multi_nn}

# 11. Feature Importance using Random Forest (binary)
rf_bin.fit(X_train, y_train_bin)
importances = rf_bin.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# 12. Plot Confusion Matrix for NN multiclass
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_multi.classes_, yticklabels=le_multi.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Neural Network Multiclass')
plt.show()

# 13. Display results
print("Binary Classification Results:")
for model, metrics in results_bin.items():
    print(f"{model}: {metrics}")

print("\nMulticlass Classification Results:")
for model, metrics in results_multi.items():
    print(f"{model}: {metrics}")

print("\nTop Feature Importances from Random Forest (Binary Classification):")
print(feature_importance_df)

