# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from imblearn.over_sampling import ADASYN

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Load datasets
normal_df = pd.read_csv('/content/drive/Normal_data.csv')
meta_df = pd.read_csv('/content/drive/metasploitable-2.csv')
ovs_df = pd.read_csv('/content/drive/OVS.csv')

# Display dataset shapes
print(f"Normal data shape: {normal_df.shape}")
print(f"Metasploitable data shape: {meta_df.shape}")
print(f"OVS data shape: {ovs_df.shape}")

# Combine all datasets
combined_df = pd.concat([normal_df, meta_df, ovs_df], ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# Data preprocessing
def preprocess_data(df):
    # Create a copy to avoid modifying original data
    df_processed = df.copy()

    # Remove non-numeric columns (Flow ID, IP addresses, Timestamp, etc.)
    # First row contains column names, so we need to handle it properly
    if df_processed.iloc[0, 0] == 'Flow ID':
        # This means the first row contains column names
        column_names = df_processed.iloc[0].values
        df_processed.columns = column_names
        df_processed = df_processed.iloc[1:].reset_index(drop=True)

    # Identify and remove non-numeric columns
    non_numeric_cols = []
    for col in df_processed.columns:
        try:
            # Try to convert to numeric
            pd.to_numeric(df_processed[col].iloc[0])
        except:
            non_numeric_cols.append(col)

    print(f"Non-numeric columns to be removed: {non_numeric_cols}")

    # Remove non-numeric columns except the label column
    if 'Label' in df_processed.columns:
        non_numeric_cols = [col for col in non_numeric_cols if col != 'Label']

    df_processed = df_processed.drop(columns=non_numeric_cols)

    # Convert all remaining columns to numeric
    for col in df_processed.columns:
        if col != 'Label':  # Don't convert label yet
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Fill NaN values with 0
    df_processed = df_processed.fillna(0)

    # Handle label encoding
    if 'Label' in df_processed.columns:
        # Binary classification: Normal = 0, Attack = 1
        df_processed['Label'] = df_processed['Label'].apply(
            lambda x: 0 if str(x).lower() == 'normal' else 1
        )

    return df_processed

# Preprocess the combined dataset
processed_df = preprocess_data(combined_df)
print(f"Processed dataset shape: {processed_df.shape}")

# Check label distribution
if 'Label' in processed_df.columns:
    label_counts = processed_df['Label'].value_counts()
    print(f"Label distribution:\n{label_counts}")
    print(f"Normal percentage: {label_counts[0]/len(processed_df)*100:.2f}%")
    print(f"Attack percentage: {label_counts[1]/len(processed_df)*100:.2f}%")

# Prepare features and labels
X = processed_df.drop('Label', axis=1).values
y = processed_df['Label'].values

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Normalize features using Z-score normalization (StandardScaler)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Apply ADASYN to handle class imbalance
print("Applying ADASYN to balance classes...")
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_normalized, y)

print(f"Data shape after ADASYN: {X_resampled.shape}")
print(f"Label distribution after ADASYN: {np.bincount(y_resampled)}")

# Reshape data for LSTM (convert to time series format)
# We need to reshape to (samples, timesteps, features)
# Since we have 84 features, we'll use timesteps=1 and treat each sample as a single timestep
# This is common for tabular data when using LSTM
timesteps = 1
features = X_resampled.shape[1]  # Use all 84 features

# Reshape the data - اینجا مشکل بود، درستش کردم
X_reshaped = X_resampled.reshape(-1, timesteps, features)
print(f"Reshaped features shape for LSTM: {X_reshaped.shape}")

# Split data into train and test sets (64% train, 36% test as in the paper)
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_resampled, test_size=0.36, random_state=42, stratify=y_resampled
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training label distribution: {np.bincount(y_train)}")
print(f"Testing label distribution: {np.bincount(y_test)}")

# Convert labels to categorical (one-hot encoding)
y_train_categorical = to_categorical(y_train, num_classes=2)
y_test_categorical = to_categorical(y_test, num_classes=2)

# Build the pure LSTM model
def create_lstm_model(input_shape, l2_lambda=0.1, dropout_rate=0.5):
    model = Sequential()

    # LSTM layer
    model.add(LSTM(128, return_sequences=False, input_shape=input_shape,
                  kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(dropout_rate))

    # Fully connected layer
    model.add(Dense(128, activation='relu',
                   kernel_regularizer=l2(l2_lambda)))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(2, activation='softmax'))

    return model

# Create model
input_shape = (X_train.shape[1], X_train.shape[2])
model = create_lstm_model(input_shape)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Display model architecture
model.summary()

# Train the model
history = model.fit(
    X_train, y_train_categorical,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
    X_test, y_test_categorical, verbose=0
)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Calculate F1-score
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
print(f"Test F1-Score: {test_f1:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix - Pure LSTM with ADASYN')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Pure LSTM with ADASYN')
plt.legend(loc="lower right")
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy - Pure LSTM with ADASYN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss - Pure LSTM with ADASYN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('/content/drive/MyDrive/Fereshte/pure_lstm_model_adasyn.h5')
print("Pure LSTM model with ADASYN saved successfully!")
