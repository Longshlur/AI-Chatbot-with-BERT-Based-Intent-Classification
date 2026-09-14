import os
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "./saved_model"

BASE_MODEL = "bert-base-uncased"

MAX_LENGTH = 64

TEST_SIZE = 0.20

RANDOM_STATE = 42

EPOCHS = 3

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n================================")
print("📂 LOADING DATASET")
print("================================")

df = pd.read_csv("snips.csv")

print("\n✅ Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

if "text" not in df.columns or "intent" not in df.columns:

    raise ValueError(
        "❌ Dataset must contain 'text' and 'intent' columns."
    )


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

print("\n================================")
print("🧹 DATA CLEANING")
print("================================")

before_rows = len(df)

df = df.dropna(
    subset=["text", "intent"]
).reset_index(drop=True)

after_rows = len(df)

print(
    f"Removed {before_rows - after_rows} rows "
    "with missing values."
)

print(
    f"Remaining samples: {after_rows}"
)


# ============================================================
# 4. DISPLAY ORIGINAL INTENTS
# ============================================================

print("\n================================")
print("🎯 ORIGINAL INTENTS")
print("================================")

intent_counts = df["intent"].value_counts()

print(intent_counts)


print(
    f"\nTotal number of intents: "
    f"{df['intent'].nunique()}"
)


# ============================================================
# 5. VERIFY 8 INTENTS
# ============================================================

expected_intents = {
    "GetWeather",
    "SetAlarm",
    "SearchCreativeWork",
    "PlayMusic",
    "AddToList",
    "BookRestaurant",
    "GetNews",
    "SearchScreeningEvent"
}

actual_intents = set(
    df["intent"].unique()
)


if actual_intents != expected_intents:

    print(
        "\n⚠️ WARNING: Dataset intents are different "
        "from the expected 8 intents."
    )

    print(
        "\nExpected:"
    )

    for intent in sorted(expected_intents):
        print(
            f"  • {intent}"
        )

    print(
        "\nFound:"
    )

    for intent in sorted(actual_intents):
        print(
            f"  • {intent}"
        )

else:

    print(
        "\n✅ Dataset contains exactly the original "
        "8 intents."
    )


# ============================================================
# 6. ENCODE INTENTS
# ============================================================

print("\n================================")
print("🔢 ENCODING INTENTS")
print("================================")

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["intent"]
)


print("\nIntent → Label mapping:")

for index, intent in enumerate(
    label_encoder.classes_
):

    print(
        f"{index} → {intent}"
    )


num_labels = len(
    label_encoder.classes_
)


print(
    f"\nTotal classes: {num_labels}"
)


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# ============================================================

print("\n================================")
print("✂️ TRAIN / VALIDATION SPLIT")
print("================================")

train_texts, val_texts, train_labels, val_labels = (
    train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )
)


print(
    f"Training samples: "
    f"{len(train_texts)}"
)

print(
    f"Validation samples: "
    f"{len(val_texts)}"
)


# ============================================================
# 8. LOAD BERT TOKENIZER
# ============================================================

print("\n================================")
print("🔤 LOADING BERT TOKENIZER")
print("================================")

tokenizer = BertTokenizer.from_pretrained(
    BASE_MODEL
)

print(
    "✅ Tokenizer loaded."
)


# ============================================================
# 9. TOKENIZATION
# ============================================================

print("\n================================")
print("🔠 TOKENIZING DATA")
print("================================")


def tokenize(texts):

    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )


train_encodings = tokenize(
    train_texts
)

val_encodings = tokenize(
    val_texts
)


print(
    "✅ Tokenization completed."
)


# ============================================================
# 10. CUSTOM DATASET CLASS
# ============================================================

class IntentDataset(Dataset):

    def __init__(
        self,
        encodings,
        labels
    ):

        self.encodings = encodings

        self.labels = labels


    def __len__(self):

        return len(
            self.labels
        )


    def __getitem__(
        self,
        idx
    ):

        item = {

            key: torch.tensor(
                value[idx]
            )

            for key, value
            in self.encodings.items()

        }


        item["labels"] = torch.tensor(
            self.labels[idx]
        )


        return item


# ============================================================
# 11. CREATE DATASETS
# ============================================================

train_dataset = IntentDataset(
    train_encodings,
    train_labels
)

val_dataset = IntentDataset(
    val_encodings,
    val_labels
)


print(
    "\n✅ PyTorch datasets created."
)


# ============================================================
# 12. CREATE BERT MODEL
# ============================================================

print("\n================================")
print("🤖 CREATING BERT MODEL")
print("================================")

print(
    f"Number of classes: {num_labels}"
)


model = BertForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=num_labels
)


print(
    "✅ BERT model created."
)


# ============================================================
# 13. TRAINING CONFIGURATION
# ============================================================

print("\n================================")
print("⚙️ TRAINING CONFIGURATION")
print("================================")

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Training batch size: "
    f"{TRAIN_BATCH_SIZE}"
)

print(
    f"Evaluation batch size: "
    f"{EVAL_BATCH_SIZE}"
)


training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    logging_steps=10,

    report_to="none",

    save_strategy="no"
)


# ============================================================
# 14. EVALUATION METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred


    predictions = np.argmax(
        logits,
        axis=1
    )


    accuracy = accuracy_score(
        labels,
        predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1

    }


# ============================================================
# 15. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics

)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\n================================")
print("🚀 BERT TRAINING STARTED")
print("================================")

print(
    "\nPlease wait while BERT is fine-tuned..."
)


trainer.train()


print("\n================================")
print("✅ TRAINING FINISHED")
print("================================")


# ============================================================
# 17. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 18. SAVE TRAINED MODEL
# ============================================================

print("\n================================")
print("💾 SAVING MODEL")
print("================================")


model.save_pretrained(
    MODEL_DIR
)


tokenizer.save_pretrained(
    MODEL_DIR
)


# Save label names
np.save(
    os.path.join(
        MODEL_DIR,
        "label_classes.npy"
    ),
    label_encoder.classes_
)


print(
    "\n✅ Model saved successfully."
)


print(
    f"📁 Model directory: "
    f"{MODEL_DIR}"
)


# ============================================================
# 19. MODEL EVALUATION
# ============================================================

print("\n================================")
print("📊 MODEL EVALUATION")
print("================================")


results = trainer.evaluate()


accuracy = results[
    "eval_accuracy"
]

precision = results[
    "eval_precision"
]

recall = results[
    "eval_recall"
]

f1 = results[
    "eval_f1"
]


print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n================================")
print("📋 CLASSIFICATION REPORT")
print("================================")


predictions = trainer.predict(
    val_dataset
)


predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)


report = classification_report(

    val_labels,

    predicted_labels,

    target_names=label_encoder.classes_,

    zero_division=0
)


print(
    report
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

print("\n================================")
print("📈 CONFUSION MATRIX")
print("================================")


cm = confusion_matrix(

    val_labels,

    predicted_labels
)


plt.figure(
    figsize=(11, 9)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=label_encoder.classes_,

    yticklabels=label_encoder.classes_
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


plt.title(
    "BERT Intent Classification - "
    "Confusion Matrix"
)


plt.tight_layout()


plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n================================")
print("🎉 TRAINING COMPLETE")
print("================================")


print(
    f"\nNumber of intents: "
    f"{num_labels}"
)


print(
    f"Training samples: "
    f"{len(train_texts)}"
)


print(
    f"Validation samples: "
    f"{len(val_texts)}"
)


print(
    f"\nFinal Accuracy : "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Final Precision: "
    f"{precision * 100:.2f}%"
)


print(
    f"Final Recall   : "
    f"{recall * 100:.2f}%"
)


print(
    f"Final F1 Score : "
    f"{f1 * 100:.2f}%"
)


print(
    "\n📁 Saved model:"
)

print(
    MODEL_DIR
)


print(
    "\n🤖 To run the chatbot:"
)

print(
    "python app.py"
)