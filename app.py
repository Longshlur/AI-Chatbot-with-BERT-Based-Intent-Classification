import os
import torch
import numpy as np

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "./saved_model"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# 1. CHECK SAVED MODEL
# ============================================================

if not os.path.exists(MODEL_DIR):

    print("\n❌ saved_model folder not found!")

    print("\nMake sure you have:")
    print("saved_model/")

    exit()


# ============================================================
# 2. LOAD TOKENIZER
# ============================================================

print("\n🔤 Loading BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_DIR
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("🤖 Loading trained BERT model...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()


# ============================================================
# 4. LOAD ORIGINAL 8 INTENTS
# ============================================================

label_file = os.path.join(
    MODEL_DIR,
    "label_classes.npy"
)


if os.path.exists(label_file):

    label_classes = np.load(
        label_file,
        allow_pickle=True
    )

else:

    # Original 8 SNIPS intents
    label_classes = np.array([
        "AddToList",
        "BookRestaurant",
        "GetNews",
        "GetWeather",
        "PlayMusic",
        "SearchCreativeWork",
        "SearchScreeningEvent",
        "SetAlarm"
    ])


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print("\n================================")
print("📦 MODEL INFORMATION")
print("================================")

print(
    f"Number of intents: {len(label_classes)}"
)

print("\nSupported intents:")

for intent in label_classes:

    print(
        f"  • {intent}"
    )


# ============================================================
# 5. INTENT RESPONSES
# ============================================================

INTENT_RESPONSES = {

    "GetWeather":
        "I can help you with weather-related questions.",

    "SetAlarm":
        "Sure! I can help you set an alarm.",

    "SearchCreativeWork":
        "I can help you search for movies, books, songs, or other creative content.",

    "PlayMusic":
        "Sure! I can help you with music-related requests.",

    "AddToList":
        "Sure! I can help you add items to your list.",

    "BookRestaurant":
        "Sure! I can help you with restaurant reservations.",

    "GetNews":
        "I can help you find news and headlines.",

    "SearchScreeningEvent":
        "I can help you find movie schedules and screening information."
}


# ============================================================
# 6. PREDICT INTENT
# ============================================================

def predict_intent(text):

    # Tokenize user input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )


    # Model prediction
    with torch.no_grad():

        outputs = model(
            **inputs
        )


    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )


    # Get highest probability
    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )


    predicted_class = predicted_class.item()

    confidence = confidence.item()


    # Convert class number to intent
    intent = label_classes[predicted_class]


    return intent, confidence


# ============================================================
# 7. START CHATBOT
# ============================================================

print("\n================================")
print("🤖 BERT CHATBOT READY")
print("================================")

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
)

print(
    "Type 'exit' to stop."
)

print("================================\n")


# ============================================================
# 8. CHAT LOOP
# ============================================================

while True:

    user_input = input(
        "You: "
    )


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if user_input.lower().strip() == "exit":

        print(
            "Bot: Goodbye 👋"
        )

        break


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not user_input.strip():

        print(
            "Bot: Please enter a message.\n"
        )

        continue


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    intent, confidence = predict_intent(
        user_input
    )


    # --------------------------------------------------------
    # DISPLAY CONFIDENCE
    # --------------------------------------------------------

    print(
        f"\n📊 Confidence: "
        f"{confidence * 100:.2f}%"
    )


    # --------------------------------------------------------
    # LOW CONFIDENCE = UNKNOWN
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        print(
            "🤖 Intent: Unknown"
        )

        print(
            "Bot: I'm sorry, I don't understand "
            "that request.\n"
        )

        continue


    # --------------------------------------------------------
    # KNOWN INTENT
    # --------------------------------------------------------

    print(
        f"🤖 Intent: {intent}"
    )


    response = INTENT_RESPONSES.get(
        intent,
        "I understood your request."
    )


    print(
        f"Bot: {response}\n"
    )