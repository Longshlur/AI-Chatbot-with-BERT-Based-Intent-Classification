# AI Chatbot with BERT-Based Intent Classification

An AI chatbot that uses **BERT (Bidirectional Encoder Representations from Transformers)** to understand user messages and classify them into predefined intents.

The project demonstrates how a Transformer-based NLP model can be fine-tuned for **intent classification** and integrated into a simple interactive chatbot.


## Project Overview

Traditional rule-based chatbots depend heavily on manually written rules and keywords.

This project uses **BERT-based text classification** to understand the meaning of user input and predict the most relevant intent.

The chatbot was trained using the **SNIPS intent classification dataset** and supports 8 different intents.

## Features

- BERT-based intent classification
- 8 predefined intents
- Confidence score for every prediction
- Confidence threshold for uncertain inputs
- Interactive command-line chatbot
- Train/validation split
- Accuracy, Precision, Recall and F1-score evaluation
- Confusion matrix during model evaluation
- Saved trained BERT model
- Simple and lightweight chatbot architecture



## Supported Intents

The chatbot currently supports:

| Intent | Description |
|---|---|
| `GetWeather` | Weather-related queries |
| `SetAlarm` | Setting alarms |
| `SearchCreativeWork` | Searching for books, movies, songs and other creative content |
| `PlayMusic` | Music-related requests |
| `AddToList` | Adding items to a list |
| `BookRestaurant` | Restaurant reservation requests |
| `GetNews` | News and headline-related queries |
| `SearchScreeningEvent` | Movie screening and schedule queries |


##  Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **BERT**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**


## Project Structure

```text
AI-Chatbot-with-BERT-Based-Intent-Classification/
│
├── app.py
├── train.py
├── snips.csv
├── .gitignore
├── README.md
│
└── saved_model/          # Local trained model