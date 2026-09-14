# AI Chatbot with BERT-Based Intent Classification

An AI chatbot that uses **BERT (Bidirectional Encoder Representations from Transformers)** to understand user messages and classify them into predefined intents.

The project demonstrates how a Transformer-based NLP model can be fine-tuned for **intent classification** and integrated into a simple interactive chatbot.

## Project Overview

Traditional rule-based chatbots depend heavily on manually written rules and keywords.

This project uses **BERT-based text classification** to understand the meaning of user input and predict the most relevant intent.

The chatbot was trained using the **SNIPS intent classification dataset** and supports 8 different intents.

## Features

* BERT-based intent classification
* 8 predefined intents
* Confidence score for every prediction
* Confidence threshold for uncertain inputs
* Interactive command-line chatbot
* Train/validation split
* Accuracy, Precision, Recall and F1-score evaluation
* Confusion matrix during model evaluation
* Saved trained BERT model
* Simple and lightweight chatbot architecture

## Supported Intents

| Intent                 | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| `GetWeather`           | Weather-related queries                                       |
| `SetAlarm`             | Setting alarms                                                |
| `SearchCreativeWork`   | Searching for books, movies, songs and other creative content |
| `PlayMusic`            | Music-related requests                                        |
| `AddToList`            | Adding items to a list                                        |
| `BookRestaurant`       | Restaurant reservation requests                               |
| `GetNews`              | News and headline-related queries                             |
| `SearchScreeningEvent` | Movie screening and schedule queries                          |

## Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **BERT**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

## Project Structure

AI-Chatbot-with-BERT-Based-Intent-Classification/
│
├── app.py
├── train.py
├── snips.csv
├── requirements.txt
├── .gitignore
├── README.md
│
└── saved_model/          # Local trained model

> The trained BERT model is not included in the GitHub repository because the model file is approximately 438 MB, which exceeds GitHub's regular 100 MB file limit.

## Dataset

The project uses the **SNIPS intent classification dataset**.

For this project:

* Total samples: **1,200**
* Training samples: **960**
* Validation samples: **240**
* Number of intents: **8**

The dataset is stored in:

snips.csv

## Model

The project uses:

bert-base-uncased

The BERT model is fine-tuned for sequence classification.

### Classification Process

User Input
    │
    ▼
BERT Tokenizer
    │
    ▼
Fine-tuned BERT Model
    │
    ▼
Intent Classification
    │
    ▼
Confidence Score
    │
    ├── ≥ 70% ──► Predicted Intent ──► Chatbot Response
    │
    └── < 70% ──► Unknown
```

## Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The current validation evaluation achieved:

Accuracy  : 100%
Precision : 100%
Recall    : 100%
F1-Score  : 100%


> These results are based on the current validation split of the SNIPS dataset used in this project.

## Example

### Example 1 — Weather

You: what is the weather now

📊 Confidence: 99.71%

🤖 Intent: GetWeather

Bot: I can help you with weather-related questions.

### Example 2 — Music

You: play some music

Confidence: 99.50%

Intent: PlayMusic

Bot: Sure! I can help you with music-related requests.

### Example 3 — Unsupported Query

You: what is your name

Confidence: 38.25%

Intent: Unknown

Bot: I'm sorry, I don't understand that request.

### Sample Intent Tests

| User Input                    | Expected Intent        |
| ----------------------------- | ---------------------- |
| What is the weather today?    | `GetWeather`           |
| Play some music               | `PlayMusic`            |
| Set an alarm for 7 AM         | `SetAlarm`             |
| Book a restaurant for tonight | `BookRestaurant`       |
| Show me today's news          | `GetNews`              |
| Add milk to my list           | `AddToList`            |
| Find a movie screening        | `SearchScreeningEvent` |
| Search for a book             | `SearchCreativeWork`   |

## Installation

### 1. Clone the repository

git clone https://github.com/Longshlur/AI-Chatbot-with-BERT-Based-Intent-Classification.git


### 2. Navigate to the project

cd AI-Chatbot-with-BERT-Based-Intent-Classification

### 3. Create a virtual environment

Windows:

python -m venv .venv

### 4. Activate the virtual environment

PowerShell:

.venv\Scripts\Activate.ps1


### 5. Install dependencies


pip install -r requirements.txt


##  Training the Model

To train the BERT model:

python train.py


The training script:

1. Loads the SNIPS dataset
2. Splits the dataset into training and validation sets
3. Tokenizes the text using the BERT tokenizer
4. Fine-tunes BERT for intent classification
5. Evaluates the model
6. Generates evaluation metrics
7. Saves the trained model locally

The trained model is saved in:


saved_model/


##  Running the Chatbot

After training the model, run:


python app.py


The chatbot will start in the terminal.

================================
🤖 BERT CHATBOT READY
================================

Confidence threshold: 70%
Type 'exit' to stop.

You:

Type:

exit

to close the chatbot.

## Confidence Threshold

The chatbot uses a confidence threshold of:

70%


If the predicted intent has a confidence below this threshold, the chatbot returns:

Intent: Unknown


This provides a simple mechanism for handling inputs that the model is uncertain about.

##  Learning Outcomes

Through this project, I worked with:

* Natural Language Processing
* Transformer architectures
* BERT
* Text classification
* Tokenization
* Model fine-tuning
* PyTorch
* Hugging Face Transformers
* Model evaluation
* Confusion matrices
* Confidence-based prediction

##  Future Improvements

Possible improvements include:

* Adding more intents
* Using a larger and more diverse dataset
* Improving handling of out-of-domain queries
* Building a web-based chatbot interface
* Deploying the chatbot as an API
* Adding conversation history

##  Author

**Longshlur Surong**

B.Tech Computer Science Engineering
Government College of Engineering, Chhatrapati Sambhajinagar

GitHub:
https://github.com/Longshlur

## License

This project is created for educational and portfolio purposes.
