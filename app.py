import nltk
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from tkinter import filedialog
import json
import random

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load intents from JSON file
with open('dataset.json') as file:
    data = json.load(file)

bow_patterns = []
bow_tags = []
data_pattern = []
data_tag = []

# Each pattern, tag is added to its corresponding BoW list
for tag in data:
    for pattern in data[tag]["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        bow_patterns.extend(tokens)
        data_pattern.append(pattern)
        data_tag.append(tag)
    if tag not in bow_tags:
        bow_tags.append(tag)

# Each pattern is lemmatized i.e brought to its base form
lemmatizer = WordNetLemmatizer()
bow_patterns = [lemmatizer.lemmatize(pattern.lower()) for pattern in bow_patterns if pattern not in string.punctuation]
bow_patterns = sorted(set(bow_patterns))
bow_tags = sorted(set(bow_tags))

training = []
out_empty = [0] * len(bow_tags)
for idx, doc in enumerate(data_pattern):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for pattern in bow_patterns:
        bow.append(1) if pattern in text else bow.append(0)
        output_row = list(out_empty)
        output_row[bow_tags.index(data_tag[idx])] = 1
        training.append([bow, output_row])
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Neural Network Model Definition (modified for integration)
def create_neural_network(vocab_size, num_classes):
    model = Sequential()
    model.add(Dense(128, input_shape=(vocab_size,), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    return model

vocab_size = len(set(bow_patterns))
num_classes = len(bow_tags)
model = create_neural_network(vocab_size, num_classes)
model.fit(x=train_x, y=train_y)

# Function to Preprocess Text (combined from both parts)
def clean_and_bag_of_words(text, vocab):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in string.punctuation]
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

# Function to Predict Class with Threshold (modified for clarity)
def predict_class(text, vocab, labels, threshold=0.5):
    bow = clean_and_bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    predicted_classes = [[i, res] for i, res in enumerate(result) if res > threshold]
    predicted_classes.sort(key=lambda x: x[1], reverse=True)  # Sort by highest probability
    return_list = [labels[i] for i, _ in predicted_classes]  # Extract class labels
    return return_list

# Function to get the response
def get_response(message):
    for tag in data:
        for pattern in data[tag]['patterns']:
            if pattern in message.lower():
                return random.choice(data[tag]['responses'])
    
    # If no exact match found, use the neural network for partial matching
    predictions = predict_class(message, bow_patterns, bow_tags)

    if not predictions:  # No predictions above threshold
        return "I'm sorry, I didn't understand that. Perhaps you could rephrase or ask something related to yoga?"
    elif len(predictions) == 1:  # Single closest prediction
        matched_intent = predictions[0]
        for tag in data:
            if tag == matched_intent:
                response = random.choice(data[tag]['responses'])
                return response
    else:  # Multiple predictions (consider returning most likely or prompting for clarification)
        response_options = ", ".join(predictions)
        return "I am not sure what you mean. Did you perhaps mean: " + response_options + "?"

def send_message(event=None):
    global chat_box
    message = entry.get()
    entry.delete(0, tk.END)

    if message.lower() == 'quit':
        window.destroy()
        return

    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, "You: " + message + "\n", "user_message")

    response = get_response(message)
    chat_box.insert(tk.END, "ChatBot: " + response + "\n\n", "bot_message")

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, "You uploaded: " + file_path + "\n", "user_message")
        chat_box.config(state=tk.DISABLED)

# Create GUI window
window = tk.Tk()
window.title("Yoga ChatBot")
window.geometry("400x500")

# Set dark mode theme
bg_color = "#efeff8"  # Dark background color
frame_color = "black"  # Light foreground color
heading_color = "#3a3d5c" # Blue color
window.configure(bg=heading_color)

# Add custom styles for dark mode
style = ttk.Style()
style.configure('TLabel', background=heading_color, foreground=frame_color)
style.configure('Label', background=frame_color, foreground=bg_color)  # Set text color to aqua green
style.configure('TButton', background="white", foreground="black")
style.configure('TEntry', background=frame_color, foreground="black")
style.configure('TFrame', background=frame_color)

# Welcome message label
welcome_label = ttk.Label(window, text="ChatBot", style='TLabel', font=('Arial', 30, 'bold'))
welcome_label.pack(pady=10)

# Create chat history display with dark background
chat_frame = ttk.Frame(window)
chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_label = ttk.Label(chat_frame, text="Chat History", style='Label', font=('Arial', 10, 'bold'))
chat_label.pack(pady=(10, 5))

chat_box = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=40, height=15, bg=bg_color, fg=frame_color)
chat_box.tag_config("user_message", foreground="black", font=('Times new roman', 13, 'bold italic'))  # Apply aqua green color to user message
chat_box.tag_config("bot_message", foreground="black", font=('Times new roman', 13, 'bold'))  # Apply light color to chatbot response
chat_box.insert(tk.END, "Welcome to the Yoga ChatBot!\nType 'quit' to exit!\n", "bot_message")
chat_box.pack(fill=tk.BOTH, expand=True)

# Create user input field with dark background
entry_frame = ttk.Frame(window)
entry_frame.pack(padx=10, pady=(0, 10), fill=tk.BOTH)
entry = ttk.Entry(entry_frame, width=40)
entry.pack(side=tk.LEFT, padx=(5, 0), fill=tk.BOTH, expand=True)

# Create send button with dark background
send_button = ttk.Button(entry_frame, text="Send", command=send_message, style='TButton')
send_button.pack(side=tk.RIGHT, padx=(0, 5))

# Create upload image button
upload_button = ttk.Button(window, text="Upload Image", command=upload_image, style='TButton')
upload_button.pack(side=tk.RIGHT, padx=(0, 10))

# Bind Enter key to send_message function
window.bind('<Return>', send_message)

# Start the GUI event loop
window.mainloop()
