import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
def tokenize(review_text):
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Predict sentiment
    sentiment_prediction = loaded_model.predict(padded_sequence)
    sentiment = "positive" if sentiment_prediction[0][0] > 0.5 else "negative"
    return sentiment