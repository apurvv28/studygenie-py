import pickle
import pandas as pd

# Load the saved model and encoders
with open('app/models/study_style_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('app/models/label_encoders.pkl', 'rb') as encoders_file:
    label_encoders = pickle.load(encoders_file)

with open('app/models/target_encoder.pkl', 'rb') as target_encoder_file:
    target_encoder = pickle.load(target_encoder_file)

# Function to preprocess new data
def preprocess_input(data):
    for column, le in label_encoders.items():
        # Replace unseen labels with a default value (e.g., the first class)
        default_value = le.classes_[0]  # Use the first class as the default
        data[column] = data[column].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([default_value])[0]
        )
    return data

# Example input data
new_data = pd.DataFrame({
    'Q1': ['Watching YouTube Videos'],
    'Q2': ['Reading textbooks'],
    'Q3': ['Handwritten Notes'],
    'Q4': ['Repeating key points in mind or aloud'],
    'Q5': ['Creating bullet points'],
    'Q6': ['Practice aloud with friends'],
    'Q7': ['Group Study'],
    'Q8': ['Writing detailed paragraphs']  
})

# Preprocess and predict
new_data = preprocess_input(new_data)
predicted_label = model.predict(new_data)
decoded_label = target_encoder.inverse_transform(predicted_label)

print(f"Predicted Learning Style: {decoded_label[0]}")