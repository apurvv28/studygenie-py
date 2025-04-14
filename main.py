from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model and encoders
try:
    with open("models/study_style_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("models/label_encoders.pkl", "rb") as encoders_file:
        label_encoders = pickle.load(encoders_file)
    with open("models/target_encoder.pkl", "rb") as target_encoder_file:
        target_encoder = pickle.load(target_encoder_file)
    print("Model and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoders: {e}")

# FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QuizResponses(BaseModel):
    responses: dict

# Helper function to preprocess input data
def preprocess_input(responses):
    input_data = np.array([responses[str(i)] for i in range(1, 9)]).reshape(1, -1)
    # Preprocess and handle unseen labels
    for i in range(input_data.shape[1]):
        col_name = f'Q{i+1}'
        le = label_encoders.get(col_name)
        if le:
            try:
                # Replace unseen labels with the default (first label)
                if input_data[0, i] not in le.classes_:
                    input_data[0, i] = le.classes_[0]  # default class
                input_data[0, i] = le.transform([input_data[0, i]])[0]
            except Exception as e:
                print(f"Error encoding response for {col_name}: {e}")
    return input_data

@app.post("/api/predictLearningStyle")
async def predict_learning_style(quiz: QuizResponses):
    try:
        # Preprocess inputs
        responses = quiz.responses
        print(f"Received responses: {responses}")
        
        # Process the input data
        input_data = preprocess_input(responses)
        print(f"Input data after encoding: {input_data}")
        
        # Predict
        prediction = model.predict(input_data)
        predicted_style = target_encoder.inverse_transform(prediction)[0]
        print(f"Prediction: {predicted_style}")
        
        return {"prediction": predicted_style}
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")