
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Customer Support Ticket Intelligence API",
    description="Predicts the intent of a customer support message.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Load the trained model and TF-IDF vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

model = joblib.load(os.path.join(MODEL_DIR, "char_svm_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "char_tfidf_vectorizer.pkl"))


# Define the incoming request format
class TicketRequest(BaseModel):
    message: str


# Define the prediction endpoint
@app.get("/")
def home():
    return {"message": "Customer Support AI API is running!"}


@app.post("/predict")
def predict_intent(ticket: TicketRequest):
    message = [ticket.message]

    # Convert the message into TF-IDF features
    features = vectorizer.transform(message)

    # Predict the customer support intent
    prediction = model.predict(features)[0]

    return {
        "customer_message": ticket.message,
        "predicted_intent": str(prediction)
    }
