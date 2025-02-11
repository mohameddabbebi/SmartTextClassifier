#This how i used my trained model with gradio interface 

import gradio as gr
import joblib
# Load the trained model
clf_loaded = joblib.load('/content/news_classifier_model.pkl')

categories = [
    "COMEDY", "SPORTS", "CRIME", "EDUCATION", "GOOD NEWS", "WORLD NEWS",
    "ENTERTAINMENT", "IMPACT", "POLITICS", "WEIRD NEWS", "BLACK VOICES",
    "WOMEN", "QUEER VOICES", "BUSINESS", "TRAVEL", "MEDIA", "TECH",
    "RELIGION", "SCIENCE", "PARENTS", "ARTS & CULTURE", "STYLE",
    "THE WORLDPOST", "HEALTHY LIVING", "TASTE"
]

# Prediction function
def predict_category(text):
    prediction = clf_loaded.predict([text])[0]  # Get the predicted category number
    return categories[prediction]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_category,  # Function to call
    inputs=gr.Textbox(lines=3, placeholder="Enter news text..."),  # User input
    outputs="text",  # Output as text
    title="News Category Classifier",
    description="Enter a news headline or article snippet, and the model will classify it into one of the predefined categories."
)

# Launch the app
interface.launch()
