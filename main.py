import streamlit as st
import google.generativeai as genai
import os
import PIL.Image

os.environ["GOOGLE_API_KEY"] = "Aad your API key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash-latest")


def analyze_human_attributes(image):
    prompt = """
    You are an AI trained to analyze human attributes from images with high accuracy. 
    Carefully analyze the given image and return the following structured details:

    You have to return all results as you have the image, don't want any apologize or empty results.

    - **Gender** (Male/Female/Non-binary)
    - **Age Estimate** (e.g., 25 years)
    - **Ethnicity** (e.g., Asian, Caucasian, African, etc.)
    - **Mood** (e.g., Happy, Sad, Neutral, Excited)
    - **Facial Expression** (e.g., Smiling, Frowning, Neutral, etc.)
    - **Glasses** (Yes/No)
    - **Beard** (Yes/No)
    - **Hair Color** (e.g., Black, Blonde, Brown)
    - **Eye Color** (e.g., Blue, Green, Brown)
    - **Headwear** (Yes/No, specify type if applicable)
    - **Emotions Detected** (e.g., Joyful, Focused, Angry, etc.)
    - **Confidence Level** (Accuracy of prediction in percentage)
    """
    response = model.generate_content([prompt, image])
    return response.text.strip()

st.title("Human Attribute Detection")
st.write("Upload an image to detect human attributes with AI.")

uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)
    person_info = analyze_human_attributes(img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write(person_info)
