import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import io

# Placeholder function to load a trained model
# Replace this with actual model loading code
def load_model():
    # Example model (replace with your actual model)
    model = RandomForestClassifier()
    return model

# Placeholder function to preprocess image
def preprocess_image(image):
    # Convert image to grayscale
    gray_image = np.array(image.convert('L'))
    
    # Example preprocessing: Flatten image
    data = gray_image.flatten()

    # Example FFT transformation
    fft_data = np.fft.fft(data)
    fft_magnitude = np.abs(fft_data)
    
    # Normalize data
    scaler = StandardScaler()
    fft_magnitude = scaler.fit_transform(fft_magnitude.reshape(-1, 1)).flatten()
    
    return fft_magnitude

# Placeholder function to predict using the model
def predict(arrhythmia_model, data):
    prediction = arrhythmia_model.predict([data])
    return prediction[0]

def main():
    # Set custom CSS for styling
    st.markdown(
        """
        <style>
        .stApp {
        background-image: url('https://www.shutterstock.com/image-illustration/stethoscope-heart-ecg-graph-on-260nw-1939091212.jpghttps://www.shutterstock.com/image-illustration/stethoscope-heart-ecg-graph-on-260nw-1939091212.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
        .main .block-container {
            background-color: rgba(300, 300, 300, 0.9);  /* Semi-transparent white */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2;  /* Shadow for main content box */
            
        }
        .main .stFileUploader {
            background-color: rgba(230, 230, 230, 0.5);  /* Semi-transparent grey */
            border-radius: 5px;
            padding: 10px;
            border: 1px solid #cccccc;
        }
        .main .stButton button {
            background-color: #0073e6;  /* Button background color */
            color: white;  /* Button text color */
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .main .stButton button:hover {
            background-color: #005bb5;  /* Button background color on hover */
        }
        .stFileUploader label {
            color: #0073e6;  /* Text color for file uploader label */
        }
        .stButton button {
            background-color: #0381ff;  /* Button background color */
            color: white;  /* Button text color */
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;  /* Change cursor to pointer when hovering */
            transition: background-color 0.3s ease;  /* Smooth transition for hover effect */
        }
        .stButton button:hover {
            background-color: #005bb5;  /* Button background color on hover */
        }
        .stTextInput input {
            border-radius: 5px;
            padding: 8px;
            border: 1px solid #cccccc;
            font-size: 16px;
        }
         .image-container {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;  /* Ensure the image is above other elements */
        }
        .image-container img {
            width: 100px;  /* Set the desired width */
            border-radius: 5px;  /* Optional: rounded corners */
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent white */
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #333333;  /* Text color */
        }
        </style>
        <div class="image-container">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZoJ0B9kAjI3lZ4GNowxrlNr7ZcbMDHBzUmA&s" alt="Image">
        </div>
        <div class="footer">
            Developed by MOHAMMAD SAOOD, FAHAD IQBAL SHAH, QAZI MOHAMMAD YEHYA
        </div>
        
        """,
        unsafe_allow_html=True
    )

    st.title('DIGITAL STETHOSCOPE')

    st.write('Upload an ECG image to predict arrhythmia type.')

    uploaded_file = st.file_uploader("Choose an ECG image...")
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image, caption='Uploaded ECG Image.', use_column_width=True)
        
        # Preprocess the image
        processed_data = preprocess_image(image)
        
        # Load your machine learning model
        model = load_model()
        
        # Predict using the model
        result = predict(model, processed_data)
        
        # Display the result
        st.write(f'Prediction: {result}')
         # Input field for the file path
   
   
   
    file_path = st.text_input("Enter the file path of the ECG image:")
    
    if st.button("Enter"):
       if file_path:
        try:
            # Open the image using the provided file path
            image = Image.open(file_path)

            # Display the image
            st.image(image, caption='Uploaded ECG Image', use_column_width=True)
            
            # Preprocess the image
            processed_data = preprocess_image(image)
            
            # Load your machine learning model
            model = load_model()
            
            # Predict using the model
            result = predict(model, processed_data)
            
            # Display the result
            st.write(f'Prediction: {result}')
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
