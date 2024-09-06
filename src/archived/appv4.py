import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import sqlite3
import hashlib
import os
from PIL import Image

# Include Bootstrap CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        .stTitle {
            font-size: 2em;
            color: #333;
        }
        .stSidebar {
            background-color: #ffffff;
        }
        .stButton {
            background-color: #007bff;
            color: #fff;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hash_password(password)))
    result = c.fetchone()
    conn.close()
    return result is not None

# Create a model with BatchNormalization
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax')  # 26 for A-Z signs
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image):
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize((224, 224))  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

def upload_and_predict():
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)
        
        # Load the model
        model = load_or_create_model()
        
        # Predict
        prediction = model.predict(preprocessed_image)
        predicted_letter = chr(np.argmax(prediction) + 65)  # Convert to A-Z
        
        st.image(image, caption=f"Uploaded Image")
        st.write(f"Predicted Letter: {predicted_letter}")

def train_model():
    # Load and preprocess training data
    X_train = []
    y_train = []
    for label in range(26):  # A-Z
        for i in range(1, 21):  # Assume 20 images per letter
            img_path = f"data/{chr(label + 65)}_{i}.jpg"
            if os.path.exists(img_path):
                img = Image.open(img_path)
                X_train.append(preprocess_image(img))
                y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(tf.keras.utils.to_categorical(y_train, num_classes=26))

    model = create_model()
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    model.save('models/sign_language_model.h5')

def test_model():
    model = tf.keras.models.load_model('models/sign_language_model.h5')
    X_test = []
    y_test = []
    for label in range(26):  # A-Z
        img_path = f"data/{chr(label + 65)}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            X_test.append(preprocess_image(img))
            y_test.append(label)

    X_test = np.array(X_test)
    y_test = np.array(tf.keras.utils.to_categorical(y_test, num_classes=26))

    loss, accuracy = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {loss}")
    st.write(f"Test Accuracy: {accuracy}")

def load_or_create_model():
    if os.path.exists('models/sign_language_model.h5'):
        return tf.keras.models.load_model('models/sign_language_model.h5')
    else:
        model = create_model()
        model.save('models/sign_language_model.h5')  # Save the default model
        return model

def main():
    st.title("TECADE Sign Language Interpreter")

    menu = ["Home", "Login", "Register", "Train", "Test", "Upload Image"]
    choice = st.sidebar.selectbox("Select Page", menu)

    model = load_or_create_model()  # Load or create the model

    if choice == "Home":
        st.write("Welcome to the TECADE Sign Language Interpreter app.")
        st.subheader("Live Video Feed")
        video_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            predicted_letter = chr(np.argmax(prediction) + 65)  # Convert to A-Z

            frame = cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_placeholder.image(frame, channels="BGR")

    elif choice == "Upload Image":
        upload_and_predict()

    elif choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if authenticate_user(username, password):
                st.success("Logged in successfully")
            else:
                st.error("Invalid username or password")

    elif choice == "Register":
        st.subheader("Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registration successful")
            else:
                st.error("Username already exists")

    elif choice == "Train":
        if st.button("Train Model"):
            train_model()
            model = load_or_create_model()  # Reload the trained model
            st.success("Model trained successfully")

    elif choice == "Test":
        if st.button("Test Model"):
            test_model()

    elif choice == "Upload Image":
        st.subheader("Upload Image")
        uploaded_files = st.file_uploader("Choose images", type=["jpg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                processed_image = preprocess_image(image)
                prediction = model.predict(np.expand_dims(processed_image, axis=0))
                predicted_letter = chr(np.argmax(prediction) + 65)  # Convert to A-Z
                st.image(uploaded_file, caption=f"Uploaded Image", use_column_width=True)
                st.write(f"Predicted Letter: {predicted_letter}")

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = cv2.resize(frame, (224, 224))  # Resize to model's input size
    frame = frame / 255.0  # Normalize
    return np.expand_dims(frame, axis=0)  # Add batch dimension

if __name__ == "__main__":
    init_db()
    main()
