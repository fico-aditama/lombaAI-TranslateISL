import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
import sqlite3
import hashlib
import os
from PIL import Image

# Include TailwindCSS for styling
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
        .custom-container {
            max-width: 1200px;
            margin: auto;
            padding: 20px;
        }
        .upload-button {
            display: block;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
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

def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(26, activation='softmax')(x)  # 26 for A-Z signs
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((64, 64))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def train_model():
    # Implement training data loading and preprocessing
    model = create_model()
    # X_train, y_train = ...
    # model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    model.save('sign_language_model.h5')

def test_model():
    model = load_model('sign_language_model.h5')
    # Implement test data loading and preprocessing
    # X_test, y_test = ...
    # loss, accuracy = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {loss}")
    st.write(f"Test Accuracy: {accuracy}")

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=-1)

def load_or_create_model():
    if os.path.exists('sign_language_model.h5'):
        return load_model('sign_language_model.h5')
    else:
        model = create_model()
        model.save('sign_language_model.h5')  # Save the default model
        return model

def main():
    st.title("TECADE Sign Language Interpreter")

    menu = ["Home", "Login", "Register", "Train", "Test"]
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
            prediction = model.predict(np.expand_dims(processed_frame, axis=0))
            predicted_letter = chr(np.argmax(prediction) + 65)  # Convert to A-Z

            frame = cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_placeholder.image(frame, channels="BGR")

    elif choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if authenticate_user(username, password):
                st.success("Logged in successfully")
                st.session_state.logged_in = True
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
        st.subheader("Test Model")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_letter = chr(np.argmax(prediction) + 65)  # Convert to A-Z
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write(f"Predicted letter: {predicted_letter}")

if __name__ == "__main__":
    init_db()
    main()
