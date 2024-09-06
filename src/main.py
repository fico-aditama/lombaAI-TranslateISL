import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import sqlite3
import hashlib
import os
from PIL import Image
import mediapipe as mp
import requests
import zipfile
import io

# Initialize MediaPipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

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
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')  # 26 for A-Z signs
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def download_and_extract_dummy_data():
    url = "https://example.com/dummy-data.zip"  # Replace with the actual URL
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall('data')

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def upload_and_predict(model):
    uploaded_files = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            preprocessed_image = preprocess_image(image)
            
            # Predict
            prediction = model.predict(preprocessed_image)
            predicted_letter = chr(np.argmax(prediction) + 65)  # Convert to A-Z
            
            st.image(image, caption=f"Uploaded Image")
            st.write(f"Predicted Letter: {predicted_letter}")

def train_model():
    if not os.path.exists('data'):
        os.makedirs('data')
        download_and_extract_dummy_data()

    # Load images from directory
    X_train, y_train = [], []
    for i in range(26):  # A-Z
        letter_dir = f"data/{chr(i + 65)}"
        if not os.path.exists(letter_dir):
            continue
        for img_file in os.listdir(letter_dir):
            img_path = os.path.join(letter_dir, img_file)
            img = Image.open(img_path)
            img = preprocess_image(img)
            X_train.append(img)
            y_train.append(i)

    X_train = np.vstack(X_train)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)

    model = create_model()
    model.fit(X_train, y_train, epochs=5, validation_split=0.2)
    model.save('models/sign_language_model.h5')
    st.success("Model trained successfully with data from the folder")

def test_model():
    if not os.path.exists('data'):
        os.makedirs('data')
        download_and_extract_dummy_data()

    # Load images from directory
    X_test, y_test = [], []
    for i in range(26):  # A-Z
        letter_dir = f"data/{chr(i + 65)}"
        if not os.path.exists(letter_dir):
            continue
        for img_file in os.listdir(letter_dir):
            img_path = os.path.join(letter_dir, img_file)
            img = Image.open(img_path)
            img = preprocess_image(img)
            X_test.append(img)
            y_test.append(i)

    X_test = np.vstack(X_test)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=26)

    model = tf.keras.models.load_model('models/sign_language_model.h5')
    loss, accuracy = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {loss}")
    st.write(f"Test Accuracy: {accuracy}")

def load_or_create_model():
    if os.path.exists('models/sign_language_model.h5'):
        return tf.keras.models.load_model('models/sign_language_model.h5')
    else:
        model = create_model()
        model.save('models/sign_language_model.h5')
        return model

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def main():
    st.title("TECADE Sign Language Interpreter")

    menu = ["Home", "Login", "Register", "Train", "Test", "Upload Image"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice in ["Train", "Test", "Upload Image", "Home"]:
        if 'authenticated' not in st.session_state or not st.session_state.authenticated:
            st.warning("Please log in to access this feature.")
            return

    model = load_or_create_model()

    if choice == "Home":
        st.write("Welcome to the TECADE Sign Language Interpreter app.")
        st.subheader("Live Video Feed")

        # Add start and stop buttons for video feed
        start_video = st.button("Start Video")
        stop_video = st.button("Stop Video")

        if start_video:
            st.write("Starting video feed...")
            video_placeholder = st.empty()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open video capture.")
                return

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video feed ended or not available.")
                    break

                # Process frame with MediaPipe Hands
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        for landmark in hand_landmarks.landmark:
                            h, w, _ = frame.shape
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    
                    # Use pre-trained model for gesture prediction
                    processed_frame = preprocess_frame(frame)
                    prediction = model.predict(processed_frame)
                    predicted_letter = chr(np.argmax(prediction) + 65)  # Convert to A-Z
                    
                    # Display prediction on the frame
                    cv2.putText(frame, f"Predicted: {predicted_letter}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display video feed
                video_placeholder.image(frame, channels="BGR")

                if stop_video:
                    st.write("Stopping video feed...")
                    cap.release()
                    cv2.destroyAllWindows()
                    break

    elif choice == "Upload Image":
        st.subheader("Upload Image")
        upload_and_predict(model)

    elif choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
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
        st.subheader("Train Model")
        if st.button("Train Model"):
            train_model()

    elif choice == "Test":
        st.subheader("Test Model")
        if st.button("Test Model"):
            test_model()

if __name__ == "__main__":
    init_db()
    main()
