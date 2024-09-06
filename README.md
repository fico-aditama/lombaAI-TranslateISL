# TranslateISL Sign Language Interpreter

## Description
The TranslateISL Sign Language Interpreter is a real-time sign language recognition system that utilizes a trained machine learning model to translate sign language gestures into text. The system leverages a webcam for live video capture and provides functionality for user login, registration, model training, and testing.

## Directory Structure
```
├── data/             # Contains labels and images
├── models/           # Contains trained models
├── notebooks/        # Contains Jupyter notebooks
├── src/              # Contains source code
├── README.md         # Project description and setup instructions
└── requirements.txt  # List of dependencies
```

## Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/fico-aditama/lombaAI-TranslateISL.git TranslateISL
   cd TranslateISL
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run src/main.py
   ```

## Usage
1. **Home**: Displays the live video feed and predicts sign language gestures in real-time.
2. **Login**: Allows users to log in using their credentials.
3. **Register**: Enables users to create a new account.
4. **Train**: Trains the model using the provided dataset and saves it.
5. **Test**: Evaluates the model’s performance on the test set.
6. **Upload Image**: Allows image uploads and predicts sign language gestures.

## Tools
- **OpenCV**: For real-time video capture.
- **TensorFlow**: For building and training machine learning models.
- **Mediapipe**: For hand landmark detection.
- **SQLite**: For user authentication and storage.

## Contributing
Please open an issue or submit a pull request to contribute to this project.
