# AICTE-Internship-P1-Image-Classification-by-Machine-Learning

# Implementation-of-ML-model-for-image-classification

Implementation-of-ML-model-for-image-classification is a Streamlit app that integrates MobileNetV2, CIFAR-10, and Xception models for image classification. Users can upload images and receive predictions with confidence scores from any of these models. It features a sleek navigation bar for easy switching and real-time results, making it ideal for learning and practical use.

## Key Features

- **Triple Model Support**:
  - **MobileNetV2 (ImageNet)**: Recognizes 1,000 different classes from the ImageNet dataset, including everyday objects, animals, and vehicles.
  - **Custom CIFAR-10 Model**: Specializes in classifying images into one of ten specific categories such as airplanes, automobiles, and birds.
  - **Xception Model**: A deep learning model with extreme inception modules, offering high accuracy for image classification tasks across diverse datasets.

- **Intuitive Interface**:
  - **Navigation Bar**: Seamlessly switch between MobileNetV2, CIFAR-10, and Xception models using a sleek sidebar menu.
  - **Real-Time Classification**: Upload an image to receive immediate predictions with confidence scores.

- **Educational and Practical Use**:
  - Ideal for learning about deep learning models and their performance.
  - Useful for practical applications where image classification is needed.

## Getting Started

### Prerequisites

- Python 3.7 or later
- A web browser

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GaganSeth07/AICTE-Internship-P1-Image-Classification-by-Machine-Learning.git
   cd Implementation-of-ML-model-for-image-classification
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Xception Model**:
   The pre-trained Xception model file is available for download [here](https://drive.google.com/file/d/1RHlHSEG18rLw0ri-kR6IHoiuaN7m9l-F/view?usp=drive_link). Ensure you save the file as `xception_model.h5` in the project directory.

5. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

6. **Open the app**: 
   The app will open in your default web browser. If not, navigate to http://localhost:8501.

### Contributing

Feel free to fork the repository, open issues, or submit pull requests to contribute to the project.

### Acknowledgements

- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
