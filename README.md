# **AICTE-Internship-P1-Image-Classification-by-Machine-Learning**  

## **Implementation-of-ML-Model-for-Image-Classification**  

This project is a **Streamlit application** designed for image classification using three powerful models: **MobileNetV2**, **CIFAR-10**, and **Xception**. Users can upload images and receive predictions with confidence scores from any of these models. Featuring a sleek navigation bar and real-time results, the app serves as an excellent tool for both educational and practical purposes.  

---

## **Key Features**  

### **1. Triple Model Support**  
- **MobileNetV2 (ImageNet)**:  
  Recognizes 1,000 diverse classes, including objects, animals, and vehicles.  
- **Custom CIFAR-10 Model**:  
  Specializes in classifying images into 10 categories such as airplanes, automobiles, and birds.  
- **Xception Model**:  
  A state-of-the-art deep learning model leveraging extreme inception modules for highly accurate classification.  

### **2. Intuitive Interface**  
- **Navigation Bar**:  
  Effortlessly switch between models using the sidebar menu.  
- **Real-Time Classification**:  
  Upload an image and receive immediate predictions along with confidence scores.  

### **3. Educational and Practical Use**  
- Gain insights into how different deep learning models perform.  
- Applicable for real-world scenarios requiring image classification.  

---

## **Getting Started**  

### **Prerequisites**  
- Python 3.7 or later.  
- A modern web browser.  

### **Installation**  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/GaganSeth07/AICTE-Internship-P1-Image-Classification-by-Machine-Learning.git
   cd Implementation-of-ML-model-for-image-classification
   ```  

2. **Set Up a Virtual Environment**:  
   ```bash
   python -m venv venv  
   source venv/bin/activate   # On Windows: `venv\Scripts\activate`  
   ```  

3. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt  
   ```  

4. **Download the Xception Model**:  
   Download the pre-trained **Xception model file** [here](https://drive.google.com/file/d/1RHlHSEG18rLw0ri-kR6IHoiuaN7m9l-F/view?usp=drive_link). Save the file as `xception_model.h5` in the project directory.  

5. **Run the Streamlit App**:  
   ```bash
   streamlit run app.py  
   ```  

6. **Access the App**:  
   Open your browser and go to: [http://localhost:8501](http://localhost:8501).  

---

## **Contributing**  
Contributions are welcome! Feel free to:  
- Fork the repository.  
- Open issues to report bugs or suggest features.  
- Submit pull requests to enhance functionality.  

---

## **Acknowledgements**  
- [Streamlit](https://streamlit.io/)  
- [TensorFlow](https://www.tensorflow.org/)  

--- 

Feel free to reach out with feedback or suggestions to further improve the app. 🚀  

