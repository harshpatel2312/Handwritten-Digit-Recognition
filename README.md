# 🧠 Handwritten Digit Detection from Images using Pretrained CNN

This repository provides a complete pipeline to detect and classify handwritten digits from image files using a pretrained Convolutional Neural Network (CNN) model. The model was trained on the [`mnist`](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data) dataset from `tensorfLow.keras.datasets.mnist` and saved as `HDRM.h5`. 

This script (`detect_digits.py`) loads the trained model, preprocesses input images, performs digit prediction, and optionally visualizes the results with OpenCV. Whether you're looking to test a batch of digit images or integrate this into a larger application, this script allows you to do so with minimal setup.

> 📂 For training details, model architecture, and more, please refer to the [`train_model`](https://github.com/harshpatel2312/Handwritten-Digit-Recognition/tree/train_model) branch.

> 📦 A ready-to-use pretrained model `HDRM.h5` is included, so you can start predicting digits without needing to retrain.

---

## 🔍 How `detect_digits.py` Works ?

The `detect_digits.py` script loads the pretrained model (`HDRM.h5`) and processes all images from the `Resources/Images/` directory. Each image is:

1. Converted to grayscale
2. Thresholded and inverted to match MNIST digit format
3. Resized to 28×28 pixels
4. Normalized and reshaped to match the model's input shape
5. Passed through the CNN to get digit predictions

> 🖼️ For visual verification, you can uncomment the debugging section in the script to display each input image alongside its predicted digit. This helps confirm the model’s output and troubleshoot misclassifications if needed.

---

## File Structure
```bash
Train_Model
├── model/
|   ├── trained/
|       ├── HDRM.py # Handwritten Digit Recognition MModel
|   ├── Resources/ # Make this directory and sub directory if not present and insert the images of digits you wanna test
|       ├── Images/
|           ├── img_1.png # example
|   ├── detect_digits.py # Classifier
```

---

## 🔄 Clone the Repository
```bash
git clone https://github.com/harshpatel2312/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

---

## 🧰 Setting Up the Environment
1. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. Install required packages
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Training Script
```bash
cd model
python detect_digits.py
```

---

## 🚀 Real-World Applications
This handwritten digit recognition model has several practical applications, including:
- **Digitized Form Processing**: Automatically recognizing numbers from scanned handwritten forms (e.g., bank cheques, tax forms, census data).
- **Postal Code Reading**: Interpreting ZIP/postal codes on envelopes for mail sorting systems.
- **Banking Automation**: Recognizing account numbers or amounts on checks.
- **Educational Tools**: Assisting students or learners with handwritten number recognition practice.
- **Mobile Input**: Enhancing handwriting input methods for mobile apps and digital notepads.

> 🧠 This project serves as a foundation for more complex computer vision models used in document understanding, OCR, and intelligent automation systems.

---

## Dataset Citation Information
```bash
@article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  volume={2},
  year={2010}
}
```

---

## 📃 License
MIT License © Harsh Patel
