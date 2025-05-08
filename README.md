# 🧠 Handwritten Digit Detection from Images using Pretrained CNN

This repository provides a complete pipeline to detect and classify handwritten digits from image files using a pretrained Convolutional Neural Network (CNN) model. The model was trained on the [`mnist`](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data) dataset from `tensorfLow.keras.datasets.mnist` and saved as `HDRM.h5`. This script (`detect_digits.py`) loads the trained model, preprocesses input images, performs digit prediction, and optionally visualizes the results with OpenCV.

> 📂 For training details, model architecture, and more, please refer to the [`train_model`](https://github.com/harshpatel2312/Handwritten-Digit-Recognition/tree/train_model) branch.

Whether you're looking to test a batch of digit images or integrate this into a larger application, this script allows you to do so with minimal setup.

> 📦 A ready-to-use pretrained model `HDRM.h5` is included, so you can start predicting digits without needing to retrain.




---

## File Structure
```bash
Train_Model
├── model/
|   ├── HDRM.py # Handwritten Digit Recognition code
|   ├── Resources/
|       ├── classification_report.png
|       ├── confusion_matrix.png
```

---

## 🔄 Clone the Repository & Switch to Branch
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

## 🧰 Model Highlights
- Built with TensorFlow and Keras
- Uses CNN with Conv2D, MaxPooling, and Dense layers
- Automatically saves the best model using `ModelCheckpoint`
- Evaluates performance using `accuracy`, `classification_report`, and `confusion_matrix`
- Easily customizable architecture for experimentation

---

## 🧠 Model Architecture
> Make sure to include `input_shape = (28, 28, 1)` as shown below, as CNN requires channel to be included alongside with the width and height. 
```text
Input: (28x28 grayscale image)

→ Conv2D(32 filters, 3x3) + ReLU  + input_shape = (28, 28, 1)
→ MaxPooling2D(2x2)  
→ Conv2D(64 filters, 3x3) + ReLU  
→ MaxPooling2D(2x2)  
→ Flatten  
→ Dense(128) + ReLU  
→ Dense(10) + Softmax 
```

## 📦 Customize
> If you would like to add more `layers/neurons` to the CNN, follow the steps
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),  # New conv layer
    Flatten(),
    Dense(256, activation="relu"),          # Increased neuron count
    Dropout(0.3),                            # Add dropout to reduce overfitting
    Dense(10, activation="softmax")
])
```

> If you would like to use other optimizer, follow the steps
```python
from tensorflow.keras.optimizers import SGD

cnn.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9), # update accordignly...
    loss="sparse_categorical_crossentropy", # update accordingly...
    metrics=["accuracy"]
)
```
### 📌 Note: Always match the loss function to your label format:
- Use sparse_categorical_crossentropy for integer labels (0–9)
- Use categorical_crossentropy for one-hot labels ([0,0,0,1,...])

---

## ▶️ Run the Training Script
```bash
cd model
python HDRM.py
```
> 📌 After running the script, the model will be automatically saved using ModelCheckpoint under the name HDRM.h5 (or any name you choose in the script). This will store the version of the model with the lowest validation loss during training.

---

## 📊 Evaluation Metrics
> 📌 Note:
The evaluation metrics listed here (accuracy, precision, recall, etc.) were obtained during one specific training run and may vary slightly depending on your system, Python version, random seed, or hardware (CPU/GPU). Feel free to retrain the model and observe your own results.

After training, the model is evaluated using:
- Test Accuracy: 99.12%
- ![Classification Report](model/classification_report.png)
- ![Confusion Matrix](model/confusion_matrix.png)

> 📌 These results may vary on your machine depending on training runs.

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
