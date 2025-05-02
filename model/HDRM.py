from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


# Load dataset to train and test set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# example
plt.imshow(X_train[0], cmap="gray")
plt.title(f"{y_train[0]}")
plt.show()

# Check for the shape of the data
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

print(X_train[0])

"""

-> Since the pixel values are in range of 0-255 which can make model bias while training.
-> To overcome this normalizing these values is a better option

"""

# Divide by the highest number value to get a range of 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Since the image is in grayscale ( Width * height ) and CNN requires a 3d array ( Width * height * channels )
# we are going to convert the matrix to tensor to include the channel for the image.
X_train = X_train[..., None]
X_test = X_test[..., None]

"""

Sequential: Stack of layers added on top of each other

First Layer (Base Layer):
    -> Conv2D: 
        - Outputs 32 feature maps
        - Each filter is of size (3x3)
        - ReLU for non-linearity
        - input shape (28, 28, 1)
    
    -> MaxPooling2D:
        - Basically selects every 2x2 matrix in the each feature map and returns the max value.

Second Layer (Finds more information):
    -> Conv2D:
        - Outputs 64 feature maps (Increased as deeper layers can get more complex patterns then the upper layers)
        - Each filter is of size (3x3)
        - ReLU for non-linearity
        - No input shape as it already has it in above layers
        
    -> MaxPooling2D:
        - Basically selects every 2x2 matrix in the each feature map and returns the max value.

We Flatten before entering deep layers as they reuqire vector shape array rather than 3d array.

Third Layer (Deep learning begins here):
    -> Dense:
        - 128 neurons

Fourth Layer (Output Layer): 
    -> Dense:
        - 10 neurons, each indicating 0-9 digits
        - softmax, as it counts probabilties and the highest one is selected representing the digit.
        
"""

cnn = Sequential([
    Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)), 
    MaxPooling2D((2, 2)), 
    Conv2D(64, (3, 3), activation = "relu"),
    MaxPooling2D((2, 2)), 
    Flatten(),
    Dense(128, activation = "relu"),
    Dense(10, activation = "softmax")
])

"""
What happens during each epoch:
    - Forward pass:
        The model predicts outputs for all training samples.
    
    - Loss computation:
        It compares predictions with actual labels using the loss function.
    
    - Backward pass (Backpropagation):    
        It calculates how much each weight contributed to the error.

    - Weight update (via optimizer):
        The optimizer (e.g., Adam) adjusts the weights to reduce the loss.

"""
cnn.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Selects the best model and saves it during training
checkpoint = ModelCheckpoint(
    "HDRM.h5", # Model name
    monitor = "val_loss", # Monitor for validation loss improvements
    save_best_only=True,
    mode = "min", # Look out for minimum losses
    verbose = 1 # Display callbacks
)

# Make 5 iterations over the dataset to learn complex patterns
# Leave 10% data for validation
cnn.fit(X_train, y_train, epochs=5, validation_split=0.1, callbacks=[checkpoint])

model = load_model("HDRM.h5") # Load best saved model

# Test Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Predict

"""

-> Forms a batch (1, 28, 28, 1) for each image automatically, as model.predict() expects a batch
-> np.argmax will get the index of maximum probability in the 2d array
-> We use axis = 1 as the 2D array will have rows and columns where each row is one image and columns represent the 10 probs.

"""

y_pred_probs = model.predict(X_test) # Shape: (10000, 10) as there 10000 images and each image will get a vector of 10 probs.
y_pred = np.argmax(y_pred_probs, axis = 1)

print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = range(10))

display_cm.plot(cmap="Blues", xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.show()
