#!/usr/bin/env python
# coding: utf-8

# In[76]:


import cv2
import os
from screeninfo import get_monitors
from tensorflow.keras.models import load_model
import numpy as np


# In[77]:


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path is not None:
            img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE) # Reading image in Grayscale as the model expects...
            images.append(img)
    return images


# In[78]:


# Load Handwritten-Digit-Recogniiton model
try:
    model = load_model("trained/HDRM.h5")
except Exception as e:
    print(f"Error loading the model: {e}")


# In[79]:


# Relative path to video
img_path = os.path.join(os.getcwd(), "Resources", "Images") # Add the image file, Note: If using in anhy IDE change os.getcwd() to os.dirname(__file__)
if not os.path.exists(img_path):
    print("Folder not found!")
    exit()

images = load_images(img_path)
if not images:
    print(f"No images found at {img_path}")
    exit()

print(images)


# In[80]:


probabilities = []

# Image preprocessing
for img in images:

    """

    Converting the image to black & white using threaholding:
        -> If pixel_value >= 128, white
        -> If pixel_value < 128, black

    """
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    """

    Converting image backgroung to black and digit to white as the model expects the image in this form.

    """

    img = cv2.bitwise_not(img)

    # Reshape image
    img_size = (28, 28)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA) # Preserves edge integrity
    img  = img / 255.0 # Normalize
    img = img[..., None] # Adding one dimnesion for channel
    img = np.expand_dims(img, axis=0) # since we are passing only one image and model.predict expects an image with a batch_size we add a size of 1 at 0th index.

    probabilities.append(model.predict(img)) # Calculating probability for each digit


# In[81]:


print(probabilities)


# In[82]:


# Choosing the index of maximum probability from the column vector
results = []
for prob in probabilities:
    results.append(np.argmax(prob, axis=1))

print(results)


# In[83]:


# # For debugging purpose
# for img, result in zip(images, results):
#     # display_img = img * 255 # Back to original format
#     # display_img = display_img.squeeze() # Remove extra dimensions

#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     img = cv2.resize(img, (280, 280), interpolation = cv2.INTER_NEAREST) # Resizing for better viewing

#     """

#     img:	The image on which to draw the text
#     f"Predicted: {digit}":	The actual text string to draw (e.g., "Predicted: 5")
#     (10, 270):	Bottom-left corner of the text in the image → (x, y) coordinates
#     cv2.FONT_HERSHEY_SIMPLEX:	Font type (OpenCV provides several built-in fonts)
#     0.8:	Font scale (how large the text appears)
#     (255):	Color of the text — (255) = white (for grayscale image)
#     2:	Thickness of the text stroke

#     """

#     cv2.putText(img, f"Predicted {result}", (11, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0), 2)

#     cv2.imshow("Digit Prediction", img)
#     cv2.waitKey(0) # Press any key to continue to next images
#     cv2.destroyAllWindows()


# In[ ]:




