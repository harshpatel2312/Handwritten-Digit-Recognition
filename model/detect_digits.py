#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import os
from screeninfo import get_monitors
from tensorflow.keras.models import load_model


# In[12]:


# Load Handwritten-Digit-Recogniiton model
try:
    model = load_model("HDRM.h5")
except Exception as e:
    print(f"Error loading the model: {e}")


# In[13]:


# Relative path to video
img_path = os.path.join(os.getcwd(), "Resources", "Images", "Screenshot 2025-05-03 230105.png") # Add the image file
if not os.path.exists(img_path):
    print("File not found!")
    exit()


# In[14]:


img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Image read error!")


# In[15]:


"""

Converting the image to black & white using threaholding:
    -> If pixel_value >= 128, white
    -> If pixel_value < 128, black

"""
_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)


# In[16]:


"""

Converting image backgroung to black and digit to white as the model expects the image in this form.

"""

img = cv2.bitwise_not(img)


# In[17]:


import numpy as np
# Reshape image
img_size = (28, 28)
img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA) # Preserves edge integrity
img  = img / 255.0 # Normalize
img = img[..., None] # Adding one dimnesion at for channel
img = np.expand_dims(img, axis=0) # since we are passing one one image and model.predict expects an image with a batch_size we add a size of 1 at 0th index.

print(img.shape) # Verifying the image shape


# In[18]:


probabilities = model.predict(img) # Calculating the probabilties
print(probabilities)


# In[19]:


# Choosing the index of maximum probability from the column vector
result = np.argmax(probabilities, axis=1)
print(result)


# In[20]:


# For debugging purpose
display_img = img * 255 # Back to original format
display_img = display_img.squeeze() # Remove extra dimensions

display_img = cv2.resize(display_img, (280, 280), interpolation = cv2.INTER_NEAREST)

"""

display_img:	The image on which to draw the text
f"Predicted: {digit}":	The actual text string to draw (e.g., "Predicted: 5")
(10, 270):	Bottom-left corner of the text in the image → (x, y) coordinates
cv2.FONT_HERSHEY_SIMPLEX:	Font type (OpenCV provides several built-in fonts)
0.8:	Font scale (how large the text appears)
(255):	Color of the text — (255) = white (for grayscale image)
2:	Thickness of the text stroke

"""

cv2.putText(display_img, f"Predicted {result}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

cv2.imshow("Digit Prediction", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




