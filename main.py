#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
import tensorflow as tf


# In[2]:


from tensorflow.keras.applications.inception_v3 import preprocess_input
from flask import Flask, redirect, url_for, request, render_template
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from werkzeug.utils import secure_filename

import numpy as np
import glob
import sys
import os
import re


# In[3]:


# Define a flask app
app = Flask(__name__)


# In[4]:


# Model saved with Keras model.save()
MODEL_PATH = "C://Users//admin//Desktop//Snakes//model_inception.h5"


# In[5]:


# Load your trained model
model = load_model(MODEL_PATH)


# In[7]:


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    
    ## Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Snake Is Cobra , Venomous"
    elif preds==1:
        preds="The Snake Is Krait , Venomous"
    elif preds==2:
        preds="The Snake Is Russels Viper , Venomous"
    elif preds==3:
        preds="The Snake Is Saw Scaled Viper, Venomous"
    else:
        preds="The Snake Is Not In India's Top 4 Most Venomous Snake"
    
    return preds


# In[8]:


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# In[9]:


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


# In[10]:


if __name__ == '__main__':
    app.run(port=5001,debug=True)

