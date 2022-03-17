"""
Created on Saturday 12th

@author: Erona Aliu
"""

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import preprocessing

st.header("CNN Classifier - Intel Image Classifier")


def main():
    file_uploaded = st.sidebar.file_uploader(
        "Choose a file: ", type=['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:  # ensure the image is not null
        image = Image.open(file_uploaded)
        fig = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(fig)
    else:
        st.write("Please upload an image of one of the following:")
        st.markdown(
            """
                * Buildings
                * Forest
                * Glacier
                * Mountain
                * Sea
                * Street
            """
        )


def predict_class(image):

    classifier_model = load_model(
        'C:\\Users\\erona\\OneDrive\\Desktop\\fyp_GIT\\FYP\\sample_projects\\project_1-intel_image_classification\\model_checkpoint\\cnn_classification.hdf5'
    )

    model = Sequential([
        hub.KerasLayer(classifier_model,
                       input_shape=(150, 150, 3))
    ])

    test_img = image.resize((150, 150))  # ensure the image is the size we want
    test_img = preprocessing.image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)

    class_names = ['buildings',
                   'forest',
                   'glacier',
                   'mountain',
                   'sea',
                   'street']

    predictions = model.predict(test_img)

    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    image_class = class_names[np.argmax(scores)]  # get the max from the scores

    result = 'Image uploaded: {}'.format(image_class)
    return result


if __name__ == "__main__":
    main()
