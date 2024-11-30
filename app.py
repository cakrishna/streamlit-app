import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.header("Cats& Dogs Classification")
# Load Model
# model = tf.keras.models.load_model("model/cats_dogs_model.h5")
model = load_model("model/best_model_cats_dogs.hdf5")

# Divide page in Columns 

# st.header ("image Classification")
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Logo")
    st.image("cats-dogs.png",width=250)

with col2:
    st.subheader("Upload Image")
# load image from dir 
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'Catty',
            1: 'Dogyy'}

#Image Prediction
if uploaded_file is not None:
    imagename = uploaded_file
    # test_image = image.load_img(imagename, target_size = (150,150))
    test_image = image.load_img(imagename, target_size = (224,224))
    test_image = np.array(test_image)
    # test_image = np.expand_dims(test_image, axis = 0)
    # prediction = np.argmax(model.predict(test_image), axis=1) 

    prediction = model.predict(np.expand_dims(test_image/255, 0))
    prediction = np.argmax(prediction)
    # map_dict[prediction]
# Display image
    st.image(uploaded_file,  width=300, channels="RGB")

        
# Prediction Button
    Genrate_pred = st.button(":red_circle: Generate Prediction")    
        
    if Genrate_pred:
        st.subheader(":mag: Predicted Label for the image is : {}".format(map_dict[prediction]))
        # above method predicting better May be because of model.predict(np.expand_dims(test_image/255, 0))
        
        # st.header(":mag: Predicted Label for the image is : {}".format(map_dict[prediction[0]]))
        # st.header (":mag: Predicted Label for the image is : {}".format(prediction))
