import numpy as np
import pandas as pd
import librosa
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from utils import song_feature_extraction
import os
import tempfile
import shutil

#Create temp directory
tempdir = tempfile.mkdtemp()

#load model from pickle
local_dir = os.path.dirname(__file__)
model_name= 'model.pkl'
model_path = os.path.join(local_dir, model_name)
loaded_model = pickle.load(open(model_path,'rb'))


#Set class name
# class_names = ['rock', 'classical', 'metal', 'disco', 'blues', 'reggae', 'country', 'hiphop', 'jazz', 'pop']


## Page expands to full width
st.set_page_config(page_title='Song Genre Classification',
                   layout='wide')

######################################
## Page Title and sub title

st.title("Song Genre Classification🎵")
st.subheader("In this app, you can upload a song clip of 30 seconds and the models will determine which genre it belongs to!")
st.write("**In total 10 genres are trained, including 'rock', 'classical', 'metal', 'disco', 'blues', 'reggae', 'country', 'hiphop', 'jazz', 'pop'**")
st.write("**Different song formats are accepted, e.g. wav, mp3**")
######################################



######################################
## Sidebar

@st.cache
def make_prediction(audio_file):
    pass
######################################


audio_file = st.file_uploader('Please upload your 30 Seconds song clip here', ['wav', 'mp3'] )

if audio_file:
    st.audio(audio_file)
    audio_file_temp_path = os.path.join(tempdir,audio_file.name)
    with open(audio_file_temp_path,"wb") as f:
        f.write(audio_file.getbuffer())

    pred_button = st.button('Predict Song Genre')
else:
    st.warning("Please upload a song.")
    st.stop()


if pred_button:

    features_array = song_feature_extraction(audio_file_temp_path)
    y_pred = loaded_model.predict(features_array)
    st.write(f"The predicted song genre is {y_pred}")

    

# choose_model = st.sidebar.selectbox(
#     "Pick model you'd like to use",
#     ("XGBoost", 
#      "Tensorflow Image classification", )
# )

#Delete the temp directory
shutil.rmtree(tempdir)



# # streamlit run /Users/gwunyim/Desktop/Music_genre_classification/main.py

