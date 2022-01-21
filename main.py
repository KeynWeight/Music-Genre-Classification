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


## Page expands to full width
st.set_page_config(page_title='Song Genre Classification',
                   layout='wide')

######################################
## Page Title and sub title

st.title("Song Genre Classification🎵")
st.subheader("In this app, you can upload a song clip of 30 seconds and the models will determine which genre it belongs to!")
st.write("**In total 10 genres are trained, which are rock, classical, metal, disco, blues, reggae, country, hiphop, jazz and pop**")
st.write("**Different song formats are accepted, e.g. wav, mp3**")
######################################



######################################
## Body

# @st.cache
def make_prediction(audio_path, model):
    features_array = song_feature_extraction(audio_path)
    y_pred = model.predict(features_array)

    return y_pred
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
    with st.spinner(text='Please wait, the model is predicting the genre of the song'):
        y_pred = make_prediction(audio_file_temp_path, loaded_model)


    st.write(f"The predicted song genre is {y_pred[0]}!")
    st.success('Prediction Successful!')


#Delete the temp directory
shutil.rmtree(tempdir)



# streamlit run /Users/gwunyim/Desktop/Music_genre_classification/main.py

