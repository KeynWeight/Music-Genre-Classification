import pandas as pd
import librosa
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from utils import song_feature_extraction
import os
import tempfile
import shutil
from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
# import numpy as np

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#Create temp directory
tempdir = tempfile.mkdtemp()

#Define the path of the local directory for easy reference of files
local_dir = os.path.dirname(__file__)


## Page expands to full width
st.set_page_config(page_title='Song Genre Classification',
                   layout='wide')

######################################
## Page Title and sub title

st.title("Song Genre Classification🎵")
st.subheader("In this app, you can upload a song clip of 25/30 seconds and the models will determine which genre it belongs to!")
st.write("**There are two different models for you to choose. **")
st.write("**For model 1, you have 10 genres to predict from your 30s song clip, which are rock, classical, metal, disco, blues, reggae, country, hiphop, jazz and pop**")
st.write("**For model 2, you have 16 genres to predict from your 25s song clip, which are Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time / Historic, Pop, Rock, Soul-RnB, Spoken**")
######################################



######################################
#Load xgboost model
xgboost_model_name= 'model.pkl'
xgboost_model_path = os.path.join(local_dir, xgboost_model_name)
loaded_xgboost_model = pickle.load(open(xgboost_model_path,'rb'))

#Load deep learning model learner
dl_learner_name='learner.pkl'
dl_learner_path = os.path.join(local_dir, dl_learner_name)
loaded_learner = load_learner(dl_learner_path)

# The list of genres trained on the deep learning model 
dl_class_list = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
    'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
    'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']


#Helper functions to make prediction to be called after
def make_prediction(audio_path, model, learner, model_num):
    if model_num == 1:
        features_array = song_feature_extraction(audio_path)
        y_pred = model.predict(features_array)
        return y_pred[0]
    else:
        y_pred = learner.predict(audio_path)
        y_pred = dl_class_list[int(y_pred[0])]
        return y_pred

######################################




######################################
##Sidebar

# Pick the model
choose_model = st.sidebar.selectbox(
    "Choose the model",
    ("1. XGBoost-10 Genres",
     "2. Deep Learning-16 Genres" 
    )
)

######################################



########################################################################################################################################################
# Logic for choossing model 1


if choose_model == "1. XGBoost-10 Genres":
    model_num = 1


    ##############################################################################################################################
    ## Body

    st.subheader("You have chosen Model 1 - XGBoost model. Please upload your 30 seconds song clips and make prediction below.")

    ##############################################################################################################################


########################################################################################################################################################
#Logic for choosing model 2

else:
    model_num = 2

    ##############################################################################################################################
    ## Body
    st.subheader("You have chosen Model 2 - The deep learning model. Please upload your 25 seconds song clips and make prediction below.")




#Create a place to upload the audio file on Streamlit
audio_file = st.file_uploader('Please upload your song clip here', ['wav', 'mp3'] )

if audio_file:
    st.audio(audio_file)
    audio_file_temp_path = os.path.join(tempdir,audio_file.name)
    with open(audio_file_temp_path,"wb") as f:
        f.write(audio_file.getbuffer())

    pred_button = st.button('Predict Song Genre')
else:
    st.warning("Please upload a song.")
    st.stop()


# The behavior after clicking the prediction button
if pred_button:
    with st.spinner(text='Please wait, the model is predicting the genre of the song'):
        y_pred = make_prediction(audio_file_temp_path, loaded_xgboost_model, loaded_learner, model_num)


    st.write(f"The predicted song genre is {y_pred}!")
    st.success('Prediction Successful!')

    
    ##############################################################################################################################



#Delete the temp directory
shutil.rmtree(tempdir)



