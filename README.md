# Music-Genre-Classification

This is a repository of a streamlit web app. The web app is to classify the song genre of the song using machine learning


## Background
This web app uses XGBoost and a deep learning model to build the classification models. 

1. The `music-genre-classification-30-secs-81-accuracy.ipynb` notebook shows the building of the XGBoost model. This notebook is a copy of the work I did on Kaggle. It has achieved 81% accuracy. This model is trained on GZTAN Dataset, which 1000 songs of 10 genres are trained in the model, which are rock, classical, metal, disco, blues, reggae, country, hiphop, jazz and pop.


2. The `music-genre-classification-25-secs-73-accuracy.ipynb` notebook shows the building of the deep learning model using Fastai. This model is trained on FMA Dataset, which 25000 songs of 16 genres are trained in the model, which are Blues, Classical, Country, Easy Listening, Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Jazz, Old-Time / Historic, Pop, Rock, Soul-RnB, Spoken. It has achieved 73% accuracy. For further information, please visit this repo
https://github.com/mdeff/fma




## Installation

Please install the dependencies in the `requirements.txt`



## Usage
1. To run the app on local host, please run
```bash
streamlit run main.py
```


2. Or you may check out the deployed version on web
https://music-genre-classification-pgtv.onrender.com

