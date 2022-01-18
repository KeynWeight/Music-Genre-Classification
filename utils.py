import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os



def normalize_features(X):
    #Load Standard Scaler from pickle

    local_dir = os.path.dirname(__file__)
    sc_name= 'sc.pkl'
    sc_path = os.path.join(local_dir, sc_name)
    
    with open(sc_path, "rb") as f:
        sc = pickle.load(f)
        X_scaled = sc.transform(X)

    return X_scaled




def song_feature_extraction(song_path, k=20):

    #Read path and get y and sr
    y, sr = librosa.load(song_path)
    
    #Extract features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.effects.harmonic(y)
    perceptr= librosa.effects.percussive(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=k)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr) 

    #Append those features in a list
    features_list = [chroma_stft, rmse, spectral_centroid,
                    spectral_bandwidth, rolloff, zcr, harmony, perceptr, tempo,mfcc,
                    chroma_cens, spectral_contrast, spectral_flatness, tonnetz]
    
    
    #Loop through each feature, find the mean and variance, and append to list
    test_list = []
    for feature in features_list:
        if feature is tempo:
            test_list.append(feature)
            continue
        mean = 0
        var = 0
        if feature is not mfcc:
            mean = np.mean(feature)
            var = np.var(feature)

            test_list.append(mean)
            test_list.append(var)
            
        else:
            for j in range(k):
                mean = np.mean(feature[j])
                var = np.var(feature[j])

                test_list.append(mean)
                test_list.append(var)
  
    #Convert to numpy array
    test_array = np.asarray([test_list])

    # Normalize the features
    test_array = normalize_features(test_array)

    # exclude 'mfcc6_mean', 'mfcc9_var', 'mfcc11_var', 'mfcc14_var', 'mfcc16_mean','mfcc16_var', 'mfcc17_var', 'mfcc19_mean','chroma_cens_var'
    dropped_features_indices = [27, 34, 38, 44, 47, 48, 50, 53, 58]
    test_array = np.delete(test_array, dropped_features_indices)
    test_array = np.reshape(test_array,(1,-1))

    return test_array
