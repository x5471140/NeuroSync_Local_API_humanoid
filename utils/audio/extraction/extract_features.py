# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# extract_features.py
import io
import librosa
import numpy as np

def extract_audio_features(audio_input, sr=88200, from_bytes=False):
    if from_bytes:
        y, sr = load_audio_from_bytes(audio_input, sr)
    else:
        y, sr = load_and_preprocess_audio(audio_input, sr)
    
    frame_length = int(0.01667 * sr)  
    hop_length = frame_length // 2  
    min_frames = 9  

    num_frames = (len(y) - frame_length) // hop_length + 1

    if num_frames < min_frames:
        print(f"Audio file is too short: {num_frames} frames, required: {min_frames} frames")
        return None, None

    all_features = []

    mfcc_features = extract_mfcc_features(y, sr, frame_length, hop_length)
    all_features.append(mfcc_features)
        
    combined_features = np.hstack(all_features)
    
    return combined_features, y

def extract_mfcc_features(y, sr, frame_length, hop_length, num_mfcc=23):
    mfcc_features = extract_overlapping_mfcc(y, sr, num_mfcc, frame_length, hop_length)
    reduced_mfcc_features = reduce_features(mfcc_features)
    return reduced_mfcc_features.T

def cepstral_mean_variance_normalization(mfcc):
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    return (mfcc - mean) / (std + 1e-10)

def extract_overlapping_mfcc(chunk, sr, num_mfcc, frame_length, hop_length):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=num_mfcc + 1, n_fft=frame_length, hop_length=hop_length)
    mfcc = cepstral_mean_variance_normalization(mfcc)
    mfcc = mfcc[1:] 
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    combined_mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return combined_mfcc    

def reduce_features(features):
    num_frames = features.shape[1]
    paired_frames = features[:, :num_frames // 2 * 2].reshape(features.shape[0], -1, 2)
    reduced_frames = paired_frames.mean(axis=2)
    
    if num_frames % 2 == 1:
        last_frame = features[:, -1].reshape(-1, 1)
        reduced_final_features = np.hstack((reduced_frames, last_frame))
    else:
        reduced_final_features = reduced_frames
    
    return reduced_final_features



def load_and_preprocess_audio(audio_path, sr=88200):
    y, sr = load_audio(audio_path, sr)
    if sr != 88200:
        y = librosa.resample(y, orig_sr=sr, target_sr=88200)
        sr = 88200
    
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr

def load_audio(audio_path, sr=88200):
    y, sr = librosa.load(audio_path, sr=sr)
    print(f"Loaded audio file '{audio_path}' with sample rate {sr}")
    return y, sr

def load_audio_from_bytes(audio_bytes, sr=88200):
    audio_file = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_file, sr=sr)
    
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr

def load_audio_file_from_memory(audio_bytes, sr=88200):
    """Load audio from memory bytes."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    print(f"Loaded audio data with sample rate {sr}")
    
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr
