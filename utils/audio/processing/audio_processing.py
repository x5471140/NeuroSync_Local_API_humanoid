# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# audio_processing.py

import numpy as np
import torch

def pad_audio_chunk(audio_chunk, frame_length, num_features):
    if audio_chunk.shape[0] < frame_length:
        pad_length = frame_length - audio_chunk.shape[0]
        padding = np.pad(
            audio_chunk,
            pad_width=((0, pad_length), (0, 0)),
            mode='reflect'
        )
        audio_chunk = np.vstack((audio_chunk, padding[-pad_length:, :num_features]))
    return audio_chunk

def decode_audio_chunk(audio_chunk, model, device):
    src_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)
        output_sequence = model.decoder(encoder_outputs)
        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
    return decoded_outputs

def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs

def process_audio_features(audio_features, model, device, config):
    # Ensure there are at least 3 frames to remove the first and last safely
    if audio_features.shape[0] < 3:
        raise ValueError("Insufficient frames in audio features. At least 3 frames are required.")

    # Remove the first and last frame
    audio_features = audio_features[1:-1, :]

    frame_length = config['frame_size']  
    num_features = audio_features.shape[1]
    num_frames = audio_features.shape[0]
    all_decoded_outputs = []

    model.eval()

    for start_idx in range(0, num_frames, frame_length):
        end_idx = min(start_idx + frame_length, num_frames)
        audio_chunk = audio_features[start_idx:end_idx]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device)
        all_decoded_outputs.append(decoded_outputs[:end_idx - start_idx])

    final_decoded_outputs = concatenate_outputs(all_decoded_outputs, num_frames)
    
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    final_decoded_outputs[:, :61] /= 100  

    return final_decoded_outputs
