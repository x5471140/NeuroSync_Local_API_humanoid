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

def linear_blend(previous_chunk, current_chunk, blend_frames):
    # Create extra frames at the end of the previous_chunk
    extended_chunk = np.copy(previous_chunk[-blend_frames:])
    
    # Blend extended_chunk with the start of current_chunk
    for i in range(blend_frames):
        alpha = i / blend_frames
        current_chunk[i] = (1 - alpha) * extended_chunk[i] + alpha * current_chunk[i]
    
    return current_chunk

def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs

def process_audio_features(audio_features, model, device, config):
    if audio_features.shape[0] < 3:
        raise ValueError("Insufficient frames in audio features. At least 3 frames are required.")

    audio_features = audio_features[1:-1, :]  # Remove first and last frame

    frame_length = config['frame_size']  
    num_features = audio_features.shape[1]
    num_frames = audio_features.shape[0]
    all_decoded_outputs = []

    blend_frames = 5  # Number of frames to blend
    previous_chunk = None

    model.eval()

    for start_idx in range(0, num_frames, frame_length):
        end_idx = min(start_idx + frame_length, num_frames)
        audio_chunk = audio_features[start_idx:end_idx]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)

        decoded_outputs = decode_audio_chunk(audio_chunk, model, device)

        if previous_chunk is not None:
            # Blend between previous chunk and current chunk
            decoded_outputs = linear_blend(previous_chunk, decoded_outputs, blend_frames)

        all_decoded_outputs.append(decoded_outputs[:end_idx - start_idx])

        # Update the previous_chunk with the last few frames for the next blending
        previous_chunk = decoded_outputs

    final_decoded_outputs = concatenate_outputs(all_decoded_outputs, num_frames)
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    final_decoded_outputs[:, :61] /= 100  # Normalize first 61 features

    return final_decoded_outputs
