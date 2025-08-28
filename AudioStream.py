import librosa
import numpy as np
import scipy
import torch

class AudioStream(object):
    def __init__(self, id, sample_rate, output_dim):
        self.blend_time = 0.5
        self.sample_width = 2
        self.id = id
        self.is_over = False
        self.sample_rate = sample_rate
        self.output_dim = output_dim
        self.target_sample_rate = 88200
        self.audio_bytes = b''
        self.cnt_audio_bytes_index = 0
        self.y = np.zeros(0, dtype=np.float32)  # 重采样到88200Hz的音频数据
        self.cnt_y_index = 0 # 当前处理到的位置
        self.combined_features = np.zeros((0, 256), dtype=np.float32)
        self.cnt_combined_features_index = 0
        self.decoded_output_list = []

    def __audio_bytes_to_pcm(self):
        # TODO: 可能此时客户端上传的音频数据量不够, 导致动作跳变
        # 待实验方案:
        # 1. 后续补0

        min_audio_bytes = self.sample_rate * self.sample_width * 1

        if self.is_over:
            # 客户端已经上传结束, 但是长度不够的情况下, 需要在结尾补0
            if len(self.audio_bytes) - self.cnt_audio_bytes_index < min_audio_bytes:
                self.audio_bytes += b'\x00' * (min_audio_bytes - (len(self.audio_bytes) - self.cnt_audio_bytes_index))

        if len(self.audio_bytes) - self.cnt_audio_bytes_index >= min_audio_bytes:
            sr = self.sample_rate
            max_value = 32768.0
            audio_bytes = self.audio_bytes[self.cnt_audio_bytes_index:]
            self.cnt_audio_bytes_index = len(self.audio_bytes)
            data = np.frombuffer(audio_bytes, dtype=np.int16)
            y = data.astype(np.float32) / max_value
            target_sr = self.target_sample_rate
            if sr != target_sr:
                num_samples = int(len(y) * target_sr / sr)
                y_resampled = scipy.signal.resample(y, num_samples)
                y = y_resampled
                sr = target_sr
            self.y = np.concatenate((self.y, y))
            return True
        return False

    def cepstral_mean_variance_normalization(self, mfcc):
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        return (mfcc - mean) / (std + 1e-10)
    
    def extract_overlapping_mfcc(self, chunk, sr, num_mfcc, frame_length, hop_length, include_deltas=True, include_cepstral=True, threshold=1e-5, is_padding=False):
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=num_mfcc, n_fft=frame_length, hop_length=hop_length, center=not is_padding)
        if include_cepstral:
            mfcc = self.cepstral_mean_variance_normalization(mfcc)

        if include_deltas:
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            combined_mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])  # Stack original MFCCs with deltas
            return combined_mfcc
        else:
            return mfcc
        
    def reduce_features(self, features):
        num_frames = features.shape[1]
        paired_frames = features[:, :num_frames // 2 * 2].reshape(features.shape[0], -1, 2)
        reduced_frames = paired_frames.mean(axis=2)
        
        if num_frames % 2 == 1:
            last_frame = features[:, -1].reshape(-1, 1)
            reduced_final_features = np.hstack((reduced_frames, last_frame))
        else:
            reduced_final_features = reduced_frames
        
        return reduced_final_features

    def fix_edge_frames_autocorr(self, autocorr_features, zero_threshold=1e-7):
        """If the first or last frame is near all-zero, replicate from adjacent frames."""
        # Check first frame energy
        if np.all(np.abs(autocorr_features[:, 0]) < zero_threshold):
            autocorr_features[:, 0] = autocorr_features[:, 1]
        # Check last frame energy
        if np.all(np.abs(autocorr_features[:, -1]) < zero_threshold):
            autocorr_features[:, -1] = autocorr_features[:, -2]
        return autocorr_features
    def extract_overlapping_autocorr(self, y, sr, frame_length, hop_length, num_autocorr_coeff=187, pad_signal=True, padding_mode="reflect", trim_padded=False):
        if pad_signal:
            pad = frame_length // 2
            y_padded = np.pad(y, pad_width=pad, mode=padding_mode)
        else:
            y_padded = y

        frames = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length)
        if pad_signal and trim_padded:
            num_frames = frames.shape[1]
            start_indices = np.arange(num_frames) * hop_length
            valid_idx = np.where((start_indices >= pad) & (start_indices + frame_length <= len(y) + pad))[0]
            frames = frames[:, valid_idx]

        frames = frames - np.mean(frames, axis=0, keepdims=True)
        hann_window = np.hanning(frame_length)
        windowed_frames = frames * hann_window[:, np.newaxis]

        autocorr_list = []
        for frame in windowed_frames.T:
            full_corr = np.correlate(frame, frame, mode='full')
            mid = frame_length - 1  # Zero-lag index.
            # Extract `num_autocorr_coeff + 1` to include the first column initially
            wanted = full_corr[mid: mid + num_autocorr_coeff + 1]
            # Normalize by the zero-lag (energy) if nonzero.
            if wanted[0] != 0:
                wanted = wanted / wanted[0]
            autocorr_list.append(wanted)

        # Convert list to array and transpose so that shape is (num_autocorr_coeff + 1, num_valid_frames)
        autocorr_features = np.array(autocorr_list).T
        # Remove the first coefficient to avoid redundancy
        autocorr_features = autocorr_features[1:, :]

        autocorr_features = self.fix_edge_frames_autocorr(autocorr_features)
                                        
        return autocorr_features
    
    def compute_autocorr_with_deltas(self, autocorr_base):
        delta_ac = librosa.feature.delta(autocorr_base)
        delta2_ac = librosa.feature.delta(autocorr_base, order=2)
        combined_autocorr = np.vstack([autocorr_base, delta_ac, delta2_ac])
        return combined_autocorr

    def extract_autocorrelation_features(self,
        y, sr, frame_length, hop_length, include_deltas=False, is_padding=False
    ):
        """
        Extract autocorrelation features, optionally with deltas/delta-deltas,
        then align with the MFCC frame count, reduce, and handle first/last frames.
        """
        autocorr_features = self.extract_overlapping_autocorr(
            y, sr, frame_length, hop_length, pad_signal=not is_padding
        )
        
        if include_deltas:
            autocorr_features = self.compute_autocorr_with_deltas(autocorr_features)

        autocorr_features_reduced = self.reduce_features(autocorr_features)

        return autocorr_features_reduced.T
    
    def extract_and_combine_features(self, y, sr, frame_length, hop_length, include_autocorr=True, is_padding=False):
        all_features = []

        overlapping_mfcc = self.extract_overlapping_mfcc(y, sr, 23, frame_length, hop_length, is_padding=is_padding)
        mfcc_features = self.reduce_features(overlapping_mfcc)
        mfcc_features = mfcc_features.T

        #mfcc_features: np.ndarray = self.extract_mfcc_features(y, sr, frame_length, hop_length)
        all_features.append(mfcc_features)
        if include_autocorr:
            autocorr_features = self.extract_autocorrelation_features(
                y, sr, frame_length, hop_length, is_padding=is_padding
            )
            all_features.append(autocorr_features)
        
        combined_features = np.hstack(all_features)

        return combined_features

    def extract_features(self):
        
        
        remaining_length = self.y.shape[0] - self.cnt_y_index

        sr = self.target_sample_rate
        frame_length = int(0.01667 * sr)
        hop_length = frame_length // 2
        y = self.y[self.cnt_y_index:]
        # 提取MFCC特征的时候不要mfcc自己在前后填充0, 自己补足
        padding_length = frame_length // 2
        if self.cnt_y_index >= padding_length:
            y = np.concatenate([self.y[self.cnt_y_index - padding_length:self.cnt_y_index], y])
        else:
            y = np.concatenate([np.zeros(padding_length, dtype=np.float32), y])
        
        # real_length = remaining_length - padding_length

        if len(y) < 9:
            print(f"Audio file is too short: {len(y)} samples, required: 9 samples")
        
        combined_features = self.extract_and_combine_features(y, sr, frame_length, hop_length, is_padding=True)
        self.cnt_y_index = self.cnt_y_index + remaining_length
        return combined_features
        

    def pad_audio_chunk(self, audio_chunk, frame_length, num_features, pad_mode='replicate'):
        if audio_chunk.shape[0] < frame_length:
            pad_length = frame_length - audio_chunk.shape[0]
            
            if pad_mode == 'reflect':
                padding = np.pad(
                    audio_chunk,
                    pad_width=((0, pad_length), (0, 0)),
                    mode='reflect'
                )
                audio_chunk = np.vstack((audio_chunk, padding[-pad_length:, :num_features]))
            
            elif pad_mode == 'replicate':
                last_frame = audio_chunk[-1:]  
                replication = np.tile(last_frame, (pad_length, 1)) 
                audio_chunk = np.vstack((audio_chunk, replication))
            
            else:
                raise ValueError(f"Unsupported pad_mode: {pad_mode}. Choose 'reflect' or 'replicate'.")
        
        return audio_chunk
    
    def smooth(self, decoded_outputs):
        return scipy.ndimage.gaussian_filter1d(decoded_outputs, sigma=2, axis=0)

    def decode_audio_chunk(self, audio_chunk, model, device, config):
        use_half_precision = config.get("use_half_precision", True)
        dtype = torch.float16 if use_half_precision else torch.float32
        src_tensor = torch.tensor(audio_chunk, dtype=dtype).unsqueeze(0).to(device)

        with torch.no_grad():
            if use_half_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    encoder_outputs = model.encoder(src_tensor)
                    output_sequence = model.decoder(encoder_outputs)
            else:
                encoder_outputs = model.encoder(src_tensor)
                output_sequence = model.decoder(encoder_outputs)

            decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
        return decoded_outputs
    
    def blend_chunks(self, chunk1, chunk2, overlap):
        actual_overlap = min(overlap, len(chunk1), len(chunk2))
        if actual_overlap == 0:
            return np.vstack((chunk1, chunk2))
        
        blended_chunk = np.copy(chunk1)
        for i in range(actual_overlap):
            alpha = i / actual_overlap 
            blended_chunk[-actual_overlap + i] = (1 - alpha) * chunk1[-actual_overlap + i] + alpha * chunk2[i]
            
        return np.vstack((blended_chunk, chunk2[actual_overlap:]))
    
    def process_audio_features_old(self, audio_features_segment, bProcessAll: bool, model, device, config):
        frame_length = config['frame_size'] # 默认128
        overlap = config['overlap'] # 默认32
        num_features = audio_features_segment.shape[1] # 256 mfcc features
        num_frames = audio_features_segment.shape[0] # 1223
        all_decoded_outputs = []
        model.eval()
        start_idx = 0
        while start_idx < num_frames:
            end_idx = min(start_idx + frame_length, num_frames)
            audio_chunk = audio_features_segment[start_idx:end_idx]
            audio_chunk = self.pad_audio_chunk(audio_chunk, frame_length, num_features)
            decoded_outputs = self.decode_audio_chunk(audio_chunk, model, device, config)
            decoded_outputs = decoded_outputs[:end_idx - start_idx]
            if all_decoded_outputs:
                last_chunk = all_decoded_outputs.pop()
                blended_chunk = self.blend_chunks(last_chunk, decoded_outputs, overlap)
                all_decoded_outputs.append(blended_chunk)
            else:
                all_decoded_outputs.append(decoded_outputs)
            start_idx += frame_length - overlap

        if all_decoded_outputs:
            final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)[:num_frames]
            final_decoded_outputs[:, :] /= 100
        else:
            final_decoded_outputs = np.zeros((0, self.output_dim), dtype=np.float32)
        return final_decoded_outputs

    def process_audio_features(self, model, device, config):
        frame_length = config['frame_size'] # 默认128
        overlap = config['overlap'] # 默认32
        num_features = self.combined_features.shape[1] # 256 mfcc features
        num_frames = self.combined_features.shape[0] # 1223
        model.eval()

        #start_idx = self.cnt_combined_features_index
        while self.cnt_combined_features_index + frame_length <= num_frames:
            end_idx = self.cnt_combined_features_index + frame_length
            feature_chunk = self.combined_features[self.cnt_combined_features_index:end_idx]
            decoded_outputs = self.decode_audio_chunk(feature_chunk, model, device, config)
            decoded_outputs /= 100.0
            self.decoded_output_list.append(decoded_outputs)
            self.cnt_combined_features_index += frame_length - overlap

        if self.is_over:
            # 推流结束, 处理剩余数据
            print("推流结束, 处理剩余数据")
            if self.cnt_combined_features_index < num_frames:
                end_idx = num_frames
                feature_chunk = self.combined_features[self.cnt_combined_features_index:end_idx]
                self.pad_audio_chunk(feature_chunk, frame_length, num_features)
                decoded_outputs = self.decode_audio_chunk(feature_chunk, model, device, config)
                decoded_outputs /= 100.0
                self.decoded_output_list.append(decoded_outputs)
                self.cnt_combined_features_index = num_frames

        
    def write(self, audio_bytes: bytes):
        """
        1. 音频数据重采样到88200Hz, 来多少处理多少, 不用考虑和后续数据融合问题
        2. 音频数据提取mfcc特征
        3. MFCC特征->Muscle动作
        """
        self.audio_bytes += audio_bytes
        

    def process_by_frames(self, model, device, config):

        if config["overlap"] * 2 > config["frame_size"]:
            raise ValueError("overlap must be less than half of frame_size")

        pcm_ok = self.__audio_bytes_to_pcm()

        output_bytes = b''

        if pcm_ok:
            combined_features = self.extract_features()
            
            self.combined_features = np.vstack((self.combined_features, combined_features))

            self.process_audio_features(model, device, config)
            
            while len(self.decoded_output_list) >= 2:
                chunk1: np.ndarray = self.decoded_output_list[0]
                chunk2 = self.decoded_output_list[1]

                chunk1 = self.smooth(chunk1)
                chunk2 = self.smooth(chunk2)

                overlap = config['overlap']
                for i in range(overlap):
                    chunk1[-overlap + i] = (1 - i / overlap) * chunk1[-overlap + i] + (i / overlap) * chunk2[i]
                output_bytes += chunk1.tobytes()
                chunk2 = chunk2[overlap:]
                self.decoded_output_list[1] = chunk2
                self.decoded_output_list.pop(0)

            if self.is_over and len(self.decoded_output_list) == 1:
                output_bytes += self.decoded_output_list[0].tobytes()
        return output_bytes