
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

from flask import Response, request, jsonify
import numpy as np
import torch
import flask

from AudioStream import AudioStream
from utils.audio.extraction.extract_features import extract_and_combine_features, extract_audio_features, extract_autocorrelation_features, extract_mfcc_features, load_pcm_audio_from_bytes
#from utils.generate_face_shapes import generate_facial_data_from_bytes, generate_muscle_data_from_bytes
from utils.audio.processing.audio_processing import ProcessAudioFeatures, blend_chunks, decode_audio_chunk, ensure_2d, pad_audio_chunk, zero_columns
from utils.model.model import load_model
from utils.config import face_config, body_config

from utils.model.model import Seq2Seq


app = flask.Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Activated device:", device)

body_model_path = 'utils/model/body.pth'
body_model: Seq2Seq = load_model(body_model_path, body_config, device)

face_model_path = 'utils/model/face.pth'
face_model: Seq2Seq = load_model(face_model_path, face_config, device)

# class AudioData:
#     """
#     自定义音频Block格式
#     int: dataId
#     int: sampleRate
#     int: sampleCount
#     byte[]: sampleBytes
#     """

#     sampleWidth = 2

#     def __init__(self, dataId: int, sampleRate: int):
#         self.dataId = dataId
#         self.sampleRate = sampleRate
#         self.sampleBytes = b''
#         self.cntFrame = 0
#         self.frontBlendFrame = 0
#         self.rearBlendFrame = 0
#         self.isOver = False
#         self.decodedOutputs = np.zeros((0, 87), dtype=np.float32)
#         self.y = np.zeros(0, dtype=np.float32)

#     def __frame2Bytes(self, frame: int):
#         return frame * self.sampleRate // 60 * self.sampleWidth

#     def GetFrameCount(self):
#         sampleCount = len(self.sampleBytes) // self.sampleWidth
#         return sampleCount * 60 // self.sampleRate
    
#     def GetFrameData(self, start: int, count: int):
#         s = self.__frame2Bytes(start)
#         e = self.__frame2Bytes(start + count)
#         return self.sampleBytes[s:e]
#     def AppendSample(self, sampleBytes: bytes):
#         self.sampleBytes += sampleBytes


audioDataDict: dict = {}


# def ProcessAudioBytes(postBytes: bytes) -> AudioData:
#     dataId: int = int.from_bytes(postBytes[:4], byteorder='little')
#     sampleRate: int = int.from_bytes(postBytes[4:8], byteorder='little')
#     ctrlCode: int = int.from_bytes(postBytes[8:12], byteorder='little')
#     sampleBytes: bytes = postBytes[12:]
    
#     audioData: AudioData = audioDataDict.get(dataId, None)

#     if (ctrlCode & 1) != 0 or audioData is None:
#         audioData = AudioData(dataId, sampleRate)
#         audioDataDict[dataId] = audioData

#     if (ctrlCode & 2) != 0:
#         audioData.isOver = True

#     audioData.AppendSample(sampleBytes)
#     return audioData

# def ExtractAndCombineFeatures(y, sr, frame_length, hop_length, include_autocorr=True):
#     all_features = []
#     mfcc_features = extract_mfcc_features(y, sr, frame_length, hop_length)
#     all_features.append(mfcc_features)


#     if include_autocorr:
#         autocorr_features = extract_autocorrelation_features(
#             y, sr, frame_length, hop_length
#         )
#         all_features.append(autocorr_features)
    
#     combined_features = np.hstack(all_features)

#     return combined_features


# def ExtractAudioFeatures(pcm: bytes, sampleRate: int):

    
#     y = load_pcm_audio_from_bytes(pcm, sampleRate)
#     print("pcm", len(pcm), "y shape:", y.shape)
#     sr = 88200
#     frame_length = int(0.01667 * sr)  # Frame length set to 0.01667 seconds (~60 fps)
#     hop_length = frame_length // 2  # 2x overlap for smoother transitions
#     min_frames = 9  # Minimum number of frames needed for delta calculation

#     num_frames = (len(y) - frame_length) // hop_length + 1


#     if num_frames < min_frames:
#         print(f"Audio file is too short: {num_frames} frames, required: {min_frames} frames")
#         return None, None

#     combined_features: np.ndarray = ExtractAndCombineFeatures(y, sr, frame_length, hop_length)

#     print("combined_features shape:", combined_features.shape)
    
#     return combined_features, y



face_audio_stream_dict: dict = {}
body_audio_stream_dict: dict = {}

@app.route('/audio_to_face_and_body', methods=['POST'])
def face_and_body_process():
    id = int.from_bytes(request.data[:4], byteorder='little')
    sample_rate = int.from_bytes(request.data[4:8], byteorder='little')
    ctrl_code = int.from_bytes(request.data[8:12], byteorder='little')
    audio_bytes = request.data[12:]
    face_audio_stream = None
    body_audio_stream = None
    global face_audio_stream_dict, body_audio_stream_dict
    if (ctrl_code & 1) != 0:
        face_audio_stream = AudioStream(id, sample_rate, 61)
        body_audio_stream = AudioStream(id, sample_rate, 87)
        face_audio_stream_dict[id] = face_audio_stream
        body_audio_stream_dict[id] = body_audio_stream
    else:
        face_audio_stream = face_audio_stream_dict[id]
        body_audio_stream = body_audio_stream_dict[id]
    
    if face_audio_stream is None or body_audio_stream is None:
        return jsonify({'error': 'Audio stream not found'}), 404

    if (ctrl_code & 2) != 0:
        face_audio_stream.is_over = True
        body_audio_stream.is_over = True
    output_bytes = b''
    face_audio_stream.write(audio_bytes)
    body_audio_stream.write(audio_bytes)
    output_bytes += face_audio_stream.process_by_frames(face_model, device, face_config)
    output_bytes += body_audio_stream.process_by_frames(body_model, device, body_config)
    return Response(
        output_bytes,
        mimetype='application/octet-stream',
        headers={
            'Content-Length': str(len(output_bytes))
        }
    )

@app.route('/audio_to_face', methods=['POST'])
def face_process():
    id = int.from_bytes(request.data[:4], byteorder='little')
    sample_rate = int.from_bytes(request.data[4:8], byteorder='little')
    ctrl_code = int.from_bytes(request.data[8:12], byteorder='little')
    audio_bytes = request.data[12:]
    if (ctrl_code & 1) != 0:
        face_audio_stream = AudioStream(id, sample_rate, face_config['output_dim'])
        face_audio_stream_dict[id] = face_audio_stream
    else:
        face_audio_stream = face_audio_stream_dict.get(id, None)
        face_audio_stream_dict[id] = face_audio_stream
    if face_audio_stream is None:
        print("出错")
        return jsonify({'error': 'Face Audio stream not found'}), 404
    face_audio_stream.write(audio_bytes)
    if (ctrl_code & 2) != 0:
        face_audio_stream.is_over = True
        face_audio_stream_dict.pop(id, None)
    print("face_audio_stream.audio_bytes", len(face_audio_stream.audio_bytes))
    output_bytes = face_audio_stream.process_by_frames(face_model, device, face_config)
    print("脸部数据下发", len(output_bytes))
    return Response(
        output_bytes,
        mimetype='application/octet-stream',
        headers={
            'Content-Length': str(len(output_bytes))
        }
    )

@app.route('/audio_to_body', methods=['POST'])
def body_process():
    id = int.from_bytes(request.data[:4], byteorder='little')
    sample_rate = int.from_bytes(request.data[4:8], byteorder='little')
    ctrl_code = int.from_bytes(request.data[8:12], byteorder='little')
    audio_bytes = request.data[12:]
    if (ctrl_code & 1) != 0:
        body_audio_stream = AudioStream(id, sample_rate, body_config['output_dim'])
        body_audio_stream_dict[id] = body_audio_stream
    else:
        body_audio_stream = body_audio_stream_dict.get(id, None)
    if body_audio_stream is None:
        print("body_audio_stream is None")
        return jsonify({'error': 'Body Audio stream not found'}), 404
    body_audio_stream.write(audio_bytes)
    if (ctrl_code & 2) != 0:
        body_audio_stream.is_over = True
        body_audio_stream_dict.pop(id, None)
    output_bytes = body_audio_stream.process_by_frames(body_model, device, body_config)

    return Response(
        output_bytes,
        mimetype='application/octet-stream',
        headers={
            'Content-Length': str(len(output_bytes))
        }
    )



if __name__ == '__main__':
    #Test()
    app.run(host='127.0.0.1', port=5000)
