# =========================================================
#  DEMO - PARALLEL CNN + ATTENTION LSTM (INFERENCE)
# =========================================================

import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib
import os

# ===================== CONFIG =====================
SAMPLE_RATE = 48000
DURATION = 3.0
OFFSET = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = {
    0: 'surprise',
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fear',
    7: 'disgust'
}

# ===================== MODEL =====================
class ParallelModel(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),

            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Dropout(0.3),

            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Dropout(0.3),

            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Dropout(0.3)
        )

        self.lstm_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        hidden_size = 128
        self.lstm = nn.LSTM(64, hidden_size, bidirectional=True, batch_first=True)
        self.attention_linear = nn.Linear(2*hidden_size,1)
        self.out_linear = nn.Linear(2*hidden_size+256, num_emotions)

    def forward(self,x):
        conv = self.conv2Dblock(x)
        conv = torch.flatten(conv, start_dim=1)

        x_red = self.lstm_maxpool(x)
        x_red = torch.squeeze(x_red,1)
        x_red = x_red.permute(0,2,1)

        lstm_out,_ = self.lstm(x_red)
        T = lstm_out.size(1)
        attn = torch.stack([self.attention_linear(lstm_out[:,t,:]) for t in range(T)], -1)
        attn = torch.softmax(attn, -1)
        context = torch.bmm(attn, lstm_out).squeeze(1)

        final = torch.cat([conv, context], dim=1)
        logits = self.out_linear(final)
        probs = torch.softmax(logits, dim=1)
        return logits, probs, attn

# ===================== AUDIO =====================
def load_audio(path):
    audio,_ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET)
    signal = np.zeros(int(SAMPLE_RATE*DURATION))
    signal[:len(audio)] = audio
    return signal

def get_mel(signal):
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_fft=1024,
        win_length=512,
        hop_length=256,
        window="hamming",
        n_mels=128,
        fmax=SAMPLE_RATE/2
    )
    return librosa.power_to_db(mel, ref=np.max)

def normalize_mel(mel, scaler):
    mel = np.expand_dims(mel, axis=0)   # (1, 128, T)
    mel = np.expand_dims(mel, axis=1)   # (1, 1, 128, T)

    b,c,h,w = mel.shape
    flat = mel.reshape(b,-1)
    flat = scaler.transform(flat)
    return flat.reshape(b,c,h,w)

# ===================== LOAD =====================
def load_model(path):
    model = ParallelModel(len(EMOTIONS)).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

# ===================== PREDICT =====================
def predict(audio_path, model, scaler):
    signal = load_audio(audio_path)
    mel = get_mel(signal)
    mel = normalize_mel(mel, scaler)

    X = torch.tensor(mel, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, probs,_ = model(X)

    probs = probs.cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return EMOTIONS[pred], probs

# ===================== MAIN =====================
if __name__ == "__main__":

    MODEL_PATH = "model/cnn_lstm_parallel_model_1.pt" # cnn_transformer_model.pt
    SCALER_PATH = "model/mel_scaler_parallel_attention.pkl" # mel_scaler_transformer.pkl
    AUDIO_FILE = "03-01-07-01-02-01-02.wav"

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    emotion, probs = predict(AUDIO_FILE, model, scaler)

    print("\nPredicted emotion:", emotion)
    print("-"*30)
    for i,p in enumerate(probs):
        print(f"{EMOTIONS[i]:>10s}: {p:.4f}")
