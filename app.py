from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import librosa
import numpy as np
import joblib
import scipy.stats

# Crear la app de Flask
app = Flask(__name__)

# Activa CORS
from flask_cors import CORS
CORS(app)

# Cargar el modelo y el scaler previamente entrenados
modelo_rf = joblib.load("modelo_rf.pkl")
scaler = joblib.load("scaler.pkl")

# Diccionario de emociones
diccionario_emociones = {
    0: 'Felicidad',
    1: 'Asco',
    2: 'Ira',
    3: 'Miedo',
    4: 'Neutral',
    5: 'Sorpresa',
    6: 'Tristeza'
}

# Variable global
ultima_emocion_detectada = None

import requests

# Función para lanzar el triguer, token API v2 
VOICE_MONKEY_TOKEN = "1806d0d32cdbe252c28f2a04991d0ff1_a0c9a2824a787b73a34068bccc2a454f"
MONKEY_NAME = "felicidaddetectada"  # o el nombre exacto del Monkey creado

def lanzar_monkey_trigger():
    url = f"https://api.voicemonkey.io/trigger"
    headers = {
        "Authorization": f"Bearer {VOICE_MONKEY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "monkey": MONKEY_NAME
    }
    response = requests.post(url, json=data, headers=headers)
    print("Voice Monkey response:", response.status_code, response.text)


@app.route('/')
def home():
    return render_template('index.html')

def extraer_caracteristicas_voz(ruta_audio):
    y, sr = librosa.load(ruta_audio, sr=None)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for stat_func in [np.mean, np.std, scipy.stats.skew, scipy.stats.kurtosis]:
        features.extend(stat_func(mfcc, axis=1))

    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.mean(delta2_mfcc, axis=1))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    for stat_func in [np.mean, np.std, np.min, np.max, scipy.stats.skew, scipy.stats.kurtosis]:
        features.extend(stat_func(centroid, axis=1))

    flatness = librosa.feature.spectral_flatness(y=y)
    for stat_func in [np.mean, np.std, np.min, np.max, scipy.stats.skew, scipy.stats.kurtosis]:
        features.extend(stat_func(flatness, axis=1))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for stat_func in [np.mean, np.std, np.min, np.max]:
        features.extend(stat_func(contrast, axis=1))

    features.append(np.max(contrast).item())
    features.append(np.min(contrast).item())

    try:
        lpc = librosa.lpc(y, order=17)
        features.extend(lpc[:17])
    except Exception as e:
        print("⚠️ Error calculando LPC:", e)
        features.extend([0.0] * 17)

    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
    except Exception as e:
        print("⚠️ Error en F0:", e)
        f0_mean = 0
    features.append(f0_mean)

    try:
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms).item())
    except Exception as e:
        print("⚠️ Error en RMS:", e)
        features.append(0.0)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
    except Exception as e:
        print("⚠️ Error en tempo:", e)
        features.append(0.0)

    features = [float(np.array(f).item()) for f in features]
    features = np.clip(features, -1000, 1000)

    return np.array(features).reshape(1, -1)

def convertir_a_wav(ruta_entrada):
    ruta_salida = 'audio_convertido.wav'
    try:
        audio = AudioSegment.from_file(ruta_entrada)
        audio.export(ruta_salida, format='wav')
        return ruta_salida
    except Exception as e:
        print("❌ Error en la conversión a WAV:", e)
        raise

@app.route('/prediccion', methods=['POST'])
def prediccion():
    if 'audio' not in request.files:
        return jsonify({"error": "No se ha enviado un archivo de audio."}), 400

    archivo_audio = request.files['audio']
    ruta_audio = 'audio_recibido.wav'
    archivo_audio.save(ruta_audio)

    ruta_audio_wav = convertir_a_wav(ruta_audio)
    muestra = extraer_caracteristicas_voz(ruta_audio_wav)
    muestra_scaled = scaler.transform(muestra)
    pred = modelo_rf.predict(muestra_scaled)
    emocion_predicha = diccionario_emociones[int(pred[0])]

    global ultima_emocion_detectada
    ultima_emocion_detectada = emocion_predicha

    # Si es felicidad, lanza el trigger para que Alexa suene
    if emocion_predicha == "felicidad":
        lanzar_monkey_trigger()

    return jsonify({"emocion_detectada": emocion_predicha}), 200

@app.route('/ultima_emocion', methods=['GET'])
def obtener_ultima_emocion():
    if ultima_emocion_detectada is None:
        return jsonify({"emocion": "desconocida"})
    return jsonify({"emocion": ultima_emocion_detectada})

if __name__ == '__main__':
    app.run()
