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

# Cargar el modelo, scaler y el selector previamente entrenados
#modelo_MLP = joblib.load("modelo_MLP.pkl")

modelo = joblib.load("modelo_LGBM.pkl")
scaler = joblib.load("scaler_LGBM.pkl")
selector = joblib.load("selector_LGBM.pkl")

# Diccionario de emociones
diccionario_emociones = {
    0: 'Felicidad',
    1: 'Ira',
    2: 'Miedo',
    3: 'Tristeza'
}

# Variable global
ultima_emocion_detectada = None
ultimo_audio_bytes = None

import requests

def lanzar_monkey_trigger(emocion):
    # Normaliza la emoción para que coincida con los nombres de los triggers
    emocion = emocion.lower()
    nombre_dispositivo = f"estado{emocion}"

    url = f"https://api-v2.voicemonkey.io/trigger?token=1806d0d32cdbe252c28f2a04991d0ff1_a0c9a2824a787b73a34068bccc2a454f&device={nombre_dispositivo}"
    
    try:
        response = requests.get(url)
        print(f"Voice Monkey triggered for '{emocion}': {response.status_code} - {response.text}")
    except Exception as e:
        print("Error durante la solicitud:", e)

@app.route('/')
def home():
    return render_template('index.html')

def extraer_caracteristicas_voz(ruta_audio):
    y, sr = librosa.load(ruta_audio, sr=None)

    # Normalización de amplitud
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    features = []

    # ===== 1. MFCCs (13 × 4 = 52) =====
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for stat_func in [np.mean, np.std, scipy.stats.kurtosis, scipy.stats.skew]:
        stats = stat_func(mfcc, axis=1)
        features.extend(stats)

    # ===== 2. ΔMFCC (13) =====
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))

    # ===== 3. ΔΔMFCC (13) =====
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2_mfcc, axis=1))

    # ===== 4. Spectral Centroid (6) =====
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    for stat_func in [np.mean, np.std, scipy.stats.kurtosis, scipy.stats.skew, np.max, np.min]:
        features.append(stat_func(centroid, axis=1)[0])

    # ===== 5. Spectral Flatness (6) =====
    flatness = librosa.feature.spectral_flatness(y=y)
    for stat_func in [np.mean, np.std, scipy.stats.kurtosis, scipy.stats.skew, np.max, np.min]:
        features.append(stat_func(flatness, axis=1)[0])

    # ===== 6. Spectral Contrast (7×4 = 28) =====
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for stat_func in [np.mean, np.std, scipy.stats.kurtosis, scipy.stats.skew]:
        stats = stat_func(contrast, axis=1)
        features.extend(stats)

    # Máximo y mínimo global del contraste (2)
    features.append(np.max(contrast))
    features.append(np.min(contrast))

    # ===== 7. LPC (17 coeficientes) =====
    try:
        lpc = librosa.lpc(y, order=17)
        features.extend(lpc[:17])
    except Exception as e:
        print("⚠️ Error calculando LPC:", e)
        features.extend([0.0] * 17)

    # ===== 8. Prosódicas: F0, RMS, Tempo (3) =====
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
    except Exception:
        f0_mean = 0.0
    features.append(f0_mean)

    try:
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms).item())
    except Exception:
        features.append(0.0)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
    except Exception:
        features.append(0.0)

    # Convertir a array y recortar/exactitud
    features = [float(np.array(f).item()) for f in features]
    #features = np.clip(features, -1000, 1000)
    features = features[:140]  # Por seguridad
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

    # Guardar en memoria para depuración
    global ultimo_audio_bytes
    ultimo_audio_bytes = archivo_audio.read()
    archivo_audio.stream.seek(0)  # Para poder seguir usándolo normalmente

    archivo_audio.save(ruta_audio)

    ruta_audio_wav = convertir_a_wav(ruta_audio)
    muestra = extraer_caracteristicas_voz(ruta_audio_wav)
                    
    # Escalar las características
    muestra_escalada = scaler.transform(muestra)
  
    # Seleccionar las mejores características utilizando SelectKBest
    muestra_seleccionada = selector.transform(muestra_escalada)

    # Realizar la predicción con el modelo MLP
    pred = modelo.predict(muestra_seleccionada)
    #emocion_predicha = diccionario_emociones[np.argmax(pred[0])]
    emocion_predicha = diccionario_emociones[int(pred[0])]

    global ultima_emocion_detectada
    ultima_emocion_detectada = emocion_predicha

    # lanza el trigger para que Alexa suene    
    lanzar_monkey_trigger(emocion_predicha)
    
    return jsonify({"emocion_detectada": emocion_predicha}), 200

@app.route('/ultima_emocion', methods=['GET'])
def obtener_ultima_emocion():
    if ultima_emocion_detectada is None:
        return jsonify({"emocion": "desconocida"})
    return jsonify({"emocion": ultima_emocion_detectada})

from flask import send_file

@app.route('/audio_guardado')
def audio_guardado():
    return send_file('audio_recibido.wav', mimetype='audio/wav')

import io

@app.route('/debug_audio')
def debug_audio():
    global ultimo_audio_bytes
    if ultimo_audio_bytes is None:
        return "No hay audio recibido aún.", 404

    return send_file(
        io.BytesIO(ultimo_audio_bytes),
        mimetype='audio/wav',
        as_attachment=False,
        download_name='debug_audio.wav'
    )


if __name__ == '__main__':
    app.run()
