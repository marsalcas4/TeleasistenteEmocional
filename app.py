from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import librosa
import numpy as np
import joblib
import scipy.stats
import threading

# Crear la app de Flask
app = Flask(__name__)

# Cargar el modelo y el scaler previamente entrenados
modelo_rf = joblib.load("modelo_rf.pkl")  # Aquí deberías tener tu modelo previamente entrenado
scaler = joblib.load("scaler.pkl")  # El scaler que usaste para normalizar los datos

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

# Ruta principal (Página de bienvenida)
@app.route('/')
def home():
    return render_template('index.html')

# Función para extraer características del audio
def extraer_caracteristicas_voz(ruta_audio):
    y, sr = librosa.load(ruta_audio, sr=None)

    # Normalizar amplitud del audio para evitar saturación
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    features = []

    # 1. MFCCs (13 × 4 estadísticas = 52)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for stat_func in [np.mean, np.std, scipy.stats.skew, scipy.stats.kurtosis]:
        features.extend(stat_func(mfcc, axis=1))

    # 2. Delta y delta-delta MFCCs (13 × 2 = 26)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.mean(delta2_mfcc, axis=1))

    # 3. Centroide espectral (6)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    for stat_func in [np.mean, np.std, np.min, np.max, scipy.stats.skew, scipy.stats.kurtosis]:
        features.extend(stat_func(centroid, axis=1))

    # 4. Planitud espectral (6)
    flatness = librosa.feature.spectral_flatness(y=y)
    for stat_func in [np.mean, np.std, np.min, np.max, scipy.stats.skew, scipy.stats.kurtosis]:
        features.extend(stat_func(flatness, axis=1))

    # 5. Contraste espectral (7 subbandas × 4 estadísticas = 28)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for stat_func in [np.mean, np.std, np.min, np.max]:
        features.extend(stat_func(contrast, axis=1))

    # 6. Máximo y mínimo global del contraste (2)
    features.append(np.max(contrast).item())
    features.append(np.min(contrast).item())

    # 7. LPC (17 coeficientes)
    try:
        lpc = librosa.lpc(y, order=17)
        features.extend(lpc[:17])
    except Exception as e:
        print("⚠️ Error calculando LPC:", e)
        features.extend([0.0] * 17)

    # 8. Prosódicas: F0, RMS, Tempo (3)
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

    # Seguridad: convertir a float y limitar valores extremos
    features = [float(np.array(f).item()) for f in features]
    features = np.clip(features, -1000, 1000)

    # Imprimir los valores de las características extraídas para el nuevo audio
    #print("Características extraídas para el audio:", ruta_audio)
    #for idx, value in enumerate(features):
    #    print(f"Característica {nombre_caracteristica(idx)}: {value:.3f}")
    
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
        

#def convertir_a_wav_valido(input_path, output_path='audio_limpio.wav'):
#    audio = AudioSegment.from_file(input_path)
#    audio = audio.set_channels(1).set_frame_rate(22050)
#    audio.export(output_path, format="wav")
#    return output_path


# Ruta para recibir un archivo de audio y hacer la predicción
@app.route('/prediccion', methods=['POST'])
def prediccion():
    if 'audio' not in request.files:
        return jsonify({"error": "No se ha enviado un archivo de audio."}), 400

    archivo_audio = request.files['audio']
    
    # Guardar el archivo temporalmente
    ruta_audio = 'audio_recibido.wav'
    archivo_audio.save(ruta_audio)
    
    # Convertir a wav antes de procesar
    ruta_audio_wav = convertir_a_wav(ruta_audio)

    # Procesar 
    muestra = extraer_caracteristicas_voz(ruta_audio_wav)

    # Extraer características del archivo de audio
    #muestra = extraer_caracteristicas_voz(ruta_audio)

    # Normalizar las características
    muestra_scaled = scaler.transform(muestra)

    # Predecir la emoción usando el modelo entrenado
    pred = modelo_rf.predict(muestra_scaled)
    emocion_predicha = diccionario_emociones[int(pred[0])]
    
    global ultima_emocion_detectada
    ultima_emocion_detectada = emocion_predicha

    # Retornar la predicción como JSON
    return jsonify({"emocion_detectada": emocion_predicha}), 200

# endpoint para que Alexa consulte la última emoción
@app.route('/ultima_emocion', methods=['GET'])
def obtener_ultima_emocion():
    if ultima_emocion_detectada is None:
        return jsonify({"emocion": "desconocida"})
    return jsonify({"emocion": ultima_emocion_detectada})


# Ejecutar la aplicación Flask en un hilo separado
def run_flask():
    app.run(debug=True, use_reloader=False)  # use_reloader=False evita que se reinicie al cargar el servidor

# Iniciar Flask en un hilo
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
