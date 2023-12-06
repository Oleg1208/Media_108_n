# Для запуска локально uvicorn main:app --reload

import tempfile
import os
from fastapi import FastAPI, File, UploadFile
import zipfile
import io
import asyncio
import time
from keras.models import load_model
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import pickle
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="site", html=True), name="static")

# Загрузка модели
model = load_model("model_audio_94%.h5")

with open('x_scaler.pickle', 'rb') as f:
    x_scaler = pickle.load(f)

# Функция для извлечения признаков из аудио
def get_features(y, sr, n_fft=4096, hop_length=256):\
    # Вычисление различных параметров (признаков) аудио

    # Хромаграмма
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Мел-кепстральные коэффициенты
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Среднеквадратическая амплитуда
    rmse = librosa.feature.rms(y=y, hop_length=hop_length)
    # Спектральный центроид
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Ширина полосы частот
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Спектральный спад частоты
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Пересечения нуля
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

    # Сборка признаков в общий список:
    # На один файл несколько векторов признаков, количество определяется
    # продолжительностью аудио и параметром hop_length в функциях расчета признаков
    features = {'rmse': rmse,
                'spct': spec_cent,
                'spbw': spec_bw,
                'roff': rolloff,
                'zcr' : zcr,
                'mfcc': mfcc,
                'stft': chroma_stft}

    return features

# Функция для объединения признаков в набор векторов
def stack_features(feat):
    features = None
    for v in feat.values():
        features = np.vstack((features, v)) if features is not None else v

    return features.T

# Функция формирования подвыборки признаков и меток класса для одного файла

def process_file(feature_set):
    x_list = []

    # Добавление данных в наборы
    for j in range(feature_set.shape[0]):
        x_list.append(feature_set[j])

    return np.array(x_list).astype('float32')

def get_audio_duration(y, sr):
    """Вычисление продолжительности аудио в секундах."""
    return len(y) / sr

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()  # Начало измерения времени

    # Сохранение и чтение файла
    with open("temp_file.mp3", "wb") as buffer:
        buffer.write(file.file.read())
    
    # Загрузка и извлечение признаков
    y, sr = librosa.load("temp_file.mp3", mono=True, duration=200)
    duration = get_audio_duration(y, sr)
    features = get_features(y, sr)
    feature_set = stack_features(features)
    file_x_data = process_file(feature_set)
    # x_scaler = StandardScaler()
    file_x_data_normalized = x_scaler.transform(file_x_data)
    print(feature_set.shape)
    # aggregated_features = np.mean(feature_set, axis=0)

    # Предсказание модели
    prediction = model.predict(file_x_data_normalized)
    predict_mean = prediction.mean(axis=0)

    processing_time = time.time() - start_time  # Вычисление времени обработки

    class_labels = ['нецелевой', 'целевой']
    predicted_class = class_labels[np.argmax(predict_mean)]

    # Вернуть предсказание
    # return {"prediction": str(prediction[0])}
    return {
        "fileName": file.filename,
        "prediction": predicted_class,
        "processingTime": processing_time,
        "audioDuration": duration,
        "prediction_vector": str(predict_mean)
    }

@app.post("/predict_zip/")
async def predict_zip(file: UploadFile = File(...)):
    content = await file.read()
    zip_file = zipfile.ZipFile(io.BytesIO(content))
    tasks = []

    for filename in zip_file.namelist():
        if filename.endswith('.mp3'):
            tasks.append(process_audio_file(filename, zip_file))

    results = await asyncio.gather(*tasks)
    return results

async def process_audio_file(filename, zip_file):
    start_time = time.time()

    # Создание временного файла
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    try:
        with zip_file.open(filename) as source_file:
            audio_data = source_file.read()
            temp_file.write(audio_data)
        temp_file.close()

        # Загрузка и извлечение признаков
        y, sr = librosa.load(temp_file.name, mono=True, duration=200)
        duration = get_audio_duration(y, sr)
        features = get_features(y, sr)
        feature_set = stack_features(features)
        file_x_data = process_file(feature_set)
        file_x_data_normalized = x_scaler.transform(file_x_data)

        # Предсказание модели
        prediction = model.predict(file_x_data_normalized)
        predict_mean = prediction.mean(axis=0)

        class_labels = ['нецелевой', 'целевой']
        predicted_class = class_labels[np.argmax(predict_mean)]

        processing_time = time.time() - start_time
    finally:
        # Удаление временного файла
        os.remove(temp_file.name)

    return {
        "fileName": filename,
        "prediction": predicted_class,
        "processingTime": processing_time,
        "audioDuration": duration,
        "prediction_vector": str(predict_mean)
    }
# async def predict_zip(file: UploadFile = File(...)):
#     content = await file.read()
#     zip_file = zipfile.ZipFile(io.BytesIO(content))
#     results = []

#     for filename in zip_file.namelist():
#         if filename.endswith('.mp3'):  # Убедитесь, что обрабатываете только аудиофайлы
#             start_time = time.time()
#             with zip_file.open(filename) as f:
#                 # Чтение файла из ZIP-архива
#                 audio_data = f.read()
#                 with open("temp_file.mp3", "wb") as audio_file:
#                     audio_file.write(audio_data)

#                 # Загрузка и извлечение признаков
#                 y, sr = librosa.load("temp_file.mp3", mono=True, duration=200)
#                 features = get_features(y, sr)
#                 feature_set = stack_features(features)
#                 file_x_data = process_file(feature_set)
#                 file_x_data_normalized = x_scaler.transform(file_x_data)

#                 # Предсказание модели
#                 prediction = model.predict(file_x_data_normalized)
#                 predict_mean = prediction.mean(axis=0)

#                 class_labels = ['нецелевой', 'целевой']
#                 predicted_class = class_labels[np.argmax(predict_mean)]

#                 processing_time = time.time() - start_time

#                 # Добавление результата в список
#                 results.append({
#                     "fileName": filename,
#                     "prediction": predicted_class,
#                     "processingTime": processing_time,
#                     "prediction_vector": str(predict_mean)
#                 })

#     return results

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)