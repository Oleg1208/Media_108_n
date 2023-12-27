import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import os  # для работы с операционной системой
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


# Запуск python app.py
# Теперь вы можете открыть веб-браузер и перейти по адресу http://localhost:8000 для загрузки аудио файлов.
# После загрузки будет показан результат определения класса на странице result.html.


# Установка cohere openai tiktoken
# pip install cohere openai tiktoken
# Установка openai-whisper
# pip install git+https://github.com/openai/whisper.git
# Установка flask
# pip install flask


class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


app = FastAPI()
UPLOAD_FOLDER = 'C:/Users/kirar/PycharmProjects/Media108-Server/'
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictionResult(BaseModel):
    predicted_class: str


# Загрузка модели whisper medium
whisper_model = whisper.load_model('medium')

# Возможное решение этого предупреждения
# /usr/local/lib/python3.10/dist-packages/whisper/transcribe.py:115:
# UserWarning: FP16 is not supported on CPU; using FP32 instead
#  warnings.warn("FP16 is not supported on CPU; using FP32 instead")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("base", device="cpu")

# Импортируем предобученную модель BERT
bert = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

# Сам BERT обучать не будем, добавим к его выходу свои слои, которые и будем обучать
for param in bert.parameters():
    param.requires_grad = False

# Объявляем модель и загружаем её в GPU
device = torch.device('cuda')
model = BERT_Arch(bert)
model = model.to(device)

# Загрузим лучшие веса для модели
model.load_state_dict(torch.load('C:/Users/kirar/Downloads/bert_weights.pt'))
tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')


# Функция для определения класса по аудио
def predict_audio_class(audio_path):
    # Whisper-транскрибация
    text_whisper = whisper_model.transcribe(audio_path, language='ru')['text']
    print(text_whisper)

    # Bert-предсказание

    # Токенизируем текст
    tokens = tokenizer.tokenize(text_whisper)

    # Преобразуем токены в числовые идентификаторы
    ids = tokenizer.convert_tokens_to_ids(tokens)

    # Добавляем паддинг до максимальной длины последовательности
    max_seq_len = 512
    ids = ids[:max_seq_len - 2]  # учитываем [CLS] и [SEP]
    ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
    mask = [1] * len(ids)
    padding = [tokenizer.pad_token_id] * (max_seq_len - len(ids))
    ids += padding
    mask += padding

    # Преобразуем числовые идентификаторы в тензор PyTorch
    test_seq = torch.tensor(ids).unsqueeze(0)
    test_mask = torch.tensor(mask).unsqueeze(0)

    # Передаем тензор через модель, чтобы получить предсказание
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))

    # Преобразуем выход модели в вероятности классов
    probs = torch.softmax(preds, dim=1)

    # Извлекаем вероятность класса text_whisper
    text_whisper_prob = probs[0, 1].item()

    # Сохраняем вероятность в переменную confidence
    confidence = text_whisper_prob

    threshold = 0.93

    if confidence >= threshold:
        return f'Звонок, путь до которого {audio_path} — ЦЕЛЕВОЙ\nВероятность Целевого - {confidence},' \
               f' Нецелевого - {1 - confidence}'
    else:
        return f'Звонок, путь до которого {audio_path} — НЕЦЕЛЕВОЙ\nВероятность Целевого - {confidence},' \
               f' Нецелевого - {1 - confidence}'


# Маршрут для загрузки файла и получения предсказанного класса
@app.post("/")
async def index(file: UploadFile = File(...)):
    # Проверка расширения файла
    if file.filename.split(".")[-1].lower() not in ["mp3"]:
        return {"error": "Invalid file format"}

    # Генерация безопасного имени файла и сохранение файла
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Вызов функции предсказания класса аудиофайла
    predicted_class = predict_audio_class(file_path)

    # Возвращение результата предсказания
    return {"predicted_class": predicted_class}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
