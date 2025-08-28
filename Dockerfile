FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --upgrade pip

# Forçar remoção do blinker problemático
RUN pip install --break-system-packages pip-autoremove
RUN pip-autoremove blinker -y || true

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt

# Copiar código da aplicação
COPY handler.py .

# Pre-download do modelo XTTS v2
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)"

# Comando para iniciar
CMD ["python", "handler.py"]