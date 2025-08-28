# handler.py - RunPod Serverless Handler
import runpod
import os
import tempfile
import torch
import logging
import time
import gc
import numpy as np
import soundfile as sf
import librosa
import base64
import json
from TTS.api import TTS
from pathlib import Path
from TTS.tts.utils.text.cleaners import portuguese_cleaners
from pydub import AudioSegment
import subprocess
import io

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variável global para o modelo
tts_model = None

# ========== FUNÇÕES ESSENCIAIS PARA ÁUDIO LIMPO ==========

def preprocess_reference_audio(input_path: str, output_path: str) -> bool:
    """Pré-processa áudio de referência para 16kHz mono (ideal para XTTS v2)"""
    try:
        audio, orig_sr = librosa.load(input_path, sr=None, mono=False)
        
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        if orig_sr != 16000:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        
        # Normalização suave
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        sf.write(output_path, audio, 16000, subtype='PCM_16')
        logger.info(f"✅ Áudio de referência: {orig_sr}Hz → 16kHz mono")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pré-processamento: {e}")
        return False

def postprocess_to_clean_audio(input_path: str, output_path: str, output_format: str = "wav") -> bool:
    """
    Pós-processa para áudio limpo e claro:
    - Converte para 44.1kHz estéreo
    - Normalização final
    - Suporte para WAV ou MP3
    """
    try:
        audio, sr = sf.read(input_path)
        logger.info(f"📊 Áudio original: {sr}Hz")
        
        # Resample para 44.1kHz (qualidade padrão universal)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        
        # Converter para estéreo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        # Normalização final para áudio limpo
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        if output_format.lower() == "mp3":
            # Salvar temporariamente como WAV e converter para MP3
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, audio, 44100, subtype='PCM_16')
            
            # Converter para MP3 usando pydub
            try:
                audio_segment = AudioSegment.from_wav(temp_wav)
                audio_segment.export(output_path, format="mp3", bitrate="320k")
                os.unlink(temp_wav)  # Remove arquivo temporário
                logger.info(f"✅ Áudio limpo: 44.1kHz estéreo MP3 (320kbps)")
            except Exception as e:
                logger.warning(f"⚠️ Falha na conversão MP3 com pydub: {e}")
                # Fallback: usar ffmpeg se disponível
                try:
                    subprocess.run([
                        'ffmpeg', '-i', temp_wav, '-codec:a', 'mp3', 
                        '-b:a', '320k', '-y', output_path
                    ], check=True, capture_output=True)
                    os.unlink(temp_wav)
                    logger.info(f"✅ Áudio limpo: 44.1kHz estéreo MP3 (320kbps) via ffmpeg")
                except Exception as ffmpeg_error:
                    logger.error(f"❌ Erro na conversão MP3: {ffmpeg_error}")
                    # Como último recurso, renomeia WAV para MP3 (não recomendado)
                    os.rename(temp_wav, output_path)
                    return False
        else:
            # Salvar como WAV
            sf.write(output_path, audio, 44100, subtype='PCM_16')
            logger.info(f"✅ Áudio limpo: 44.1kHz estéreo WAV")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pós-processamento: {e}")
        return False

def cleanup_temp_files(file_paths: list):
    """Remove arquivos temporários"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover {path}: {e}")

def clean_text_for_portuguese_tts(text):
    """Limpa texto para TTS em português"""
    import re
    
    # Expandir abreviações
    abbrevs = {"Dr.": "Doutor", "Dra.": "Doutora", "Sr.": "Senhor", "Sra.": "Senhora"}
    for old, new in abbrevs.items():
        text = text.replace(old, new)
    
    # Aplicar cleaner português
    try:
        text = portuguese_cleaners(text)
    except:
        logger.warning("⚠️ Falha no cleaner português, usando texto original")
    
    # Números decimais
    text = re.sub(r'(\d+)\.(\d+)', r'\1 vírgula \2', text)
    text = text.replace('.', ';\n')
    text = text.replace('-', ' ')
    
    return text

def decode_base64_audio(base64_data: str, output_path: str) -> bool:
    """Decodifica áudio base64 para arquivo"""
    try:
        # Remove prefixo data:audio se presente
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        audio_bytes = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao decodificar áudio base64: {e}")
        return False

def encode_audio_to_base64(audio_path: str) -> str:
    """Codifica áudio para base64"""
    try:
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"❌ Erro ao codificar áudio para base64: {e}")
        return None

# ========== INICIALIZAÇÃO DO MODELO ==========

def initialize_model():
    """Inicializa o modelo TTS"""
    global tts_model
    
    if tts_model is not None:
        return True
    
    try:
        logger.info("🚀 Inicializando modelo XTTS-v2...")
        
        logger.info(f"🔍 CUDA disponível: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 GPU: {gpu_name}")
        else:
            device = "cpu"
            logger.warning("⚠️ Usando CPU")
        
        start_time = time.time()
        
        tts_model = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2", 
            gpu=torch.cuda.is_available()
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Modelo carregado em {load_time:.2f}s!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return False

# ========== HANDLER PRINCIPAL ==========

def handler(job):
    """Handler principal do RunPod"""
    try:
        # Inicializar modelo se necessário
        if not initialize_model():
            return {"error": "Falha ao inicializar modelo TTS"}
        
        # Extrair parâmetros do job
        job_input = job.get('input', {})
        
        # Parâmetros obrigatórios
        text = job_input.get('text', '').strip()
        reference_audio_b64 = job_input.get('reference_audio', '')  # Base64
        
        if not text:
            return {"error": "Parâmetro 'text' é obrigatório"}
        
        if not reference_audio_b64:
            return {"error": "Parâmetro 'reference_audio' (base64) é obrigatório"}
        
        # Parâmetros opcionais
        language = job_input.get('language', 'pt')
        output_format = job_input.get('output_format', 'wav').lower()
        temperature = float(job_input.get('temperature', 0.75))
        length_penalty = float(job_input.get('length_penalty', 1.0))
        repetition_penalty = float(job_input.get('repetition_penalty', 5.0))
        top_k = int(job_input.get('top_k', 50))
        top_p = float(job_input.get('top_p', 0.85))
        speed = float(job_input.get('speed', 1.0))
        enable_text_splitting = job_input.get('enable_text_splitting', True)
        text_cleaner = job_input.get('text_cleaner', True)
        
        # Validar formato de saída
        if output_format not in ["wav", "mp3"]:
            return {"error": "output_format deve ser 'wav' ou 'mp3'"}
        
        start_time = time.time()
        ref_path = None
        processed_ref_path = None
        output_path = None
        final_path = None
        
        try:
            logger.info(f"🎵 TTS {output_format.upper()} - Texto: {len(text)} chars, Idioma: {language}")
            
            # 1. Decodificar e salvar áudio de referência
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
                ref_path = ref_file.name
            
            if not decode_base64_audio(reference_audio_b64, ref_path):
                return {"error": "Falha ao decodificar áudio de referência"}
            
            # 2. Pré-processar para 16kHz mono
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_ref_file:
                processed_ref_path = processed_ref_file.name
            
            if not preprocess_reference_audio(ref_path, processed_ref_path):
                return {"error": "Falha no pré-processamento do áudio"}
            
            # 3. Gerar áudio usando streaming do XTTS v2
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                output_path = output_file.name
            
            logger.info("🎤 Gerando áudio com streaming...")
            
            text_to_process = clean_text_for_portuguese_tts(text) if text_cleaner else text
            
            tts_model.tts_to_file(
                text=text_to_process,
                speaker_wav=processed_ref_path,
                language=language,
                file_path=output_path,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                speed=speed,
                split_sentences=enable_text_splitting
            )
            
            # 4. Pós-processar para áudio limpo
            file_extension = "mp3" if output_format == "mp3" else "wav"
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as final_file:
                final_path = final_file.name
            
            if not postprocess_to_clean_audio(output_path, final_path, output_format):
                # Fallback
                if output_format == "mp3":
                    final_path = output_path
                    output_format = "wav"
                    file_extension = "wav"
                    logger.warning("⚠️ Fallback para WAV devido falha na conversão MP3")
                else:
                    final_path = output_path
                    logger.warning("⚠️ Usando áudio sem pós-processamento")
            
            # Verificar arquivo final
            if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
                return {"error": "Falha na geração do áudio"}
            
            # 5. Codificar resultado para base64
            audio_base64 = encode_audio_to_base64(final_path)
            if not audio_base64:
                return {"error": "Falha ao codificar áudio de saída"}
            
            # Estatísticas
            generation_time = time.time() - start_time
            file_size = os.path.getsize(final_path)
            
            try:
                if output_format == "wav":
                    final_audio, final_sr = sf.read(final_path)
                    duration = len(final_audio) / final_sr
                else:
                    audio_segment = AudioSegment.from_file(final_path)
                    duration = len(audio_segment) / 1000.0
                    final_sr = audio_segment.frame_rate
                
                logger.info(f"✅ {output_format.upper()} gerado em {generation_time:.2f}s - {duration:.1f}s @ {final_sr}Hz")
            except:
                logger.info(f"✅ {output_format.upper()} gerado em {generation_time:.2f}s - {file_size} bytes")
                duration = 0
                final_sr = 44100
            
            # Resultado final
            result = {
                "audio_base64": audio_base64,
                "format": output_format,
                "generation_time": round(generation_time, 2),
                "file_size": file_size,
                "duration": round(duration, 2) if duration > 0 else 0,
                "sample_rate": final_sr,
                "clean_audio": True,
                "version": "3.1.0-runpod"
            }
            
            return result
            
        finally:
            # Limpar arquivos temporários
            cleanup_temp_files([ref_path, processed_ref_path, output_path, final_path])
    
    except Exception as e:
        logger.error(f"❌ Erro no handler: {str(e)}")
        return {"error": f"Erro na síntese: {str(e)}"}

# ========== INICIALIZAÇÃO RUNPOD ==========

if __name__ == "__main__":
    logger.info("🚀 Iniciando RunPod Serverless Handler...")
    runpod.serverless.start({"handler": handler})