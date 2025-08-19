import os
import re
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from moviepy.editor import VideoFileClip
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
import shutil
import time

# Cargar variables de entorno desde el archivo .env (para desarrollo local)
load_dotenv()

# --- Configuración ---
# Configura un logger para mostrar información de manera más clara
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configura tu clave de Gemini.
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logging.error("La variable de entorno GEMINI_API_KEY no está configurada.")
    exit()

# --- Funciones Auxiliares ---

def obtener_ruta_unica(ruta_propuesta: Path) -> Path:
    """
    Verifica si una ruta existe. Si es así, añade un sufijo numérico
    entre paréntesis hasta encontrar un nombre de archivo único.
    """
    if not ruta_propuesta.exists():
        return ruta_propuesta

    contador = 2
    while True:
        nueva_ruta = ruta_propuesta.with_stem(f"{ruta_propuesta.stem}({contador})")
        if not nueva_ruta.exists():
            return nueva_ruta
        contador += 1

def gemini_request_with_retry(gemini_model, prompt, max_retries=3, delay=5):
    """
    Envía una petición a Gemini con reintentos en caso de error de cuota.
    """
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            return response
        except ResourceExhausted as e:
            logging.warning(f"Límite de cuota alcanzado. Reintentando en {delay} segundos... (Intento {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            # Para otros errores, fallar directamente
            raise e
    # Si todos los reintentos fallan
    raise Exception("Se superó el número máximo de reintentos por límite de cuota.")


# --- Funciones Principales de IA ---

def transcribir_clip(video_path: Path, gemini_model, start_time: float, end_time: float, temp_dir: Path) -> Optional[str]:
    """
    Extrae un clip de un vídeo a un directorio temporal, lo sube y lo transcribe.
    """
    audio_file = None
    temp_clip_path = temp_dir / f"temp_clip_{int(time.time())}.mp4"

    try:
        logging.info(f"Extrayendo clip de audio a '{temp_clip_path}'...")
        with VideoFileClip(str(video_path)) as clip:
            clip_sub = clip.subclip(start_time, end_time)
            clip_sub.write_videofile(str(temp_clip_path), audio_codec="aac", verbose=False, logger=None)
        
        logging.info("Subiendo archivo de audio para transcripción...")
        audio_file = genai.upload_file(path=temp_clip_path)
        
        logging.info(f"Esperando a que el archivo '{audio_file.name}' esté activo...")
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(name=audio_file.name)

        if audio_file.state.name != "ACTIVE":
            raise Exception(f"El procesamiento del archivo falló: {audio_file.state.name}")

        logging.info("Transcribiendo audio con Gemini...")
        prompt = ["Por favor, transcribe este audio en español.", audio_file]
        response = gemini_request_with_retry(gemini_model, prompt)
        
        return response.text

    except Exception as e:
        logging.error(f"No se pudo transcribir el clip de vídeo: {e}", exc_info=True)
        return None
    finally:
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
            except Exception as delete_e:
                logging.error(f"Error al eliminar el archivo de la nube: {delete_e}")

def obtener_nombre_pelicula(transcripcion: str, gemini_model) -> Optional[str]:
    if not transcripcion: return None
    
    prompt = (
        "Analiza la siguiente transcripción. Tu tarea es extraer el nombre de la película principal.\n"
        "- IMPORTANTE: Si el título en la transcripción está en inglés, busca y devuelve el título con el que se estrenó oficialmente en España.\n"
        "- Si la película existe y estás 100% seguro, devuelve ÚNICAMENTE su título oficial en español (de España).\n"
        "- Si no estás seguro, no existe o no encuentras el título español, DEBES devolver: pelicula_no_encontrada\n\n"
        f"Transcripción: \"{transcripcion}\""
    )
    try:
        logging.info("Consultando a Gemini para obtener el nombre de la película...")
        response = gemini_request_with_retry(gemini_model, prompt)
        nombre_sucio = response.text.strip()
        nombre_limpio = re.sub(r'[\\/*?:"<>|]', "", nombre_sucio)
        return nombre_limpio if nombre_limpio else "nombre_desconocido"
    except Exception as e:
        logging.error(f"Error al contactar con la API de Gemini para el título: {e}", exc_info=True)
        return "nombre_desconocido"

def obtener_puntuacion(transcripcion: str, gemini_model) -> str:
    if not transcripcion: return "no"
    prompt = (
        "Analiza la siguiente transcripción. Extrae la puntuación numérica (de 0 a 10).\n"
        "- Si encuentras una puntuación (incluyendo decimales o 'y medio'), devuelve ÚNICAMENTE el número (ej: '8', '9.5').\n"
        "- Si no hay puntuación, DEBES devolver: no\n\n"
        f"Transcripción: \"{transcripcion}\""
    )
    try:
        logging.info("Consultando a Gemini para obtener la puntuación...")
        response = gemini_request_with_retry(gemini_model, prompt)
        puntuacion_str = response.text.strip().replace(',', '.')
        try:
            valor_numerico = float(puntuacion_str)
            if 0 <= valor_numerico <= 10:
                return puntuacion_str
            else:
                return "no"
        except ValueError:
            return "no"
    except Exception as e:
        logging.error(f"Error al contactar con la API de Gemini para la puntuación: {e}", exc_info=True)
        return "no"

# --- Función Principal de Orquestación ---

def procesar_un_video(ruta_video: Path, gemini_model, temp_dir: Path, destination_dir: Path):
    """
    Contiene la lógica para procesar un único archivo de vídeo.
    """
    logging.info("-" * 50)
    logging.info(f"Procesando: {ruta_video.name}")

    try:
        with VideoFileClip(str(ruta_video)) as clip:
            duracion_video = clip.duration
        
        texto_inicio = transcribir_clip(ruta_video, gemini_model, 0, min(20, duracion_video), temp_dir)
        if not texto_inicio: return

        nombre_pelicula = obtener_nombre_pelicula(texto_inicio, gemini_model)
        if not nombre_pelicula or nombre_pelicula in ["nombre_desconocido", "pelicula_no_encontrada"]:
            logging.warning(f"No se pudo obtener un nombre de película válido. Respuesta: '{nombre_pelicula}'.")
            
            # --- NUEVA LÓGICA DE MOVIMIENTO ROBUSTA ---
            error_dir = destination_dir / "error"
            os.makedirs(error_dir, exist_ok=True)
            error_destination = error_dir / ruta_video.name
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Intentando mover archivo no reconocido a '{error_destination}' (Intento {attempt + 1})...")
                    shutil.move(str(ruta_video), str(error_destination))
                    logging.info(f"Archivo movido a la carpeta de errores: '{error_destination}'")
                    return # Salir de la función si el movimiento tiene éxito
                except Exception as move_error:
                    if attempt < max_retries - 1:
                        logging.warning(f"No se pudo mover el archivo (reintentando en 2 segundos): {move_error}")
                        time.sleep(2)
                    else:
                        logging.error(f"FALLO DEFINITIVO al mover el archivo '{ruta_video.name}' a la carpeta de errores: {move_error}", exc_info=True)
                        return # Salir de la función después del último intento fallido
            return
            
        logging.info(f"Película identificada: '{nombre_pelicula}'")

        texto_final = transcribir_clip(ruta_video, gemini_model, max(0, duracion_video - 20), duracion_video, temp_dir)
        if not texto_final: return
        logging.info(f"Texto extraído (final): \"{texto_final.strip()}\"")
        
        puntuacion = obtener_puntuacion(texto_final, gemini_model)
        logging.info(f"Puntuación identificada: '{puntuacion}'")

        nombre_puntuacion_seguro = puntuacion.replace('.', '_')
        nuevo_nombre_base = f"{nombre_pelicula}_puntos_{nombre_puntuacion_seguro}"
        
        nueva_ruta_video = obtener_ruta_unica(ruta_video.with_stem(nuevo_nombre_base))
        ruta_video.rename(nueva_ruta_video)
        logging.info(f"Vídeo renombrado a: '{nueva_ruta_video.name}'")

        try:
            final_destination = destination_dir / nueva_ruta_video.name
            shutil.move(str(nueva_ruta_video), str(final_destination))
            logging.info(f"Archivo movido a: '{final_destination}'")
        except Exception as move_error:
            logging.error(f"No se pudo mover el archivo '{nueva_ruta_video.name}' a '{destination_dir}': {move_error}", exc_info=True)

    except Exception as e:
        logging.error(f"Ha ocurrido un error inesperado durante el procesamiento de '{ruta_video.name}': {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Renombra vídeos de críticas de cine usando IA.")
    parser.add_argument("archivo", nargs='?', default=None, type=str, help="Ruta opcional a un archivo de vídeo específico.")
    args = parser.parse_args()

    try:
        base_dir = Path(os.environ["VIDEO_PROCESSING_DIR"])
    except KeyError:
        logging.error("La variable de entorno VIDEO_PROCESSING_DIR no está configurada.")
        return

    try:
        logging.info("Inicializando el modelo de Gemini...")
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logging.error(f"No se pudo inicializar el modelo de Gemini: {e}", exc_info=True)
        return
    
    temp_dir = None
    
    try:
        if args.archivo:
            ruta_archivo = Path(args.archivo)
            if not ruta_archivo.is_file():
                logging.error(f"La ruta especificada no es un archivo válido: {args.archivo}")
                return
            
            temp_dir = ruta_archivo.parent / "temp"
            os.makedirs(temp_dir, exist_ok=True)
            procesar_un_video(ruta_archivo, gemini_model, temp_dir, base_dir)

        else:
            upload_dir = base_dir / "upload"
            if not upload_dir.is_dir():
                logging.error(f"El directorio de subida no existe: {upload_dir}")
                return

            temp_dir = base_dir / "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            videos_a_procesar = list(upload_dir.glob("*.mp4")) + list(upload_dir.glob("*.mov"))
            if not videos_a_procesar:
                logging.info(f"No se encontraron vídeos en {upload_dir}.")
                return
            
            logging.info(f"Encontrados {len(videos_a_procesar)} vídeos para procesar en {upload_dir}.")
            for ruta_video in videos_a_procesar:
                procesar_un_video(ruta_video, gemini_model, temp_dir, base_dir)

    finally:
        # Limpieza final del directorio temporal
        if temp_dir and temp_dir.exists():
            logging.info(f"Limpiando directorio temporal: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
