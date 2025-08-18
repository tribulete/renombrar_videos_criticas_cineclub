import os
import re
import google.generativeai as genai
from moviepy.editor import VideoFileClip
import tempfile
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
import shutil
import time

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración ---
# Configura un logger para mostrar información de manera más clara
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configura tu clave de Gemini. Es más seguro leerla de una variable de entorno.
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logging.error("La variable de entorno GEMINI_API_KEY no está configurada. Asegúrate de crear un archivo .env con tu clave.")
    exit()

# --- Funciones Auxiliares ---

def obtener_ruta_unica(ruta_propuesta: Path) -> Path:
    """
    Verifica si una ruta existe. Si es así, añade un sufijo numérico
    entre paréntesis hasta encontrar un nombre de archivo único.
    Ejemplo: 'Dune.mp4' -> 'Dune(2).mp4'
    """
    if not ruta_propuesta.exists():
        return ruta_propuesta

    contador = 2
    while True:
        # --- CAMBIO AQUÍ ---
        # Se usa el formato (2), (3), etc. para los duplicados.
        nueva_ruta = ruta_propuesta.with_stem(f"{ruta_propuesta.stem}({contador})")
        if not nueva_ruta.exists():
            return nueva_ruta
        contador += 1

# --- Funciones Principales de IA ---

def transcribir_clip(video_path: Path, gemini_model, start_time: float, end_time: float) -> Optional[str]:
    """
    Extrae un clip de un vídeo, lo sube y lo transcribe usando Gemini.
    """
    temp_dir = None
    audio_file = None
    try:
        logging.info(f"Extrayendo clip de audio de '{video_path.name}' desde {start_time}s hasta {end_time}s...")
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_clip.mp4")

        with VideoFileClip(str(video_path)) as clip:
            clip_sub = clip.subclip(start_time, end_time)
            clip_sub.write_videofile(temp_path, audio_codec="aac", verbose=False, logger=None)
        
        logging.info("Subiendo archivo de audio para transcripción en la nube...")
        audio_file = genai.upload_file(path=temp_path)
        
        logging.info(f"Esperando a que el archivo '{audio_file.name}' esté activo...")
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(name=audio_file.name)

        if audio_file.state.name != "ACTIVE":
            raise Exception(f"El procesamiento del archivo falló: {audio_file.state.name}")

        logging.info("El archivo está activo. Transcribiendo audio con Gemini...")
        response = gemini_model.generate_content(["Por favor, transcribe este audio en español.", audio_file])
        
        return response.text

    except Exception as e:
        logging.error(f"No se pudo transcribir el clip de vídeo con Gemini: {e}", exc_info=True)
        return None
    finally:
        if audio_file:
            try:
                logging.info(f"Eliminando archivo temporal de la nube: {audio_file.name}")
                genai.delete_file(audio_file.name)
            except Exception as delete_e:
                logging.error(f"Error al eliminar el archivo de la nube: {delete_e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def obtener_nombre_pelicula(transcripcion: str, gemini_model) -> Optional[str]:
    """
    Usa Gemini para extraer el nombre de la película de un texto y verificar su existencia.
    """
    if not transcripcion: return None
    
    prompt = (
        "Analiza la siguiente transcripción. Tu tarea es extraer el nombre de la película principal y verificar que existe.\n"
        "- Si la película existe y estás 100% seguro, devuelve ÚNICAMENTE su título. No incluyas artículos ni comillas.\n"
        "- Si no estás seguro o la película no parece existir, DEBES devolver la frase exacta: pelicula_no_encontrada\n\n"
        f"Transcripción: \"{transcripcion}\""
    )
    try:
        logging.info("Consultando a Gemini para obtener el nombre de la película...")
        response = gemini_model.generate_content(prompt)
        nombre_sucio = response.text.strip()
        nombre_limpio = re.sub(r'[\\/*?:"<>|]', "", nombre_sucio)
        return nombre_limpio if nombre_limpio else "nombre_desconocido"
    except Exception as e:
        logging.error(f"Error al contactar con la API de Gemini para el título: {e}", exc_info=True)
        return "nombre_desconocido"

def obtener_puntuacion(transcripcion: str, gemini_model) -> str:
    """
    Usa Gemini para extraer la puntuación de un texto.
    """
    if not transcripcion: return "no"

    prompt = (
        "Analiza la siguiente transcripción. Tu tarea es extraer la puntuación numérica (de 0 a 10) que se le da a la película.\n"
        "- Si encuentras una puntuación clara (incluyendo decimales o frases como 'y medio'), devuelve ÚNICAMENTE el número. Usa un punto como separador decimal (ej: '8', '9.5').\n"
        "- Si no se menciona ninguna puntuación, DEBES devolver la palabra exacta: no\n\n"
        f"Transcripción: \"{transcripcion}\""
    )
    try:
        logging.info("Consultando a Gemini para obtener la puntuación...")
        response = gemini_model.generate_content(prompt)
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

def procesar_video(ruta_video: Path):
    """
    Función principal que orquesta el proceso para un único vídeo.
    """
    logging.info("Inicializando el modelo de Gemini...")
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logging.error(f"No se pudo inicializar el modelo de Gemini: {e}", exc_info=True)
        return

    logging.info("-" * 50)
    logging.info(f"Procesando: {ruta_video.name}")

    # 1. Transcribir el inicio para obtener el título
    duracion_video = VideoFileClip(str(ruta_video)).duration
    texto_inicio = transcribir_clip(ruta_video, gemini_model, 0, min(20, duracion_video))
    if not texto_inicio: return
    logging.info(f"Texto extraído (inicio): \"{texto_inicio.strip()}\"")

    nombre_pelicula = obtener_nombre_pelicula(texto_inicio, gemini_model)
    if not nombre_pelicula or nombre_pelicula in ["nombre_desconocido", "pelicula_no_encontrada"]:
        logging.warning(f"No se pudo obtener un nombre de película válido. Respuesta: '{nombre_pelicula}'. No se renombrará.")
        return
    logging.info(f"Película identificada: '{nombre_pelicula}'")

    # 2. Transcribir el final para obtener la puntuación
    texto_final = transcribir_clip(ruta_video, gemini_model, max(0, duracion_video - 20), duracion_video)
    if not texto_final: return
    logging.info(f"Texto extraído (final): \"{texto_final.strip()}\"")
    
    puntuacion = obtener_puntuacion(texto_final, gemini_model)
    logging.info(f"Puntuación identificada: '{puntuacion}'")

    # 3. Renombrar el archivo
    nombre_puntuacion_seguro = puntuacion.replace('.', '_')
    nuevo_nombre_base = f"{nombre_pelicula}_puntos_{nombre_puntuacion_seguro}"
    try:
        nueva_ruta_video = obtener_ruta_unica(ruta_video.with_stem(nuevo_nombre_base))
        ruta_video.rename(nueva_ruta_video)
        logging.info(f"Vídeo renombrado a: '{nueva_ruta_video.name}'")
    except Exception as e:
        logging.error(f"No se pudo renombrar el vídeo: {e}", exc_info=True)
        return

def main():
    parser = argparse.ArgumentParser(description="Renombra un vídeo de crítica de cine usando IA.")
    parser.add_argument("archivo", type=str, 
                        help="La ruta completa al archivo de vídeo a procesar.")
    args = parser.parse_args()

    ruta_archivo = Path(args.archivo)
    if not ruta_archivo.is_file():
        logging.error(f"La ruta especificada no es un archivo válido: {args.archivo}")
        return

    procesar_video(ruta_archivo)

if __name__ == "__main__":
    main()
