import os
import whisper
import re
import google.generativeai as genai
from moviepy.editor import VideoFileClip
import tempfile
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

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
    hasta encontrar un nombre de archivo único.
    Ejemplo: 'Dune.mp4' -> 'Dune_2.mp4'
    """
    if not ruta_propuesta.exists():
        return ruta_propuesta

    contador = 2
    while True:
        nueva_ruta = ruta_propuesta.with_stem(f"{ruta_propuesta.stem}_{contador}")
        if not nueva_ruta.exists():
            return nueva_ruta
        contador += 1

# --- Funciones Principales ---

def transcribir_video(video_path: Path, model) -> str | None:
    """
    Extrae los primeros 20 segundos de un vídeo, los transcribe y devuelve el texto.
    Reutiliza el modelo Whisper ya cargado.
    """
    try:
        logging.info(f"Extrayendo audio de '{video_path.name}'...")
        # Crear un archivo temporal para el clip de audio/video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
            with VideoFileClip(str(video_path)) as clip:
                # Extraer los primeros 20 segundos o la duración total si es más corto
                subclip_duration = min(20, clip.duration)
                clip_sub = clip.subclip(0, subclip_duration)
                clip_sub.write_videofile(temp_file.name, audio_codec="aac", verbose=False, logger=None)
            
            logging.info("Transcribiendo audio con Whisper...")
            result = model.transcribe(temp_file.name, language="es")
            return result["text"]
    except Exception as e:
        logging.error(f"No se pudo transcribir el vídeo '{video_path.name}': {e}")
        return None

def obtener_nombre_pelicula(transcripcion: str, gemini_model) -> str | None:
    """
    Usa Gemini para extraer el nombre de la película de un texto.
    """
    if not transcripcion:
        return None
    
    prompt = (
        "De la siguiente transcripción de una crítica de cine, extrae y devuelve ÚNICAMENTE el nombre "
        "de la película principal. No incluyas artículos (el, la, los, las), comillas, ni ninguna "
        "otra palabra. Solo el título.\n\n"
        f"Transcripción: \"{transcripcion}\""
    )
    try:
        logging.info("Consultando a Gemini para obtener el nombre de la película...")
        response = gemini_model.generate_content(prompt)
        # Limpiar el nombre de caracteres no válidos para nombres de archivo usando una expresión regular.
        # Esto elimina caracteres como / \ : * ? " < > |
        nombre_sucio = response.text.strip()
        nombre_limpio = re.sub(r'[\\/*?:"<>|]', "", nombre_sucio)
        return nombre_limpio if nombre_limpio else "nombre_desconocido"
    except Exception as e:
        logging.error(f"Error al contactar con la API de Gemini: {e}")
        return "nombre_desconocido"

def procesar_videos(carpeta_videos: Path, modelo_whisper: str):
    """
    Función principal que orquesta el proceso para todos los vídeos en una carpeta.
    """
    # Cargar modelos UNA SOLA VEZ fuera del bucle
    logging.info(f"Cargando el modelo Whisper '{modelo_whisper}'... (puede tardar un momento)")
    try:
        whisper_model = whisper.load_model(modelo_whisper)
    except Exception as e:
        logging.error(f"No se pudo cargar el modelo Whisper: {e}")
        return

    # *** CORRECCIÓN: Inicializar el modelo de Gemini aquí ***
    logging.info("Inicializando el modelo de Gemini...")
    try:
        gemini_model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        logging.error(f"No se pudo inicializar el modelo de Gemini: {e}")
        return

    videos_encontrados = list(carpeta_videos.glob("*.mp4")) + \
                         list(carpeta_videos.glob("*.mov")) + \
                         list(carpeta_videos.glob("*.avi")) + \
                         list(carpeta_videos.glob("*.mkv"))

    if not videos_encontrados:
        logging.warning(f"No se encontraron vídeos en la carpeta '{carpeta_videos}'")
        return

    logging.info(f"Se encontraron {len(videos_encontrados)} vídeos. Iniciando procesamiento...")

    for ruta_video_actual in videos_encontrados:
        logging.info("-" * 50)
        logging.info(f"Procesando: {ruta_video_actual.name}")

        texto_transcrito = transcribir_video(ruta_video_actual, whisper_model)
        if not texto_transcrito:
            continue

        nombre_pelicula = obtener_nombre_pelicula(texto_transcrito, gemini_model)
        if not nombre_pelicula or nombre_pelicula == "nombre_desconocido":
            logging.warning(f"No se pudo obtener el nombre de la película para '{ruta_video_actual.name}'. Saltando archivo.")
            continue

        logging.info(f"Película identificada: '{nombre_pelicula}'")

        # Renombrar el vídeo
        try:
            nueva_ruta_video = obtener_ruta_unica(ruta_video_actual.with_stem(nombre_pelicula))
            ruta_video_actual.rename(nueva_ruta_video)
            logging.info(f"Vídeo renombrado a: '{nueva_ruta_video.name}'")
        except Exception as e:
            logging.error(f"No se pudo renombrar el vídeo '{ruta_video_actual.name}': {e}")
            continue

        # Guardar la transcripción
        try:
            ruta_txt = obtener_ruta_unica(nueva_ruta_video.with_suffix(".txt"))
            with open(ruta_txt, "w", encoding="utf-8") as f:
                f.write(texto_transcrito)
            logging.info(f"Transcripción guardada en: '{ruta_txt.name}'")
        except Exception as e:
            logging.error(f"No se pudo guardar la transcripción para '{nombre_pelicula}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Renombra vídeos de críticas de cine usando IA.")
    # --- CAMBIO AQUÍ ---
    # Se convierte 'carpeta' en un argumento opcional con un valor por defecto.
    parser.add_argument("--carpeta", type=str, default="Z:\\criticas", 
                        help="La ruta a la carpeta que contiene los vídeos. Por defecto es 'Z:\\criticas'.")
    parser.add_argument("--modelo", type=str, default="tiny", choices=["tiny", "base", "small", "medium", "large"],
                        help="El modelo de Whisper a utilizar (ej. 'tiny', 'base').")
    args = parser.parse_args()

    carpeta_videos = Path(args.carpeta)
    if not carpeta_videos.is_dir():
        logging.error(f"La ruta especificada no es una carpeta válida: {args.carpeta}")
        return

    procesar_videos(carpeta_videos, args.modelo)

if __name__ == "__main__":
    main()
