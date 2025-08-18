import google.generativeai as genai

# Configura la clave de API
genai.configure(api_key="AIzaSyDb9lvU-j9NYxqv_WolRd0pdOylnHvUjlQ")

# Crea una instancia del modelo
model = genai.GenerativeModel('gemini-2.5-flash')

# Genera contenido
response = model.generate_content("¿Cómo funciona la inteligencia artificial?")
print(response.text)
