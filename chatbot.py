# Importando librerías necesarias
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Inicializando el lematizador
lemmatizer = WordNetLemmatizer()

# Cargando datos del archivo JSON que contiene patrones de conversación
intents = json.loads(open('intents.json', encoding='utf-8').read())

# Cargando datos previamente guardados de palabras y clases
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Cargando el modelo del chatbot
model = load_model("chatbot_model.keras")

# Función para limpiar la oración y lematizar las palabras
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Función para convertir una oración en un vector de palabras
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Función para predecir la clase de una oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Función para obtener una respuesta aleatoria de acuerdo a la clase
def get_response(tag, intents_json):
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

# Bucle principal del chatbot
while True:
    message = input(" ")  # Solicita al usuario que ingrese un mensaje
    ints = predict_class(message)  # Predice la clase del mensaje
    tag = ints[0]['intent']  # Obtiene la etiqueta de la clase predicha
    if tag == "salir":
        print("Hasta luego.")
        break  # Sale del bucle si la etiqueta es "salir"
    res = get_response(tag, intents)  # Obtiene una respuesta basada en la clase
    print(res)  # Imprime la respuesta
