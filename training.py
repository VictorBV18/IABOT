# Importación de bibliotecas necesarias
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Inicialización del lematizador de palabras
lemmatizer = WordNetLemmatizer()

# Carga de datos de intenciones desde el archivo JSON
intents = json.loads(open('intents.json', encoding='utf-8').read())


# Inicialización de listas y variables
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Procesamiento de intenciones y patrones
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenización de patrones en palabras
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # Creación de documentos con patrones y tags de intenciones
        documents.append((wordList, intent['tag']))
        # Actualización de clases si no existe en la lista
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematización y filtrado de palabras
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Guarda palabras y clases en archivos pickle para uso futuro
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Creación de datos de entrenamiento usando la técnica de "bolsa de palabras"
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    # Creación de vector de salida correspondiente a la clase
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

# Mezcla aleatoria de los datos de entrenamiento
random.shuffle(training)
training = np.array(training)

# División de datos de entrada y salida
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Definición del modelo de red neuronal
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(600, input_shape=(len(trainX[0]),), activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.6))
model.add(tf.keras.layers.Dense(300, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.6))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Configuración del modelo para el entrenamiento
sgd = tf.keras.optimizers.SGD(learning_rate=0.50, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(trainX, trainY, epochs=2000, batch_size=300, verbose=1)

# Guarda el modelo entrenado en un archivo
model.save('chatbot_model.keras')

# Imprime un mensaje indicando la finalización del proceso
print('Done')
