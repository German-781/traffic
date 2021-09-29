import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

# categorias small 3, total 43

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):

    print("load data ")

    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.


    Cargue los datos de la imagen desde el directorio `data_dir`.

    Suponga que `data_dir` tiene un directorio con el nombre de cada categoría, numerado
    0 a NUM_CATEGORIES - 1. Dentro de cada directorio de categorías habrá algunos
    número de archivos de imagen.

    Devuelve la tupla `(imágenes, etiquetas)`. `images` debe ser una lista de todos
    de las imágenes en el directorio de datos, donde cada imagen se formatea como un
    ndarray numpy con dimensiones IMG_WIDTH x IMG_HEIGHT x 3. Las `etiquetas` deben
    ser una lista de etiquetas enteras, que representan las categorías para cada una de las
    correspondientes "imágenes".

    """""

    imagenes = []
    etiquetas = []


    for dir in range(0, NUM_CATEGORIES):
        dire = os.path.join(data_dir, str(dir))
        for ruta in os.listdir(dire):
            ruta_final = os.path.join(data_dir, str(dir), ruta)

            imagen = cv2.imread(ruta_final)
            imagen_ajustada = cv2.resize(imagen, (IMG_WIDTH, IMG_HEIGHT))
            #print("imagen AJUSTADA ", imagen_ajustada.shape)
 
            imagenes.append(imagen_ajustada)
            etiquetas.append(dir)

    return (imagenes, etiquetas)


    #raise NotImplementedError


def get_model():

    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.

    ***************

    Devuelve un modelo de red neuronal convolucional compilado. Suponga que el
    `input_shape` de la primera capa es` (IMG_WIDTH, IMG_HEIGHT, 3) `.
    La capa de salida debe tener `NUM_CATEGORIES` unidades, una para cada categoría.

    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (IMG_WIDTH, IMG_HEIGHT,3)),        
        tf.keras.layers.ReLU(name = "relu1"),        
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.ReLU(name = "relu2"),        
        tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
        tf.keras.layers.ReLU(name = "relu3"),        

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])

    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return model

    
    #raise NotImplementedError


if __name__ == "__main__":
    main()
