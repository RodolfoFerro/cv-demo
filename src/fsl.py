"""Few-Shot Learning module."""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda


def euclidean_distance(vectors):
    """Function to compute the euclidean distance from given input vectors.

    Parameters
    ----------
    vectors : tf.Tensor
        A pair of vectors to compute the distance.

    Returns
    -------
    distance : tf.Tensor
        The result of the distance as a tensor.
    """

    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    distance = K.sqrt(K.maximum(sum_square, K.epsilon()))

    return distance


def build_siamese_network(input_shape=(256, 256, 1)):
    """Function to build the Siamese Network for FSL task.

    The shared network creates a 64-dimensional embedding of an input image.

    Parameters
    ----------
    input_shape : tuple or list
        The dimensions of the image input. Must be 3-dimensional as it should
        include the number of channels (e.g. (28, 28, 1)).

    Returns
    -------
    siamese_net : tf.models.Model
        A TensorFlow model ready to be compiled and trained.
    """

    # Input data
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Shared subnetwork
    shared_network = Sequential([
        # 1st conv layer
        Input(input_shape),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # 2nd conv layer
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Create a 64d embedding
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu")
    ])

    # Generate the embeddings from inputs
    embedding_a = shared_network(input_a)
    embedding_b = shared_network(input_b)

    # Compute the Euclidean distance between embeddings
    distance = Lambda(euclidean_distance)([embedding_a, embedding_b])

    # Compute the similarity score
    prediction = Dense(1, activation="sigmoid")(distance)

    # Build model
    siamese_net = Model(inputs=[input_a, input_b], outputs=prediction)

    return siamese_net


if __name__ == "__main__":
    model = build_siamese_network((28, 28, 1))
    model.summary()
