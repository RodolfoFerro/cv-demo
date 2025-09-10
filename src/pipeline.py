"""Pipeline module for inference."""

from glob import glob

import numpy as np

from src.dataset import prepare_image


def load_query_images(query_folder):
    """Function to load the query data folder for one-shot learning task.

    Paramaters
    ----------
    query_folder : str
        The query folder.

    Returns
    -------
    query_images : list
        A list with the query images.
    class_names : list
        A list with the class names of images.
    """

    # Load query images
    query_images = []
    class_names = []
    for img_path in glob(query_folder):
        class_name = img_path.split("/")[-1][:-4]
        img = prepare_image(img_path, target_size=(256, 256))
        img = np.expand_dims(img, axis=0)
        query_images.append(img)
        class_names.append(class_name)
    return query_images, class_names


def inference_pipeline(model, input_image, query_images):
    """Function to run the inference pipe over the query images in the
    one-shot learning task.

    Paramaters
    ----------
    model : tf.model.Model
        The loaded model for one-shot learning task.
    input_image : np.array
        A prepared image ready for inference.
    query_images : list
        A list with the query images.

    Returns
    -------
    scores : list
        A list with resulting scores of inference.
    """

    scores = []
    for image in query_images:
        score = model.predict([input_image, image])
        scores.append(score[0][0])
    return scores
