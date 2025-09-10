"""Dataset utilities module."""

import random
import os

import numpy as np
import cv2


def prepare_image(img_path, target_size=(256, 256), operation="dilate"):
    """Prepare an image for model inference/training.

    The function applies several subprocesses using openCV:
        - Loads an image and converts it to grayscale from a given path.
        - Binarizes the image using an Otsu threshold.
        - Applies a morphological transformation (erosion, dilation or none).
        - Resizes the image to a given size.
        - Adds the dimension channel (1).
        - Normalizes the image to range [0, 1].

    Parameters
    ----------
    filepath : str
        Path to the image.
    target_size : tuple
        Final size of the image (w, h).
    operation : str
        Whether to apply a morphological operation. It takes values
        "erode", "dilate" or None.

    Returns
    -------
    img_proc : np.ndarray
        Processed image with shape (h, w, 1).
    """

    # Load and convert to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"[ERROR] Could not load image: {img_path}")

    # Binarize image with Otsu filter
    _, img_bin = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create a kernel for morphological operations
    kernel = np.ones((1, 1), np.uint8)

    # Apply morphological operations
    if operation == "erode":
        img_proc = cv2.erode(img_bin, kernel, iterations=2)
    elif operation == "dilate":
        img_proc = cv2.dilate(img_bin, kernel, iterations=2)
    else:
        img_proc = img_bin

    # Resize image
    img_proc = cv2.resize(img_proc, target_size)

    # Add channel
    img_proc = np.expand_dims(img_proc, axis=-1)

    # Normalize to [0,1]
    img_proc = img_proc.astype(np.float32) / 255.0

    return img_proc


def load_dataset_sample(dataset_dir,
                        n_samples=100,
                        target_size=(256, 256),
                        operation=None):
    """Function to load a dataset sample from image folders.

    This funciton uses OpenCV for data loading and preparation.

    Parameters
    ----------
    dataset_dir :str
        The path to the dataset folder.
    n_samples : int
        The total number of samples to take from folders (balanced between
        classes).
    target_size : tuple
        The reshape size of the images (w, h).

    Returns
    -------
    images : np.ndarray
        A numpy array containing images with shape (n, h, w, c).
    labels : np.ndarray
        A numpy array containing integer labels with shape (n,).
    class_names : list
        A list of names of the given classes.
    """

    images = []
    labels = []
    class_names = sorted([
        c for c in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, c))
    ])

    num_classes = len(class_names)
    samples_per_class = max(1, n_samples // num_classes)  # imágenes por clase

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        file_list = [
            os.path.join(class_dir, f) for f in os.listdir(class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # If there are less images than samples per class, we take all
        sampled_files = random.sample(file_list,
                                      min(samples_per_class, len(file_list)))

        for img_path in sampled_files:
            img = prepare_image(img_path,
                                target_size=target_size,
                                operation=operation)

            images.append(img)
            labels.append(idx)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, class_names


def make_pairs(images, labels):
    """Function to build labeled pairs of images for training.

    Parameters
    ----------
    images : np.array
        A numpy array consisting of loaded and transformed images.
    labels : np.array
        A numpy array consisting of corresponding labels.

    Returns
    -------
    pairs : tuple
        A 2-tuple of prepared image pairs and labels.
    """

    pair_images = []
    pair_labels = []

    # Calculate the total number of classes and then build a list of
    # indices for each class
    n_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, n_classes)]

    for idx_A in range(len(images)):
        # Grab current image
        current_image = images[idx_A]
        label = labels[idx_A]

        # Randomly pick an image that belongs to the same class
        idx_B = np.random.choice(idx[label])
        positive_image = images[idx_B]

        # Prepare a positive pair and update the images and labels
        pair_images.append([current_image, positive_image])
        pair_labels.append([1])

        # Grab the indices for each of the class labels not equal to
        # the current label and randomly pick an image corresponding
        # to a label from different class
        negatve_idx = np.where(labels != label)[0]
        negative_image = images[np.random.choice(negatve_idx)]

        # Prepare a negative pair of images and update our lists
        pair_images.append([current_image, negative_image])
        pair_labels.append([0])

    pairs = (np.array(pair_images), np.array(pair_labels))

    return pairs
