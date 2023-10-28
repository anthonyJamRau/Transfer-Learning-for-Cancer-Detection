import datetime
import os
from typing import Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
from cleanvision import Imagelab
from matplotlib import ticker
from PIL import Image
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow.keras import datasets, layers, losses, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Models:
    def __init__(self) -> None:
        self.AlexNet_x_test = None
        self.AlexNet_y_test = None
        self.AlexNetBreaKHis_test = None
        self.VGGNet_imagenet_path = f"VGGNet_ImageNet_Model.pkl"
        self.AlexNet_mnist_path = f"AlexNet_MNIST_Model.pkl"

    def build_VGGNet_imagenet(self):
        model = VGG19(weights="imagenet")
        joblib.dump(model, self.VGGNet_imagenet_path)

    def build_AlexNet_mnist(self, visualize_training=False):
        (x_train, y_train), (
            x_test,
            y_test,
        ) = datasets.mnist.load_data()
        x_train = x_train[:6000]
        y_train = y_train[:6000]
        x_test = x_test[:1000]
        y_test = y_test[:1000]
        x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
        x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
        x_train = tf.expand_dims(x_train, axis=3, name=None)
        x_test = tf.expand_dims(x_test, axis=3, name=None)
        x_train = tf.repeat(x_train, 3, axis=3)
        x_test = tf.repeat(x_test, 3, axis=3)
        x_val = x_train[-2000:, :, :, :]
        y_val = y_train[-2000:]
        x_train = x_train[:-2000, :, :, :]
        y_train = y_train[:-2000]

        model = models.Sequential()
        model.add(
            layers.experimental.preprocessing.Resizing(
                224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]
            )
        )
        model.add(layers.Conv2D(96, 11, strides=4, padding="same"))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(256, 5, strides=4, padding="same"))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(384, 3, strides=4, padding="same"))
        model.add(layers.MaxPooling2D(2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation="softmax"))
        model.summary()
        model.compile(
            optimizer="adam",
            loss=losses.sparse_categorical_crossentropy,
            metrics=["accuracy"],
        )
        history = model.fit(
            x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val)
        )

        model.save(self.AlexNet_mnist_path)

        if visualize_training:
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            axs[0].plot(history.history["loss"])
            axs[0].plot(history.history["val_loss"])
            axs[0].title.set_text("Training Loss vs Validation Loss")
            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Loss")
            axs[0].legend(["Train", "Val"])
            axs[1].plot(history.history["accuracy"])
            axs[1].plot(history.history["val_accuracy"])
            axs[1].title.set_text("Training Accuracy vs Validation Accuracy")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].legend(["Train", "Val"])
            plt.show()

    def build_AlexNet_breakhis(self, visualize_training=False):
        SEED = 51432
        tf.keras.utils.set_random_seed(SEED)
        tf.config.experimental.enable_op_determinism()
        fold_info = pd.read_csv(
            "/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/models/data/BreaKHis_v1/Folds.csv"
        )
        fold_info["label"] = fold_info["filename"].str.extract("(malignant|benign)")
        train = fold_info.query("grp == 'train'")
        test = fold_info.query("grp == 'test'")
        classes = dict(benign=0, malignant=1)
        y = train["label"].map(classes)
        IMG_SIZE = 224
        BATCH_SIZE = 28

        def load_image(filename: str, label: int) -> Tuple[tf.Tensor, str]:
            file = tf.io.read_file("data/" + filename)
            img = tf.image.decode_png(file, channels=3)
            img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
            return img, label

        # Prepare training and validation datasets
        X_train, X_valid, y_train, y_valid = train_test_split(
            train["filename"], train["label"].map(classes), random_state=SEED
        )
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .map(load_image)
            .batch(BATCH_SIZE)
        )
        validation_ds = (
            tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
            .map(load_image)
            .batch(BATCH_SIZE)
        )
        # Prepare test dataset
        test = test.sample(frac=1, random_state=SEED)  # shuffle test data
        test_ds = (
            tf.data.Dataset.from_tensor_slices(
                (test["filename"], test["label"].map(classes))
            )
            .map(load_image)
            .batch(BATCH_SIZE)
        )
        self.AlexNetBreaKHis_test = test_ds

        # Cache and prefetch data for faster training
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        MAX_EPOCHS = 25
        BASE_LEARNING_RATE = 0.001

        # ACTUAL ALEXNET PART
        model = models.Sequential()

        model.add(
            layers.experimental.preprocessing.Resizing(
                224,
                224,
                interpolation="bilinear",
                input_shape=(
                    IMG_SIZE,
                    IMG_SIZE,
                    3,
                ),
            )
        )
        model.add(layers.Conv2D(96, 11, strides=4, padding="same"))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(256, 5, strides=3, padding="same"))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(384, 3, strides=4, padding="same"))
        model.add(layers.MaxPooling2D(2, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation="softmax"))
        model.summary()

        print("Compiling...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(BASE_LEARNING_RATE),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="roc_auc"), "binary_accuracy"],
        )
        print("Finished compiling.")
        early_stopping = EarlyStopping(
            min_delta=1e-4, patience=5, verbose=1, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)
        print("Fitting...")
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=MAX_EPOCHS,
            callbacks=[early_stopping, reduce_lr],
        )
        print("Finished fitting.")

        model.save("AlexNet_BreaKHis.pkl")

        if visualize_training:
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            axs[0].plot(history.history["loss"])
            axs[0].plot(history.history["val_loss"])
            axs[0].title.set_text("Training Loss vs Validation Loss")
            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Loss")
            axs[0].legend(["Train", "Val"])
            axs[1].plot(history.history["accuracy"])
            axs[1].plot(history.history["val_accuracy"])
            axs[1].title.set_text("Training Accuracy vs Validation Accuracy")
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Accuracy")
            axs[1].legend(["Train", "Val"])
            plt.show()

    def load_model(self, path):
        if "AlexNet_BreaKHis" in path:
            SEED = 51432
            tf.keras.utils.set_random_seed(SEED)
            tf.config.experimental.enable_op_determinism()
            fold_info = pd.read_csv(
                "/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/models/data/BreaKHis_v1/Folds.csv"
            )
            fold_info["label"] = fold_info["filename"].str.extract("(malignant|benign)")
            train = fold_info.query("grp == 'train'")
            test = fold_info.query("grp == 'test'")
            classes = dict(benign=0, malignant=1)
            y = train["label"].map(classes)
            IMG_SIZE = 224
            BATCH_SIZE = 28

            def load_image(filename: str, label: int) -> Tuple[tf.Tensor, str]:
                file = tf.io.read_file("data/" + filename)
                img = tf.image.decode_png(file, channels=3)
                img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
                return img, label

            # Prepare training and validation datasets
            # X_train, X_valid, y_train, y_valid = train_test_split(
            #     train["filename"], train["label"].map(classes), random_state=SEED
            # )
            # Prepare test dataset
            test = test.sample(frac=1, random_state=SEED)  # shuffle test data
            test_ds = (
                tf.data.Dataset.from_tensor_slices(
                    (test["filename"], test["label"].map(classes))
                )
                .map(load_image)
                .batch(BATCH_SIZE)
            )
            self.AlexNetBreaKHis_test = test_ds
            return load_model(path)
        elif "Alex" in path:
            (_, _), (
                x_test,
                y_test,
            ) = datasets.mnist.load_data()
            x_test = x_test[:1000]
            y_test = y_test[:1000]
            x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
            x_test = tf.expand_dims(x_test, axis=3, name=None)
            x_test = tf.repeat(x_test, 3, axis=3)
            self.AlexNet_x_test = x_test
            self.AlexNet_y_test = y_test
            return load_model(path)
        else:
            return joblib.load(path)
