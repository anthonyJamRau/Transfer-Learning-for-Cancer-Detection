import zipfile
from typing import Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, losses, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.model_selection import GroupShuffleSplit
import os
import keras_tuner as kt

# Prepares data to be used in testing the models.
SEED = 51432
IMG_SIZE = 224
BATCH_SIZE = 28
MAX_EPOCHS = 20
BASE_LEARNING_RATE = 0.001
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# read info file
fold_df = pd.read_csv("Folds.csv", dtype={"mag": "string"})

# reduce the info file for the first fold (folds 2-5 are just duplicate images of fold 1 - we only have 7909 unique values)
fold_df = fold_df[fold_df["fold"] == 1]

# create informative filename, target class, encoded class, id field for splitting
fold_df["class"] = fold_df["filename"].apply(lambda x: x.split("/")[3])
fold_df["sub_class"] = fold_df["filename"].apply(lambda x: x.split("/")[5])
fold_df["patient_id"] = fold_df["filename"].apply(lambda x: x.split("/")[-1])
cols = ["mag", "class", "sub_class", "patient_id"]
fold_df["input_path"] = fold_df[cols].apply(
    lambda row: "_".join(row.values.astype(str)), axis=1
)
fold_df["encoded_class"] = fold_df["class"].apply(lambda x: 0 if x == "benign" else 1)

# make train and test set
splitter = GroupShuffleSplit(test_size=0.20, n_splits=2, random_state=7)
split = splitter.split(fold_df, groups=fold_df["patient_id"])
train_inds, test_inds = next(split)
train = fold_df.iloc[train_inds].reset_index(drop=True)
temp_test = fold_df.iloc[test_inds].reset_index(drop=True)

# make validation set from test set
splitter_2 = GroupShuffleSplit(test_size=0.50, n_splits=2, random_state=8)
split_2 = splitter_2.split(temp_test, groups=temp_test["patient_id"])
test_inds, validation_inds = next(split_2)
test = temp_test.iloc[test_inds].reset_index(drop=True)
validation = temp_test.iloc[validation_inds].reset_index(drop=True)

# intitialize series for tf
# pulling in data from 2 directories. One I created with Carson's normalized pngs, another from augmented images generated to Augmented_train
# X_train = pd.concat(
#     [
#         "Normalized/" + train["input_path"],
#         "Augmented_train/" + pd.Series(os.listdir("Augmented_train")),
#     ],
#     ignore_index=True,
# )
# y_train = pd.concat(
#     [
#         train["encoded_class"],
#         pd.Series([0 for i in range(len(os.listdir("Augmented_train")))]),
#     ],
#     ignore_index=True,
# )

# # Wierdly, I am missing one file from Carson's normalized images - just me?
# X_train = X_train[
#     X_train != "Normalized/40_malignant_ductal_carcinoma_SOB_M_DC-14-15572-40-008.png"
# ]
# y_train = y_train[y_train.index != 1867]

# X_validation = "Normalized/" + validation["input_path"]
# y_validation = validation["encoded_class"]
X_test = "Normalized/" + test["input_path"]
y_test = test["encoded_class"]


def load_image(filename: str, label: int) -> Tuple[tf.Tensor, str]:
    file = tf.io.read_file(filename)
    img = tf.image.decode_png(file, channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    return img, label


# train_ds = (
#     tf.data.Dataset.from_tensor_slices((X_train, y_train))
#     .map(load_image)
#     .batch(BATCH_SIZE)
# )
# validation_ds = (
#     tf.data.Dataset.from_tensor_slices((X_validation, y_validation))
#     .map(load_image)
#     .batch(BATCH_SIZE)
# )

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(load_image)
    .batch(BATCH_SIZE)
)

# Cache and prefetch data for faster training
AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


class Models:
    """
    Contains the logic for building all models as well as functions to load the models once they are built.
    """

    def __init__(self) -> None:
        self.AlexNet_x_test = None
        self.AlexNet_y_test = None
        self.AlexNetBreaKHis_test = test_ds
        self.VGGNet_imagenet_path = f"VGGNet_ImageNet_Model"
        self.AlexNet_mnist_path = f"AlexNet_MNIST_Model"
        self.VGGNet_breakhis_path = f"VGGNet_BreaKHis"
        self.VGGNet_breakhis_path_optimized = f"VGGNet_BreaKHis_optimized"
        self.AlexNet_breakhis_path_optimized = f"AlexNet_BreaKHis_optimized"
        self.AlexNet_breakhis_path = f"AlexNet_BreaKHis"

    def build_VGGNet_imagenet(self):
        """
        Builds an instance of VGG19 from ImageNet
        """
        model = VGG19(weights="imagenet")
        model.save(self.VGGNet_imagenet_path)

    def build_VGGNet_breakhis(self):
        """
        Builds an instance of VGG19 using transfer learning from ImageNet and trained further on BreaKHis

        about 250K learnables
        """
        base = VGG19(
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights="imagenet",
            pooling="avg",
        )
        base.trainable = False
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
                # Data augmentation
                layers.RandomBrightness(0.2, seed=SEED),
                layers.RandomFlip(seed=SEED),
                layers.RandomRotation(0.2, seed=SEED),
                # VGG19
                layers.Lambda(tf.keras.applications.vgg19.preprocess_input),
                base,
                layers.Dropout(0.4),
                # Fully connected layers
                layers.Dense(384, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="VGG19",
        )
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

        model.save(self.VGGNet_breakhis_path)

    def build_AlexNet_breakhis(self, visualize_training=False):
        """
        Builds an instance of AlexNet trained on the BreaKHis data.

        I added relu activations in the 2D convolutions because if we don't pass that in there is no default and no activation is used

        About 20M trainable params

        """
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
        model.add(layers.Conv2D(96, 11, strides=4, padding="same", activation="relu"))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(256, 5, strides=3, padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(384, 3, strides=4, padding="same", activation="relu"))
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

        model.save(self.AlexNet_breakhis_path)

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

    def build_AlexNet_breakhis_hp_optimized(self, hp):
        """
        Builds an instance of AlexNet trained on the BreaKHis data. Must be called with the bayesian tuner

        I rebuilt the structure of this network to be more like the structure in Matlab with grouped convolutional layers

        Haven't been able to run this model yet because of memory limitations. There are too many learnable params in this network, would need to utilize ROSIE for training

        About 60M trainables

        Optimizes HPs whether or not to include grouped convolutions, hidden neurons in FC layer, dropout percentage, learning rate
        """
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

        model.add(layers.Conv2D(96, 11, strides=4, padding="valid", activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(
            layers.Conv2D(
                256, 5, strides=1, groups=2, padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.BatchNormalization())
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(384, 3, strides=1, padding="same", activation="relu"))
        include_conv1 = hp.Boolean("include_conv1", default=False)
        if include_conv1:
            model.add(
                layers.Conv2D(
                    384, 3, strides=1, groups=2, padding="same", activation="relu"
                )
            )
        include_conv2 = hp.Boolean("include_conv2", default=False)
        if include_conv2:
            model.add(
                layers.Conv2D(
                    384, 3, strides=1, groups=2, padding="same", activation="relu"
                )
            )
        model.add(layers.MaxPool2D(3, strides=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(
            layers.Dense(
                hp.Int("hidden_size", 1096, 4096, step=1000, default=4096),
                activation="relu",
            )
        )
        model.add(layers.Dropout(hp.Float("dropout", 0, 0.5, step=0.1, default=0.5)))
        model.add(layers.Dense(1, activation="softmax"))
        model.summary()

        print("Compiling...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-3, 1e-2, sampling="log")
            ),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="roc_auc"), "accuracy"],
        )
        print("Finished compiling.")

        return model

    def train_AlexNet_with_bayesian_tuner(self, trials=20):
        # call build method wich whill compile the model, 20 configurations of networks will be trained if trials not passed in and results saved to a folder with project_name.

        tuner = kt.tuners.BayesianOptimization(
            self.build_AlexNet_breakhis_hp_optimized,
            objective="val_accuracy",
            max_trials=trials,
            overwrite=True,
            project_name=self.AlexNet_breakhis_path_optimized + "_Bayesian_trials",
        )

        early_stopping = EarlyStopping(
            min_delta=1e-4, patience=5, verbose=1, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)

        # call tuner.search instead of model.fit
        tuner.search(
            train_ds,
            validation_data=validation_ds,
            epochs=MAX_EPOCHS,
            callbacks=[early_stopping, reduce_lr],
        )

        try:
            tuner.get_best_models(1)[0].save(self.AlexNet_breakhis_path)
        except:
            return tuner
        return tuner

    def build_VGGNet_breakhis_hp_optimized(self, hp):
        """
        Builds an instance of VGG19 using transfer learning from ImageNet and trained further on BreaKHis, called with tuner method below
        Optimizes HPs dropout percnetage, number neurons in FC layer, learning rate

        About 600K learnables
        """
        base = VGG19(
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights="imagenet",
            pooling="avg",
        )
        base.trainable = False
        model = tf.keras.Sequential(
            [
                layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
                # Data augmentation
                layers.RandomBrightness(0.2, seed=SEED),
                layers.RandomFlip(seed=SEED),
                layers.RandomRotation(0.2, seed=SEED),
                # VGG19
                layers.Lambda(tf.keras.applications.vgg19.preprocess_input),
                base,
                layers.Dropout(hp.Float("dropout", 0, 0.5, step=0.1, default=0.4)),
                # Fully connected layers
                layers.Dense(
                    hp.Int("hidden_size", 384, 1384, step=200, default=384),
                    activation="relu",
                ),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="VGG19",
        )
        model.summary()

        print("Compiling...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-3, 1e-2, sampling="log")
            ),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="roc_auc"), "accuracy"],
        )
        print("Finished compiling.")

        return model

    def train_VGG19_with_bayesian_tuner(self, trials=20):
        # call build method wich whill compile the model, 20 configurations of networks will be trained by default if trials not passed

        tuner = kt.tuners.BayesianOptimization(
            self.build_VGGNet_breakhis_hp_optimized,
            objective="val_accuracy",
            max_trials=trials,
            overwrite=True,
            project_name=self.VGGNet_breakhis_path_optimized + "_Bayesian_trials",
        )

        early_stopping = EarlyStopping(
            min_delta=1e-4, patience=5, verbose=1, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)

        # call tuner.search instead of model.fit
        tuner.search(
            train_ds,
            validation_data=validation_ds,
            epochs=MAX_EPOCHS,
            callbacks=[early_stopping, reduce_lr],
        )

        try:
            tuner.get_best_models(1)[0].save(self.VGGNet_breakhis_path_optimized)
        except:
            return tuner
        return tuner

    def open_model(self, path):
        if "AlexNet_BreaKHis" in path:
            SEED = 51432
            tf.keras.utils.set_random_seed(SEED)
            tf.config.experimental.enable_op_determinism()
            fold_info = pd.read_csv("Folds.csv")
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
            return load_model(path)
