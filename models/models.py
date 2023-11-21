from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import datetime, re, pickle
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


SEED = 51432
data_dir_train = 'Normalized-Train/'
data_dir_validation = 'Normalized-Validation/'
data_dir_test = 'Normalized-Test/'
BATCH_SIZE = 48
IMG_SIZE = 224
BASE_LEARNING_RATE = 0.001
tf.keras.utils.set_random_seed(SEED)
MAX_EPOCHS = 10


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  seed=SEED,
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE,
  label_mode = "binary",
  shuffle = True)

validation_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_validation,
  seed=SEED,
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE,
  label_mode = "binary",
  shuffle = True)

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_test,
  seed=SEED,
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=BATCH_SIZE,
  label_mode = "binary",
  shuffle = True)

# # Cache and prefetch data for faster training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


class Models:
    """
    Contains the logic for building all models as well as functions to load the models once they are built.
    """

    def __init__(self) -> None:
        self.AlexNet_x_test = None
        self.AlexNet_y_test = None
#         self.AlexNetBreaKHis_test = test_ds
        self.VGGNet_imagenet_path = f"VGGNet_ImageNet_Model"
        self.AlexNet_mnist_path = f"AlexNet_MNIST_Model"
        self.VGGNet_breakhis_path = f"VGGNet_BreaKHis"
        self.VGGNet_breakhis_path_optimized = f"VGGNet_BreaKHis_optimized"
        self.AlexNet_breakhis_path_optimized = f"AlexNet_BreaKHis_optimized"
        self.AlexNet_breakhis_path = f"AlexNet_BreaKHis"
        self.VGGNetFineTuned_path = f"VGG_BreaKHis_fineTuned"

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
        
        does fine tuning 11/20
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
#                 layers.RandomBrightness(0.2, seed=SEED),
#                 layers.RandomFlip(seed=SEED),
#                 layers.RandomRotation(0.2, seed=SEED),
                # VGG19
                layers.Lambda(tf.keras.applications.vgg19.preprocess_input),
                base,
                layers.Dropout(0.2),
                # Fully connected layers
                # setting the trainable params to ones found with Bayesian optimization in previous training runs
                layers.Dense(384, activation="relu"),
                layers.Dropout(0.1),
                layers.Dense(48, activation="relu"),
                layers.Dropout(0.0),
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

#         model.save(self.VGGNet_breakhis_path)
        base.trainable = True
        print(model.summary())

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="roc_auc"), "binary_accuracy"],
        )

        epochs = 5
        model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
        
        return model
       

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
        model.add(layers.Conv2D(96, 11, strides=4, padding="same",activation = 'relu'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(256, 5, strides=3, padding="same",activation = 'relu'))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Conv2D(384, 3, strides=4, padding="same",activation = 'relu'))
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
            
    def build_AlexNet_breakhis_hp_optimized(self,hp):
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
        
        model.add(layers.Conv2D(96, 11, strides=4, padding="valid", activation = "relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(256, 5, strides=1, groups = 2,padding="same", activation = "relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(384, 3, strides=1, padding="same", activation = "relu"))
        include_conv1 = hp.Boolean('include_conv1', default = False)
        if include_conv1:
            model.add(layers.Conv2D(384, 3, strides=1, groups = 2,padding="same", activation = "relu"))
        include_conv2 = hp.Boolean('include_conv2', default = False)
        if include_conv2:
            model.add(layers.Conv2D(384, 3, strides=1, groups = 2,padding="same", activation = "relu"))
        model.add(layers.MaxPool2D(3, strides = 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense( hp.Int('hidden_size', 1096, 4096, step=1000, default=4096), activation="relu"))
        model.add(layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)))
        model.add(layers.Dense(1, activation="softmax"))
        model.summary()

        print("Compiling...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-3, 1e-2, sampling='log')),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="roc_auc"), "accuracy"],
        )
        print("Finished compiling.")
        
        return model

    def train_AlexNet_with_bayesian_tuner(self,trials = 20):
        
        # call build method wich whill compile the model, 20 configurations of networks will be trained if trials not passed in and results saved to a folder with project_name.
        
        tuner = kt.tuners.BayesianOptimization(
          self.build_AlexNet_breakhis_hp_optimized,
          objective='val_accuracy',
          max_trials=trials,
          overwrite = True,
          project_name=self.AlexNet_breakhis_path_optimized+"_Bayesian_trials"
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
    
    def build_VGGNet_breakhis_hp_optimized(self,hp):
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
            
            hp_activation_func = hp.Choice('activation_func', values=['sigmoid','softmax','tanh'])
            
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
                    layers.Dropout(hp.Float('dropout1', 0.2, 0.4, step=0.1, default=0.4)),
                    # Fully connected layers
                    layers.Dense( hp.Int('hidden_size1', 184, 384, step=100, default=384), activation="relu"),
                    layers.Dropout(hp.Float('dropout2', 0.1, 0.3, step=0.1, default=0.3)),
                    layers.Dense( hp.Int('hidden_size2', 32, 64, step=16, default=64), activation="relu"),
                    layers.Dropout(hp.Float('dropout3', 0, 0.2, step=0.1, default=0.2)),
                    layers.Dense(1, activation=hp_activation_func),
                ],
                name="VGG19",
            )
            model.summary()
            
            hp_lr = hp.Choice('lr', values=[0.001,0.01,0.0001])

            print("Compiling...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(hp_lr),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC(name="roc_auc"), "binary_accuracy"],
            )
            print("Finished compiling.")

            return model
        
    def fine_tune(self,model,epochs = 10):
        
        base.trainable = True
        print(model.summary())

#         model.compile(
#             optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
#             loss=keras.losses.BinaryCrossentropy(from_logits=True),
#             metrics=[keras.metrics.BinaryAccuracy()],
#         )

#         epochs = 10
#         model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

#         model.save(self.VGGNetFineTuned_path)
        
        
    
    def train_VGG19_with_bayesian_tuner(self,trials = 20):
        
        # call build method which whill compile the model, 20 configurations of networks will be trained by default if trials not passed
        
        tuner = kt.tuners.BayesianOptimization(
          self.build_VGGNet_breakhis_hp_optimized,
          objective='binary_accuracy',
          max_trials=trials,
          overwrite = True,
          project_name=self.VGGNet_breakhis_path_optimized+"_Bayesian_trials"
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
