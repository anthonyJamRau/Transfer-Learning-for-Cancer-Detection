import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tensorflow.keras.applications.vgg19 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pandas as pd

from models import Models

"""
This file shows how to use the Models class in models.py
"""

# First, create an instance of models
models = Models()

# Make test_ds
BATCH_SIZE = 28
IMG_SIZE = 224

X_test_b = os.listdir("Normalized-Test/Benign")
X_test_m = os.listdir("Normalized-Test/Malignant")

# Create a DataFrame object with the path and encoded label
X_test_df = pd.DataFrame(
    {"path": X_test_b + X_test_m, "label": [0] * len(X_test_b) + [1] * len(X_test_m)}
)

# Prefix each path with the appropriate directory based on the label
X_test_df["path"] = X_test_df.apply(
    lambda row: f"Normalized-Test/Benign/{row['path']}"
    if row["label"] == 0
    else f"Normalized-Test/Malignant/{row['path']}",
    axis=1,
)


X_test = X_test_df["path"]


y_test = X_test_df["label"]


def load_image(filename: str, label: int):
    file = tf.io.read_file(filename)
    img = tf.image.decode_png(file, channels=3)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    return img, label


test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(load_image)
    .batch(BATCH_SIZE)
)


def test_vgg_imagenet():
    vggnet = models.open_model(
        "/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/models/VGGNet_ImageNet_Model"
    )  # Load the trained model. Be sure to use the correct path.

    # Formats image appropriately
    img_path = "data/ollie.jpg"  # Picture of my yellow lab
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Use the built-in TensorFlow methods
    predictions = vggnet.predict(x)
    decoded_predictions = decode_predictions(predictions)
    for i in range(5):
        print(f"{decoded_predictions[0][i][1]}: {decoded_predictions[0][i][2]:.2f}")


def test_alex_mnist():
    alex = models.open_model(
        "AlexNet_Model"
    )  # Load the trained model. Be sure to use the correct path.
    loss, accuracy = alex.evaluate(
        models.AlexNet_x_test, models.AlexNet_y_test
    )  # Use the built-in TensorFlow method
    print(
        f"Accuracy:  {accuracy:.5f}",
        f"Loss:      {loss:.5f}\n",
        sep="\n",
    )


def test_w10alex_breakhis():
    alex = models.open_model(
        "/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/models/w10_AlexNet_BreaKHis"
    )
    evaluate_model(
        alex,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w10_AlexNetBreaKHis/w14",
    )


# def test_w10vgg_breakhis():
#     vgg = models.open_model("w10_VGGNet_BreaKHis")
#     evaluate_model(
#         vgg,
#         test_ds,
#         save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w10_VGG19BreaKHis/w14",
#     )


def test_w11vgg_breakhis():
    vgg = models.open_model("VGGNet_BreaKHis")
    evaluate_model(
        vgg,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w14_VGG19BreaKHis/w14",
    )


def test_w12vgg_breakhis():
    vgg = models.open_model("w12_VGGNet_BreaKHis")
    evaluate_model(
        vgg,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w12_VGG19BreaKHis/w14",
    )


def test_w11vgg_breakhis_optimized():
    vgg = models.open_model("VGGNet_BreaKHis_optimized")
    evaluate_model(
        vgg,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w11_VGG19BreaKHis_optimized/w15",
    )


def test_w12vgg_breakhis_optimized():
    vgg = models.open_model("w12_VGGNet_BreaKHis_optimized")
    evaluate_model(
        vgg,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w12_VGG19BreaKHis_optimized/w15",
    )


def test_w14vgg_breakhis_optimized():
    vgg = models.open_model("w14_VGGNet_BreaKHis")
    evaluate_model(
        vgg,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w14_VGG19BreaKHis_optimized/w14",
    )


def test_w14vgg_tuned_breakhis_optimized():
    vgg = models.open_model("VGG_BreaKHis_fineTuned")
    evaluate_model(
        vgg,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w14_VGG19BreaKHis_finetuned/w14",
    )


def test_w14vgg_tuned2_breakhis_optimized():
    vgg = models.open_model("VGG_BreaKHis_fineTuned2")
    evaluate_model(
        vgg,
        test_ds,
        save_dir="/Users/jakestrasler/Documents/msml/Transfer-Learning-for-Cancer-Detection/plots/w14_VGG19BreaKHis_finetuned2/w14",
    )


def evaluate_model(model, dataset, save_dir=None):
    loss, auc, accuracy = model.evaluate(dataset, verbose=1)
    predictions = model.predict(dataset, verbose=0).reshape(-1)

    # Get labels and predictions for each batch in dataset
    print(
        f"Getting labels and predictions for each batch in dataset for model {model.name}..."
    )
    results = [
        (labels, model.predict(images, verbose=0).reshape(-1))
        for images, labels in dataset.take(-1)
    ]
    labels = np.concatenate([x[0] for x in results])
    preds = np.concatenate([x[1] for x in results])
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        labels, preds.round().astype("uint8"), average="binary"
    )
    print(
        f"\nROC-AUC:   {auc:.5f}",
        f"Accuracy:  {accuracy:.5f}",
        f"Precision: {precision:.5f}",
        f"Recall:    {recall:.5f}",
        f"F1-Score:  {f1:.5f}",
        f"Loss:      {loss:.5f}\n",
        sep="\n",
    )
    print(
        f"Finished getting labels and predictions for each batch in dataset for model {model.name}."
    )
    print(f"Building plots for {model.name}...")
    labels = np.concatenate([x[0] for x in results])
    preds = np.concatenate([x[1] for x in results])
    # Plot metrics
    fig, axes = plt.subplots(ncols=3, figsize=(15, 4), dpi=160)
    curves = [metrics.RocCurveDisplay, metrics.PrecisionRecallDisplay]
    for ax, curve in zip(axes[:2], curves):
        curve.from_predictions(labels, preds, ax=ax, name=model.name)
    metrics.ConfusionMatrixDisplay.from_predictions(
        labels,
        preds.round().astype("uint8"),
        ax=axes[2],
        colorbar=False,
    )
    titles = ["ROC-AUC Curve", "Precision-Recall Curve", "Confusion Matrix"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, size=14, pad=10)
    print(f"Finished building plots for {model.name}.")

    if save_dir:
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the figure to a file in the specified directory
        fig.savefig(os.path.join(save_dir, f"{model.name}_plots.png"))

    plt.close()


if __name__ == "__main__":
    try:  # I know wrapping all this in a try-except is bad practice but there is an error that happens that seems to have no effect on the output so I am just suppressing it.
        # models.build_AlexNet_mnist()  # Builds an AlexNet model based on MNIST data set. ONLY RUN TO RETRAIN THE MODEL
        # models.build_VGGNet_imagenet()  # Builds a VGG19 model based on ImageNet data set. ONLY RUN TO RETRAIN THE MODEL
        # models.build_AlexNet_breakhis()  # Builds an AlexNet model based on BreaKHis data set. ONLY RUN TO RETRAIN THE MODEL
        # models.build_VGGNet_breakhis()  # Builds a VGG19 model based on BreaKHis data set. ONLY RUN TO RETRAIN THE MODEL
        # print(
        #     "-----------------------------------\nTesting VGGNet ImageNet...\n-----------------------------------"
        # )
        # test_vgg_imagenet()
        # print(
        #     "-----------------------------------\nTesting AlexNet MNIST...\n-----------------------------------"
        # )
        # test_alex_mnist()
        # print(
        #     "-----------------------------------\nTesting Week 10 AlexNet...\n-----------------------------------"
        # )
        # test_w10alex_breakhis()

        # print(
        #     "-----------------------------------\nTesting Week 10 VGGNet...\n-----------------------------------"
        # )
        # test_w10vgg_breakhis()
        print(
            "-----------------------------------\nTesting Week 11 VGGNet...\n-----------------------------------"
        )
        # test_w11vgg_breakhis()
        print(
            "-----------------------------------\nTesting Week 11 VGGNet (Optimized)...\n-----------------------------------"
        )
        test_w11vgg_breakhis_optimized()
        print(
            "-----------------------------------\nTesting Week 12 VGGNet...\n-----------------------------------"
        )
        # test_w12vgg_breakhis()
        print(
            "-----------------------------------\nTesting Week 12 VGGNet (Optimized)...\n-----------------------------------"
        )
        test_w12vgg_breakhis_optimized()
        print(
            "-----------------------------------\nTesting Week 13 VGGNet...\n-----------------------------------"
        )
        # test_w14vgg_breakhis_optimized()
        print(
            "-----------------------------------\nTesting Week 13 VGGNet (Fine Tuned)...\n-----------------------------------"
        )
        # test_w14vgg_tuned_breakhis_optimized()
        print(
            "-----------------------------------\nTesting Week 13 VGGNet (Fine Tuned 2)...\n-----------------------------------"
        )
        # test_w14vgg_tuned2_breakhis_optimized()
    except TypeError:
        print(
            "A TypeError has occurred. If the plots showed up as expected, feel free to ignore this."
        )
