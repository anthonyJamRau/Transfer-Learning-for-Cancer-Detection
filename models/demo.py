from models import (
    Models,
)  # Be sure to import the class. It might be from models.models or a different path depending on where you're working
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np
from scriptime import Timer

timer = Timer(method="env")

"""
This file shows how to use the Models class in models.py
"""

# First, create an instance of models
models = Models()


def test_vgg_imagenet():
    vggnet = models.load_model(
        "VGGNet_Model.pkl"
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
    alex = models.load_model(
        "AlexNet_Model.pkl"
    )  # Load the trained model. Be sure to use the correct path.
    alex.evaluate(
        models.AlexNet_x_test, models.AlexNet_y_test
    )  # Use the built-in TensorFlow method


def test_alex_breakhis():
    alex = models.load_model("AlexNet_BreaKHis.pkl")
    loss, auc, accuracy = alex.evaluate(models.AlexNetBreaKHis_test, verbose=1)
    print(
        f"\nROC-AUC:   {auc:.5f}",
        f"Accuracy:  {accuracy:.5f}",
        f"Loss:      {loss:.5f}\n",
        sep="\n",
    )


def test_vgg_breakhis():
    vgg = models.load_model("VGGNet_BreaKHis.pkl")
    loss, auc, accuracy = vgg.evaluate(models.AlexNetBreaKHis_test, verbose=1)
    print(
        f"\nROC-AUC:   {auc:.5f}",
        f"Accuracy:  {accuracy:.5f}",
        f"Loss:      {loss:.5f}\n",
        sep="\n",
    )


if __name__ == "__main__":
    timer.start()
    # models.build_AlexNet_mnist()  # Builds an AlexNet model based on MNIST data set. ONLY RUN TO RETRAIN THE MODEL
    # models.build_VGGNet_imagenet()  # Builds a VGGNet model based on ImageNet data set. ONLY RUN TO RETRAIN THE MODEL
    # models.build_AlexNet_breakhis()
    models.build_VGGNet_breakhis()
    timer.send_email(["jstr36@gmail.com", "9064202283@txt.att.net"])
    # print(
    #     "-----------------------------------\nTesting VGGNet ImageNet...\n-----------------------------------"
    # )
    # test_vgg_imagenet()
    # print(
    #     "-----------------------------------\nTesting AlexNet MNIST...\n-----------------------------------"
    # )
    # test_alex_mnist()
    # print(
    #     "-----------------------------------\nTesting AlexNet BreaKHis...\n-----------------------------------"
    # )
    # test_alex_breakhis()
