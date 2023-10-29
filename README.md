# Transfer-Learning-for-Cancer-Detection

Link to potential data sets <br>
https://www.kaggle.com/datasets/ambarish/breakhis/ <br>
https://iciar2018-challenge.grand-challenge.org/ (need to register for access) <br>
MIFLUDAN Project (need to request access)

## Week 8 Work
 - [Pretrained Model Exploration](pretrained_model_exploration)

## Week 9 Work
 - [Trained models on BreaKHis data](models/)

### Instructions for Using the Models

1. Download the zip files from [this Google Drive folder](https://drive.google.com/drive/folders/1DgCZn3C6yaeGEUnvJTnDEKkcK7UU0XQJ?usp=sharing) (the files were too big to add to git.)
2. Unzip the four zip files. There should be two AlexNet, two VGG19 models. 
3. Download the BreaKHis_V1 dataset and have the following file structure:
    ```
    .
    └── models/
        ├── data/
        │   ├── BreaKHis_V1/
        │   │   ├── histology_slides
        │   │   └── Folds.csv
        │   └── Ollie.jpg
        ├── models.py
        └── demo.py
    ```
4. You should now just be able to run `python demo.py` (or `python3` depending on how you have things set up.) You'll probably need to `pip install` some stuff too. 
5. After running the demo you should see an ouput that looks something like:
   ```
    -----------------------------------
    Testing VGGNet ImageNet...
    -----------------------------------
    WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
    1/1 [==============================] - 0s 268ms/step
    Labrador_retriever: 0.81
    American_Staffordshire_terrier: 0.05
    beagle: 0.02
    English_foxhound: 0.01
    Walker_hound: 0.01
    -----------------------------------
    Testing AlexNet MNIST...
    -----------------------------------
    32/32 [==============================] - 2s 74ms/step - loss: 0.4094 - accuracy: 0.9090
    Accuracy:  0.90900
    Loss:      0.40945
    -----------------------------------
    Testing AlexNet BreaKHis...
    -----------------------------------
    489/489 [==============================] - 57s 117ms/step - loss: 0.1629 - roc_auc: 0.5000 - binary_accuracy: 0.6696

    ROC-AUC:   0.50000
    Accuracy:  0.66959
    Loss:      0.16292

    -----------------------------------
    Testing VGGNet BreaKHis...
    -----------------------------------
    489/489 [==============================] - 1658s 3s/step - loss: 0.3430 - roc_auc: 0.9335 - binary_accuracy: 0.8723

    ROC-AUC:   0.93352
    Accuracy:  0.87230
    Loss:      0.34301
   ```

At this point you should be able to use the models in any other code in the same way it is used in the demo code (replacing `evaluate()` with `predict()` if you are trying to predict for an item/set of items.) 

One other thing to note: when you `from models import Models`, be sure to double check that path when you are doing this in another file. Depending on the location of the file you're working in this path may need to be changed. The `from models` is referring to models.py, `import Models` the class within the file.
