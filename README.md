# dogs-vs-cats-transfer-learning
A Colab-based deep learning project using MobileNetV2 transfer learning to classify images of dogs and cats from the Kaggle dataset with over 97% accuracy.

 Dogs vs Cats Image Classification (Transfer Learning)

This project uses **MobileNetV2** transfer learning to classify dog and cat images from the [Kaggle Dogs vs Cats competition](https://www.kaggle.com/c/dogs-vs-cats).  
Built and tested in **Google Colab**, achieving **97%+ accuracy**.

---

 Features
- Downloads dataset directly from Kaggle using the Kaggle API
- Preprocesses and resizes images to 224x224
- Uses **MobileNetV2** pretrained on ImageNet
- Fine-tunes for binary classification (Dog vs Cat)
- Allows prediction on custom images

---

 Setup: Kaggle API in Colab

1. **Get Kaggle API Token**
   - Go to your [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll to **API** and click **Create New API Token**
   - This downloads a file `kaggle.json`

2. **Upload `kaggle.json` to Colab**
```python
from google.colab import files
files.upload()  # Upload kaggle.json

    Place it in the correct folder

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

    Download the dataset

!kaggle competitions download -c dogs-vs-cats

    Unzip the dataset

from zipfile import ZipFile

with ZipFile('dogs-vs-cats.zip', 'r') as zip_ref:
    zip_ref.extractall()

 Running the Notebook

    Open TRANSFERL3.ipynb in Google Colab

    Run all cells sequentially

    At the end, you can test the model with your own image:

input_image_path = "path/to/your/image.jpg"

 Results

    Accuracy: ~97.5% on test set

    Base Model: MobileNetV2 (frozen layers)

    Custom Layers: Global Average Pooling + Dense(2, softmax)
