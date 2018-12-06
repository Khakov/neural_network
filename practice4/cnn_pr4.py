from keras.applications import Xception, VGG19, InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.vgg16 import VGG16

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

models = {'VGG16': VGG16, 'Xception': Xception, 'VGG19': VGG19, 'InceptionV3': InceptionV3, 'MobileNet': MobileNet,
          'ResNet50': ResNet50, 'MobileNetV2': MobileNetV2, 'InceptionResNetV2': InceptionResNetV2}
for key, model_type in models.items():
    model = model_type(weights='imagenet')

    img_path = 'image/8.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted for {} model: '.format(key), decode_predictions(preds, top=3)[0])
