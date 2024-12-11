import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.applications import Xception, MobileNetV2 # type: ignore
from tensorflow.keras.datasets import cifar10 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Resize images to match the input shape of Xception
x_train_xception = np.array([np.resize(img, (71, 71, 3)) for img in x_train])
x_test_xception = np.array([np.resize(img, (71, 71, 3)) for img in x_test])

x_train_xception = x_train_xception.astype('float32') / 255.0
x_test_xception = x_test_xception.astype('float32') / 255.0

# Resize images to match the input shape of MobileNetV2
x_train_mobilenetv2 = np.array([np.resize(img, (224, 224, 3)) for img in x_train])
x_test_mobilenetv2 = np.array([np.resize(img, (224, 224, 3)) for img in x_test])

x_train_mobilenetv2 = x_train_mobilenetv2.astype('float32') / 255.0
x_test_mobilenetv2 = x_test_mobilenetv2.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train and save Xception model
base_model_xception = Xception(weights='imagenet', include_top=False, input_shape=(71, 71, 3))
model_xception = Sequential([
    base_model_xception,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model_xception.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_xception.fit(x_train_xception, y_train, batch_size=64, epochs=10, validation_data=(x_test_xception, y_test))
model_xception.save('xception_model.h5')

# Train and save MobileNetV2 model
base_model_mobilenetv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_mobilenetv2 = Sequential([
    base_model_mobilenetv2,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model_mobilenetv2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_mobilenetv2.fit(x_train_mobilenetv2, y_train, batch_size=64, epochs=10, validation_data=(x_test_mobilenetv2, y_test))
model_mobilenetv2.save('mobilenetv2_model.h5')

# Train and save CIFAR-10 model
model_cifar10 = Sequential([
    Dense(512, activation='relu', input_shape=(32, 32, 3)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model_cifar10.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cifar10.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
model_cifar10.save('cifar10_model.h5')
