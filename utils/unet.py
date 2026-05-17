import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    Dropout
)

#---------------- YAML ---------------- 
import yaml
with open("config_brats_new.yaml", "r") as file:
    config = yaml.safe_load(file)

SEED = config["training"]["seed"]
tf.random.set_seed(SEED)


# ---------------- STANDARD CONV BLOCK ---------------- 
def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def unet(input_size=(256,256,3)):
    inputs = Input(input_size)

    # ---------------- ENCODER ---------------- 
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    #---------------- BOTTLENECK  ---------------- 
    c5 = conv_block(p4, 1024)
    c5 = Dropout(0.2)(c5)

    # ---------------- DECODER ---------------- 
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])  
    c6 = conv_block(u6, 512)
    c6 = Dropout(0.2)(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])   
    c7 = conv_block(u7, 256)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])   
    c8 = conv_block(u8, 128)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])   
    c9 = conv_block(u9, 64)

    # ---------------- OUTPUT ---------------- 
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    return Model(inputs=[inputs], outputs=[outputs])