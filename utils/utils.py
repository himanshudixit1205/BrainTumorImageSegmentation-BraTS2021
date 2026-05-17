import os
import random
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import json
import keras
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from glob import glob
from dotenv import load_dotenv
import yaml

# ---------------- LOAD CONFIG ----------------
with open("../configs/config_brats.yaml", "r") as file:
    config = yaml.safe_load(file)

# ---------------- LOAD DOTENV FILE ----------------load_dotenv()   

# ---------------- PARAMETERS ----------------SEED = config["training"]["seed"]
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
SMOOTH = float(config["training"]["smooth"])
IMG_SIZE = config["training"]["image_size"]

# ---------------- CALLBACKS ----------------
class SaveBestEpoch(keras.callbacks.Callback):
    def __init__(self, filepath="best_epoch.json"):
        super().__init__()
        self.filepath = filepath
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_dice_coefficients")
        if current is not None and current > self.best:
            self.best = current
            data = {
                "best_epoch": int(epoch + 1),
                "best_val_dice": float(current)
            }
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=4)
            print(f"✅ Saved best epoch: {epoch+1}")

# ---------------- DICE COEFFICIENT ----------------
def dice_coefficients(y_true, y_pred, smooth=SMOOTH):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - dice_coefficients(y_true, y_pred, smooth) # Loss Metric

def iou(y_true, y_pred, smooth=SMOOTH):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


# ---------------- JACCARD DISTANCE ----------------
def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return 1 - iou(y_true_flatten, y_pred_flatten) # Loss Metric


# ---------------- LOSS FUNCTIONS ----------------
def bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(binary_crossentropy(y_true, y_pred))


def dice_loss(y_true, y_pred, smooth=SMOOTH):

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    dice = ((2.0 * intersection + smooth) / ( tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

    return 1.0 - dice


# ---------------- BCE + DICE LOSS ----------------
def bce_dice_loss(y_true, y_pred):
    bce = bce_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def combined_loss(y_true, y_pred):
    """
    Combines Focal Tversky Loss and Dice Loss.
    Focal Tversky handles class imbalance, Dice improves boundary overlap.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Your existing focal_tversky_loss (imported or defined above)
    ft_loss = focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.33)

    # Dice loss = 1 - Dice coefficient
    dice_loss = 1 - dice_coefficients(y_true, y_pred, smooth=SMOOTH)

    return ft_loss + dice_loss

# ---------------- IMAGE WITH MASK VISUALIZATION ----------------
def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
    fig = plt.figure(figsize=(12,12))
    rnge = rows * columns + 1

    for i in range(1, rnge):
        fig.add_subplot(rows, columns, i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)

    plt.show()

# ---------------- IMAGE MASK SIDE-BY-SIDE ----------------
def show_img_mask_rows(n, list_img_path, list_mask_path):
    fig = plt.figure(figsize=(6, 3 * n))

    plot_idx = 1

    for i in range(n):
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)  # grayscale mask

        # IMAGE ---
        fig.add_subplot(n, 2, plot_idx)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")
        plot_idx += 1

        # MASK ---
        fig.add_subplot(n, 2, plot_idx)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis("off")
        plot_idx += 1

    plt.tight_layout()
    plt.show()


# ---------------- NORMALIZATION ----------------
def normalize_and_diagnose(img, mask):
    img = (img / 255.0).astype(np.float32)
    mask = (mask / 255.0)
    mask = (mask > 0.5).astype(np.float32)
    return (img, mask)



# ---------------- DATA GENERATORS ----------------
def train_generator(
    data_frame,
    batch_size,
    augmentation_dict,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    save_to_dir=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    seed=1,
):
    image_datagen = ImageDataGenerator(**augmentation_dict)
    
    mask_datagen = ImageDataGenerator(
        rotation_range=augmentation_dict.get("rotation_range", 0),
        width_shift_range=augmentation_dict.get("width_shift_range", 0),
        height_shift_range=augmentation_dict.get("height_shift_range", 0),
        shear_range=augmentation_dict.get("shear_range", 0),
        zoom_range=augmentation_dict.get("zoom_range", 0),
        horizontal_flip=augmentation_dict.get("horizontal_flip", False),
        vertical_flip=augmentation_dict.get("vertical_flip", False),
        fill_mode='nearest'
    )
    

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,                   # Pandas DataFrame containing file paths
        x_col="image",                # Column name with image file paths
        class_mode=None,              # No labels returned (augmentation only)
        color_mode=image_color_mode,  # 'rgb' or 'grayscale' image mode
        target_size=target_size,      # Resize images to (height, width)
        batch_size=batch_size,        # Images per batch
        save_to_dir=save_to_dir,      # Directory to save augmented images
        save_prefix=image_save_prefix,# Prefix for saved filenames
        seed=seed,                    # Random seed for reproducibility
        shuffle=False                  # No shuffling of images
    )


    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        shuffle=False

    )

    train_gen = zip(image_generator, mask_generator)

    # ---------------- NORMALIZATION GENERATOR OUTPUT ----------------
    for (img, mask) in train_gen:
        img, mask = normalize_and_diagnose(img, mask)
        yield (img, mask)
        
        
def train_generator_shuffle(
    data_frame,
    batch_size,
    augmentation_dict,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    save_to_dir=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    seed=1,
    shuffle=True,                     
):
    image_datagen = ImageDataGenerator(**augmentation_dict)
    
    mask_datagen = ImageDataGenerator(
        rotation_range=augmentation_dict.get("rotation_range", 0),
        width_shift_range=augmentation_dict.get("width_shift_range", 0),
        height_shift_range=augmentation_dict.get("height_shift_range", 0),
        shear_range=augmentation_dict.get("shear_range", 0),
        zoom_range=augmentation_dict.get("zoom_range", 0),
        horizontal_flip=augmentation_dict.get("horizontal_flip", False),
        vertical_flip=augmentation_dict.get("vertical_flip", False),
        fill_mode='nearest'
    )
    

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,                   # Pandas DataFrame containing file paths
        x_col="image",                # Column name with image file paths
        class_mode=None,              # No labels returned (augmentation only)
        color_mode=image_color_mode,  # 'rgb' or 'grayscale' image mode
        target_size=target_size,      # Resize images to (height, width)
        batch_size=batch_size,        # Images per batch
        save_to_dir=save_to_dir,      # Directory to save augmented images
        save_prefix=image_save_prefix,# Prefix for saved filenames
        seed=seed,                    # Random seed for reproducibility
        shuffle=shuffle               # Shuffles the images
    )


    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        shuffle=shuffle               
    )

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = normalize_and_diagnose(img, mask)
        yield (img, mask)

# ---------------- LOAD IMAGE AND MASK FILES ----------------
def load_image_filename(split_path="train"):
    image_files = []
    mask_files = []

    base_path = os.getenv("DATASET_PATH")
    path = os.path.join(base_path, split_path)

    # Recursively find all mask files
    all_mask_files = glob(f"{path}/**/*_mask.tif", recursive=True)

    for mask_path in all_mask_files:
        # Get corresponding image path
        img_path = mask_path.replace("_mask.tif", ".tif")

        # Ensure both exist
        if os.path.exists(img_path):
            image_files.append(img_path)
            mask_files.append(mask_path)

    return image_files, mask_files


# ---------------- LOAD VALID PAIRS ----------------
def load_pair(image, mask_files):
    # Check if valid images are added

    valid_pairs = []
    for img, mask in zip(image, mask_files):
        if os.path.exists(img) and os.path.exists(mask):
            valid_pairs.append((img, mask))

    df = pd.DataFrame(valid_pairs, columns=['image', 'mask'])
    return df

# ---------------- PLOT TRAINING CURVES ----------------
def plot_accuracy_loss(history):
    history_post_training = history.history
    train_dice_coeff_list = history_post_training['dice_coefficients']
    test_dice_coeff_list = history_post_training['val_dice_coefficients']

    train_jaccard_list = history_post_training['iou']
    test_jaccard_list = history_post_training['val_iou']

    train_loss_list = history_post_training['loss']
    test_loss_list = history_post_training['val_loss']
    
    os.makedirs("results", exist_ok=True)

    plt.figure(1)
    plt.plot(test_loss_list, 'b-')
    plt.plot(train_loss_list, 'r-')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss graph', fontsize=12)
    plt.savefig(f"results/LossGraph.png",
                    dpi=300,
                    bbox_inches='tight')
    plt.figure(2)
    plt.plot(train_dice_coeff_list, 'b-')
    plt.plot(test_dice_coeff_list, 'r-')
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('accuracy graph', fontsize=12)
    plt.savefig(f"results/Accuracy_Graph.png",
                    dpi=300,
                    bbox_inches='tight')
    plt.show()
    
    
def safe_mean(x):
    return np.mean(x) if len(x) > 0 else 0


# ---------------- FOCAL LOSS ----------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # Binary crossentropy formulation
        ce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        focal = alpha * (1 - y_pred) ** gamma * y_true * (-tf.math.log(y_pred)) + \
                (1 - alpha) * y_pred ** gamma * (1 - y_true) * (-tf.math.log(1 - y_pred))
        return tf.keras.backend.mean(focal, axis=-1)
    return focal_loss_fixed


# ---------------- STRATIFIED METRICS ----------------
def size_stratified_metrics(model, test_images, test_masks, size_thresholds=(500, 2000)):
    small_dice, medium_dice, large_dice = [], [], []
    small_iou, medium_iou, large_iou = [], [], []
    small_recall, medium_recall, large_recall = [], [], []
    
    for img, mask in zip(test_images, test_masks):
        pred = (model.predict(np.expand_dims(img, 0))[0] > 0.5).astype(np.uint8)
        
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        tumor_size = np.sum(mask)
        
        # Dice
        intersection = np.sum(pred * mask)
        dice = (2. * intersection) / (np.sum(pred) + np.sum(mask) + 1e-7)
        # IoU
        iou = intersection / (np.sum(pred) + np.sum(mask) - intersection + 1e-7)
        # Recall = TP / (TP+FN)
        tp = intersection
        fn = np.sum(mask) - tp
        recall = tp / (tp + fn + 1e-7)
        
        if tumor_size < size_thresholds[0]:
            small_dice.append(dice); small_iou.append(iou); small_recall.append(recall)
        elif tumor_size < size_thresholds[1]:
            medium_dice.append(dice); medium_iou.append(iou); medium_recall.append(recall)
        else:
            large_dice.append(dice); large_iou.append(iou); large_recall.append(recall)
    
    return {
            'small': {
                'dice': safe_mean(small_dice),
                'iou': safe_mean(small_iou),
                'recall': safe_mean(small_recall)
            },
            'medium': {
                'dice': safe_mean(medium_dice),
                'iou': safe_mean(medium_iou),
                'recall': safe_mean(medium_recall)
            },
            'large': {
                'dice': safe_mean(large_dice),
                'iou': safe_mean(large_iou),
                'recall': safe_mean(large_recall)
            }
    }
    
    
def compute_size_stratified_metrics(y_true, y_pred, thresholds=(1000, 5000)):
    # Flatten to (N, H, W)
    if y_true.ndim == 4 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)
        y_pred = y_pred.squeeze(-1)
    
    sizes = np.sum(y_true, axis=(1,2))  # tumor pixels per slice
    
    small_mask = sizes <= thresholds[0]
    medium_mask = (sizes > thresholds[0]) & (sizes <= thresholds[1])
    large_mask = sizes > thresholds[1]
    
    results = {}
    for name, mask in zip(['small', 'medium', 'large'], [small_mask, medium_mask, large_mask]):
        if np.sum(mask) == 0:
            results[name] = {'dice': np.nan, 'iou': np.nan, 'recall': np.nan}
            continue
        
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        # Dice
        intersection = np.sum(y_t * y_p, axis=(1,2))
        dice = (2 * intersection + 1e-6) / (np.sum(y_t, axis=(1,2)) + np.sum(y_p, axis=(1,2)) + 1e-6)
        
        # IoU
        union = np.sum(y_t, axis=(1,2)) + np.sum(y_p, axis=(1,2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # Recall
        recall = (intersection + 1e-6) / (np.sum(y_t, axis=(1,2)) + 1e-6)
        
        results[name] = {
            'dice': np.mean(dice),
            'iou': np.mean(iou),
            'recall': np.mean(recall)
        }
    
    return results


# ---------------- TVERSKY LOSSES ----------------
def tversky_loss(y_true, y_pred, alpha=0.2, beta=0.8, smooth=SMOOTH):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1 - y_true) * y_pred)
    FN = tf.reduce_sum(y_true * (1 - y_pred))

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky

def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.33):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1 - y_true) * y_pred)
    FN = tf.reduce_sum(y_true * (1 - y_pred))

    tversky = (TP + SMOOTH) / (TP + alpha * FP + beta * FN + SMOOTH)

    return tf.pow((1 - tversky), gamma)


# ---------------- EVALUATION METRICS ----------------
def precision_m(y_true, y_pred, smooth=SMOOTH):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    return (tp + smooth) / (tp + fp + smooth)

def recall_m(y_true, y_pred, smooth=SMOOTH):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    return (tp + smooth) / (tp + fn + smooth)

def volume_dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + 1e-6) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def volume_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def volume_precision(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return (tp + 1e-6) / (tp + fp + 1e-6)

def volume_recall(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return (tp + 1e-6) / (tp + fn + 1e-6)

def volume_specificity(y_true, y_pred):
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return (tn + 1e-6) / (tn + fp + 1e-6)


# ---------------- FINAL METRIC COMPUTATION ----------------
EPS = 1e-6

def compute_metrics(y_true, y_pred):

    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)

    sum_true = np.sum(y_true)
    sum_pred = np.sum(y_pred)

    union = sum_true + sum_pred - intersection

    tp = intersection
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))

    dice = (2.0 * intersection + EPS) / (
        sum_true + sum_pred + EPS
    )

    iou_val = (intersection + EPS) / (
        union + EPS
    )

    precision = (tp + EPS) / (
        tp + fp + EPS
    )

    if sum_true == 0:
        recall = np.nan
    else:
        recall = (tp + EPS) / (
            tp + fn + EPS
        )

    return dice, iou_val, precision, recall