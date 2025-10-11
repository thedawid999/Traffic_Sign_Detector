import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay # type: ignore
import matplotlib.pyplot as plt # type: ignore

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", tf.config.list_physical_devices("GPU"))

train_dir = "./data/images/cnn/train/"
val_dir = "./data/images/cnn/val/"
logdir = "logs"
imgsz = 32
batch = 32
epochs = 50

# ---------- CREATE DATASETS FOR TRAINING AND VALIDATION ---------- #
train_ds_augmented = ImageDataGenerator(
    rescale=1./255,    # map RGB-values from 0 - 255 to 0.0 - 1.0
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0
)

ds_rescaled = ImageDataGenerator(
    rescale=1./255, # map RGB-values from 0 - 255 to 0.0 - 1.0
)

train_ds = train_ds_augmented.flow_from_directory(
    train_dir,
    target_size=(imgsz, imgsz),
    batch_size=batch,
    class_mode="sparse",
    shuffle=True
)
val_ds = ds_rescaled.flow_from_directory(
    val_dir,
    target_size=(imgsz, imgsz),
    batch_size = batch,
    class_mode = "sparse",
    shuffle = False,
)

# ---------- CREATE CNN-MODEL ---------- #
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation="relu", input_shape=(imgsz,imgsz,3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), 1, activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), 1, activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(43, activation="softmax"))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])
model.summary()


# ---------- TRAINING ---------- #
precisions = []
recalls = []
f1_scores = []
losses = []
val_losses = []
accuracies = []
val_accuracies = []

# calculate Precision, Recall, F1-Score for every epoch
def on_epoch_end(epoch, logs):
    losses.append(logs.get('loss'))
    val_losses.append(logs.get('val_loss'))
    accuracies.append(logs.get('accuracy'))
    val_accuracies.append(logs.get('val_accuracy'))
    
    y_true = val_ds.labels
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

    print(f"Epoch {epoch+1}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

# calculate and display Confusion-Matrix for the last epoch only
def on_train_end(self, logs=None):
    y_true = val_ds.labels
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    cm_global = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20)) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_global)
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title("Confusion Matrix - Letzte Epoche")
    plt.savefig('code/cnn/plots/confusion_matrix.png') 
    plt.show()

lambda_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end = on_epoch_end, on_train_end = on_train_end)
results = model.fit(train_ds, epochs = epochs, validation_data = val_ds, callbacks = [lambda_callback])


# ---------- SAVE MODEL ---------- #
model.save("code/cnn/cnn_classifier.h5")


# ---------- PLOT LOSS AND ACCURACY ---------- #
epochs = range(1, len(losses) + 1)

fig = plt.figure()
plt.plot(epochs, losses, color='blue', label='loss')
plt.plot(epochs, val_losses, color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.savefig('code/cnn/plots/loss_plot.png')
plt.show()

fig = plt.figure()
plt.plot(epochs, accuracies, color='blue', label='accuracy')
plt.plot(epochs, val_accuracies, color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.savefig('code/cnn/plots/accuracy_plot.png') 
plt.show()


# ---------- PLOT PRECISION, RECALL, F1-SCORE AND CONFUSION MATRIX ---------- #
plt.plot(epochs, precisions, label='Precision')
plt.plot(epochs, recalls, label='Recall')
plt.plot(epochs, f1_scores, label='F1 Score')

plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Precision, Recall and F1 Score per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('code/cnn/plots/evaluation_plot.png') 
plt.show()
