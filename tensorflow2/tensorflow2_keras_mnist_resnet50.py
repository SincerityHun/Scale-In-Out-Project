import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta
import numpy as np
import os

# ADD Dynamic allocating memory
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# Initialize Horovod
hvd.init()

# Configure GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# ResNet50 expects 3-channel images, but MNIST is 1-channel. We'll repeat the channels to get 3-channel images.
train_images = np.repeat(train_images[..., np.newaxis], 3, -1)
test_images = np.repeat(test_images[..., np.newaxis], 3, -1)

# Resize images from 28x28 to 224x224, required by ResNet50
train_images = tf.image.resize(train_images, [224, 224])
test_images = tf.image.resize(test_images, [224, 224])

# Normalize the images to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoded format
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Create a dataset object
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(10000).batch(32)

# Load the ResNet50 model, with weights pre-trained on ImageNet.
model = ResNet50(weights=None, input_shape=(224, 224, 3), classes=10)

# Scale the learning rate
lr = 1.0 * hvd.size()

# Compile the model
opt = Adadelta(lr)
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

def on_state_reset():
    tf.keras.backend.set_value(model.optimizer.lr, lr * hvd.size())

# Horovod: adjust the number of steps based on the number of GPUs.
steps_per_epoch = 500 // hvd.size()

state = hvd.elastic.KerasState(model, batch=100, epoch=0)
state.register_reset_callbacks([on_state_reset])

callbacks = [
    hvd.elastic.CommitStateCallback(state),
    hvd.elastic.UpdateBatchStateCallback(state),
    hvd.elastic.UpdateEpochStateCallback(state),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3),
]

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

@hvd.elastic.run
def train(state):
    model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              epochs=state.epoch,
              verbose=1 if hvd.rank() == 0 else 0)

train(state)

