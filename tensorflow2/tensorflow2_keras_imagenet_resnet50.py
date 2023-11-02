import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize Horovod
hvd.init()

# Configure GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# ResNet50 expects 3-channel images of shape 224x224, CIFAR-10 images are 32x32x3
# We will upsample the images to the correct shape
train_images = tf.image.resize(train_images, [224, 224])
test_images = tf.image.resize(test_images, [224, 224])

# Normalize the images to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to one-hot encoded format
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train_images)

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
steps_per_epoch = len(train_images) // hvd.size() // 32

state = hvd.elastic.KerasState(model, batch=32, epoch=0)
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
    model.fit(datagen.flow(train_images, train_labels, batch_size=32),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              epochs=state.epoch,
              verbose=1 if hvd.rank() == 0 else 0,
              validation_data=(test_images, test_labels))

train(state)

