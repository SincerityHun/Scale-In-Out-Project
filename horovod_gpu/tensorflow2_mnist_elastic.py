import tensorflow as tf
import horovod.tensorflow as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Setting GPU memory growth to true to avoid consuming all memory.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Set the visible device to the GPU corresponding to the Horovod local rank
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    # Print which GPU is being used
    print(f"Process {hvd.rank()} is using GPU {gpus[hvd.local_rank()].name}")
else:
    print("No GPUs found, running on CPU.")

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

# --------------------------------------------------------

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

# Horovod: adjust learning rate based on number of GPUs.
lr = 0.001
opt = tf.optimizers.Adam(lr * hvd.size())


@tf.function
def training_step(images, labels, allreduce=True):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Horovod: add Horovod Distributed GradientTape.
    if allreduce:
        tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    return loss_value


# Horovod: initialize model and optimizer state so we can synchronize across workers
for batch_idx, (images, labels) in enumerate(dataset.take(1)):
    training_step(images, labels, allreduce=False)


@hvd.elastic.run
def train(state):
    start_batch = state.batch

    # Horovod: adjust number of steps based on number of GPUs.
    for batch_idx, (images, labels) in enumerate(dataset.skip(state.batch).take(10000 // hvd.size())):
        state.batch = start_batch + batch_idx
        loss_value = training_step(images, labels)

        if state.batch % 10 == 0 and hvd.local_rank() == 0:
            print('Step #%d\tLoss: %.6f' % (state.batch, loss_value))

        # Horovod: commit state at the end of each batch
        state.commit()


def on_state_reset():
    opt.lr.assign(lr * hvd.size())

state = hvd.elastic.TensorFlowKerasState(mnist_model, opt, batch=0)
state.register_reset_callbacks([on_state_reset])

train(state)

checkpoint_dir = './checkpoints'
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)
