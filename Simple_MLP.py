import tensorflow as tf

# Load Dataset (Minst usde for simplicity)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

NUM_CLASSES = 10 # Number of classes in MNIST

# MLP
inputs = tf.keras.Input(shape=(28, 28), name="input") # <--- Specifys input shape, needs to be chnaged for our point cloud (shape=(Points, Dimensions))
x = tf.keras.layers.Flatten(name="flatten")(inputs) # <--- Flattens the input (order-sensitive), for point cloud we should use somethign else (mean/std pooling)
x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="logits")(x) # <--- NUM_CLASSES needs to be defined based on the dataset
model = tf.keras.Model(inputs, outputs, name="mlp_mnist_base")

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Training and Evaluation
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

