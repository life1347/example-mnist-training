import os
import tensorflow as tf
mnist = tf.keras.datasets.mnist
batch_size = int(os.environ.get('BATCH_SIZE', 32))
epochs = int(os.environ.get('EPOCHS', 5))

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# batch size: https://stackoverflow.com/a/52215772
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
model.evaluate(x_test, y_test)
