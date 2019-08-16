from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Model
import datetime


class Model(Model):
	def __init__(self):
		super(Model, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(16, [5, 5], input_shape=(28, 28, 1))
		self.pool1 = tf.keras.layers.MaxPool2D([2, 2], 2)
		self.conv2 = tf.keras.layers.Conv2D(32, [5, 5])
		self.pool2 = tf.keras.layers.MaxPool2D([2, 2], 2)
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(256, activation='relu')
		self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
		self.dropout1 = tf.keras.layers.Dropout(0.2)

	def call(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dropout1(x)
		return self.dense2(x)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load train and test data, x = data, y = lables
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalise pixel values to between 0 and 1

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
	(x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model = Model()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
	predictions = model(images)
	t_loss = loss_object(labels, predictions)

	test_loss(t_loss)
	test_accuracy(labels, predictions)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

EPOCHS = 5

for epoch in range(EPOCHS):
	for images, labels in train_ds:
		train_step(images, labels)
	with train_summary_writer.as_default():
		tf.summary.scalar('loss', train_loss.result(), step=epoch)
		tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)


	for test_images, test_labels in test_ds:
		test_step(test_images, test_labels)
	with test_summary_writer.as_default():
		tf.summary.scalar('loss', test_loss.result(), step=epoch)
		tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)


	template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
	print(template.format(epoch+1,
							train_loss.result(),
							train_accuracy.result()*100,
							test_loss.result(),
							test_accuracy.result()*100))

	# Reset the metrics for the next epoch
	train_loss.reset_states()
	train_accuracy.reset_states()
	test_loss.reset_states()
	test_accuracy.reset_states()
