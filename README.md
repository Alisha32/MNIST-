# MNIST-
Analyzing MNIST dataset using Machine Learning
MNIST is a  DATSET WITH 70,000 IMAGES, WE HAVE TO PREDICT THE NUMBERS
NUMBERS CAN BE 0,1,2,3,4,5,6,7,8,9
DEFINE AN ALGORITHM TO DETECT WHICH NUMBER IS WRITTEN
CLASSIFICATION PROBLEM WITH 10 CLASSES

#importing relevant libraraies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

mnist_dataset,mnist_info=tfds.load(name='mnist',with_info=True,as_supervised=True)
#with_info gives whole info about the dataset which gets stored in the mnist_info.
# as_supervised spits the data into inputs and targets.

mnist_train,mnist_test=mnist_dataset['train'],mnist_dataset['test']
#validation?--- 10 % of training dataset
num_validation_samples = 0.1*mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples,tf.int64)
#(tf.cast==converts into specified type)
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples,tf.int64)
#SCALING 
def scale(image,label):
  image = tf.cast(image,tf.float32)
  image/=255.
  return image,label
scaled_train_and_validation_data=mnist_train.map(scale)
test_data=mnist_test.map(scale)
#SHUFFLING
buffer_size=10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(buffer_size)
#validation data and train data splitting
validation_data=shuffled_train_and_validation_data.take(num_validation_samples)
train_data=shuffled_train_and_validation_data.skip(num_validation_samples)
#BATCHING
batch_size=100
train_data=train_data.batch(batch_size)
validation_data=validation_data.batch(num_validation_samples)
test_data=test_data.batch(num_test_samples)

#AS SUPERVISED = TRUE , 2-tuples, inputs and targets
validation_inputs,validation_targets=next(iter(validation_data))
#'''next=load next batch
#iter = make data iterable(for and while loop as)'''

#OUTLINING THE MODEL
input_size=784
output_size=10
hidden_layer_size=50
model= tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
        tf.keras.layers.Dense(output_size,activation='Softmax')
])

#tf.keras.Sequential--stack the layers
#tf.keras.layers.Flatten--trnaforms into vectors readable
#tf.keras.layers.Dense(sizes,activation functions to be used)


#LEARNING
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#.compile takes arguments as optimizer and loss function and metrics such as accuracy to know

#TRAINING
NUM_EPOCHS=5
model.fit(train_data,epochs=NUM_EPOCHS,validation_data=(validation_inputs,validation_targets),verbose=2)
#.fit fits the dataset and at starting of each epoch,training loss=0
# iterate over preset number of batches
#weights and biases are updated as many times as tehir are batches
# velue of loss function
# training is going
#training accuarcy
# at end of each epoch,will forward propagate validation dataset.
OUTPUT:
Epoch 1/5
540/540 - 13s - loss: 0.4031 - accuracy: 0.8861 - val_loss: 0.2267 - val_accuracy: 0.9373 - 13s/epoch - 24ms/step
Epoch 2/5
540/540 - 6s - loss: 0.1877 - accuracy: 0.9456 - val_loss: 0.1670 - val_accuracy: 0.9523 - 6s/epoch - 11ms/step
Epoch 3/5
540/540 - 5s - loss: 0.1450 - accuracy: 0.9573 - val_loss: 0.1356 - val_accuracy: 0.9618 - 5s/epoch - 9ms/step
Epoch 4/5
540/540 - 6s - loss: 0.1217 - accuracy: 0.9635 - val_loss: 0.1199 - val_accuracy: 0.9645 - 6s/epoch - 10ms/step
Epoch 5/5
540/540 - 5s - loss: 0.1018 - accuracy: 0.9700 - val_loss: 0.1038 - val_accuracy: 0.9692 - 5s/epoch - 9ms/step
<keras.callbacks.History at 0x7f71f22bbf40>

#TESTING
test_loss,test_accuracy=model.evaluate(test_data)
# determines te loss and accuracy metrics for the testig data
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
#compare with the validation accuracy!!
OUTPUT:
1/1 [==============================] - 1s 899ms/step - loss: 0.1126 - accuracy: 0.9664
Test loss: 0.11. Test accuracy: 96.64%

# testing accuarcy is 96.64% whereas the validation accuracy is 96.92%
# The two are quite close, our model is well defined
