
            #Deep_learning answer ----- 1

import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Common CNN Model Parameters
input_shape = (28, 28, 1)
num_classes = 10
max_params = 8000

# Model 1: Simple CNN
model1 = models.Sequential()
model1.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(128, activation='relu'))
model1.add(layers.Dense(num_classes, activation='softmax'))

# Model 2: CNN with Strided Convolution
model2 = models.Sequential()
model2.add(layers.Conv2D(8, (3, 3), strides=(2, 2), activation='relu', input_shape=input_shape))
model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(num_classes, activation='softmax'))

# Model 3: CNN with Depthwise Separable Convolution
model3 = models.Sequential()
model3.add(layers.DepthwiseConv2D(3, activation='relu', input_shape=input_shape))
model3.add(layers.Conv2D(16, (1, 1), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Flatten())
model3.add(layers.Dense(64, activation='relu'))
model3.add(layers.Dense(num_classes, activation='softmax'))

# Compile and train the models
models = [model1, model2, model3]
batch_size = 128
epochs = 10
evaluation_results = []

for i, model in enumerate(models):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    evaluation_results.append({'Model': f"Model {i+1}", 'Loss': loss, 'Accuracy': accuracy})
    
    # Check model parameters
    num_params = model.count_params()
    if num_params > max_params:
        print(f"Model {i+1} has {num_params} parameters, exceeding the limit of {max_params} parameters.")

# Print the comparison table
print("\nComparison Table:")
print("-----------------")
print("{:<8} {:<10} {:<10}".format('Model', 'Loss', 'Accuracy'))
print("-----------------")
for result in evaluation_results:
    print("{:<8} {:<10.4f} {:<10.2%}".format(result['Model'], result['Loss'], result['Accuracy']))
