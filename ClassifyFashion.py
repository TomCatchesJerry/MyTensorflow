import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#load Fashion data
fashion_mnist=tf.keras.datasets.fashion_mnist
(train_iamges,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
#fashion image is 28*28 numpy array

class_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker",\
             "Bag","Ankle boot"]
train_iamges=train_iamges/255.0
test_images=test_images/255.0

#set layer
#需要了解各个层的意义
#需要了解激活函数
model =tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation = "relu"),
    # tf.keras.layers.Dense(64,activation = "relu"),
    tf.keras.layers.Dense(10)
])

#需要了解各种损失函数，优化器，度量方法
#set loss and optimizer
model.compile(optimizer = "adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])
# tf.keras.metrics.Metric
model.fit(train_iamges,train_labels,epochs = 10)

#verbose 是信息显示方式
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose = 2)

print("Test Accuracy:",test_acc)
print("Test loss:",test_loss)

#需要了解softmax
#将数值映射到(0,1)
#Si=e^i/Σ
probability_model=tf.keras.Sequential([model,
                                       tf.keras.layers.Softmax()])
predictions=probability_model.predict(test_images)

print(predictions[0])
print(np.sum(predictions[0]))

print(np.argmax(predictions[0]))


# Grab an image from the test dataset.
img = test_images[1]
#(28, 28)
print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
#(1, 28, 28)
print(img.shape)
predictions_single = probability_model.predict(img)

print(predictions_single)