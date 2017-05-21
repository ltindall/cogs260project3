
# coding: utf-8

# # Problem 1

# ## 1.a

# In[40]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

train = []
train_labels = []
train_both = []
file = open('iris/iris_train.data', 'r') 
for line in file: 
    l = line.rstrip('\n').split(',')
    
    nums = [float(x) for x in l[:4]]
    train_labels.append(l[-1])
    nums.append(l[-1])
    train_both.append(nums)
    train.append(nums[0:-1])
    
    
test = []
test_labels = []
test_both = []
file = open('iris/iris_test.data', 'r') 
for line in file: 
    l = line.rstrip('\n').split(',')
    
    nums = [float(x) for x in l[:4]]
    test_labels.append(l[-1])
    nums.append(l[-1])
    test_both.append(nums)
    test.append(nums[0:-1])

test = np.array(test)
test_labels = np.array(test_labels)
test_both = np.array(test_both)
#print train_both
#print train
train = np.array(train)
train_labels = np.array(train_labels)

train, train_labels = shuffle(train, train_labels, random_state=0)

unique_labels = np.unique(train_labels)



sepal_length_setosa = [x[0] for x in train_both if x[-1] == unique_labels[0]]
sepal_width_setosa = [x[1] for x in train_both if x[-1] == unique_labels[0]]
petal_length_setosa = [x[2] for x in train_both if x[-1] == unique_labels[0]]
petal_width_setosa = [x[3] for x in train_both if x[-1] == unique_labels[0]]

sepal_length_versicolor = [x[0] for x in train_both if x[-1] == unique_labels[1]]
sepal_width_versicolor = [x[1] for x in train_both if x[-1] == unique_labels[1]]
petal_length_versicolor = [x[2] for x in train_both if x[-1] == unique_labels[1]]
petal_width_versicolor = [x[3] for x in train_both if x[-1] == unique_labels[1]]


plt.figure(figsize=(10,10)),plt.subplot(231)
plt.plot(sepal_length_setosa, sepal_width_setosa, 'g^',label='setosa')
plt.plot(sepal_length_versicolor, sepal_width_versicolor, 'bo',label='versicolor')
plt.legend(loc='upper left')
plt.xlabel('sepal length')
plt.ylabel('sepal width')


plt.subplot(232),plt.plot(sepal_length_setosa, petal_length_setosa, 'g^',label='setosa')
plt.plot(sepal_length_versicolor, petal_length_versicolor, 'bo',label='versicolor')
plt.legend(loc='upper left')
plt.xlabel('sepal length')
plt.ylabel('petal length')


plt.subplot(233)
plt.plot(sepal_length_setosa, petal_width_setosa, 'g^',label='setosa')
plt.plot(sepal_length_versicolor, petal_width_versicolor, 'bo',label='versicolor')
plt.legend(loc='upper left')
plt.xlabel('sepal length')
plt.ylabel('petal width')


plt.subplot(234)
plt.plot(sepal_width_setosa, petal_length_setosa, 'g^',label='setosa')
plt.plot(sepal_width_versicolor, petal_length_versicolor, 'bo',label='versicolor')
plt.legend(loc='upper left')
plt.xlabel('sepal width')
plt.ylabel('petal length')


plt.subplot(235)
plt.plot(sepal_width_setosa, petal_width_setosa, 'g^',label='setosa')
plt.plot(sepal_width_versicolor, petal_width_versicolor, 'bo',label='versicolor')
plt.legend(loc='upper left')
plt.xlabel('sepal width')
plt.ylabel('petal width')

plt.subplot(236)
plt.plot(petal_length_setosa, petal_width_setosa, 'g^',label='setosa')
plt.plot(petal_length_versicolor, petal_width_versicolor, 'bo',label='versicolor')
plt.legend(loc='upper left')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()







# ## 1.b

# In[ ]:



def perceptronAccuracy(data, labels, weights):

    missed = 0

    testWithBias = np.append(np.ones((len(data),1)),data,axis=1)

    for sample,label in zip(testWithBias,labels): 
        predicted = int(np.dot(sample,weights) >= 0)
        
    
        if predicted != label: 
            missed = missed + 1

    correct = len(labels) - missed 
    accuracy = 1.0 * correct / len(labels)
    print "Perceptron classifiction accuracy on test set = ",accuracy * 100,"%"






def trainPerceptron(learn_rate, data, labels): 
    weights = np.array([0.0 for x in range(data.shape[1]+1)])
    errors = 1
    iterations = 0
    
    trainWithBias = np.append(np.ones((len(data),1)),data,axis=1)
    
    while errors: 
        errors = 0
        iterations = iterations + 1 

        for sample,label in zip(trainWithBias,labels): 

            predicted = int(np.dot(sample,weights) >= 0)

            error = label - predicted

            if error != 0: 
                errors = errors + 1

            weights = weights + (learn_rate * error * sample)
            
    print "Number of iterations till convergence = ",iterations 
    return weights 
    
    
    
learning_rate = 0.1

print "learning rate = ",learning_rate

train_labels_perceptron = np.array([0 if x == unique_labels[0] else 1 for x in train_labels])
test_labels_perceptron = np.array([0 if x == unique_labels[0] else 1 for x in test_labels])

w = trainPerceptron(learning_rate, train, train_labels_perceptron)        
perceptronAccuracy(test, test_labels_perceptron, w)

    


# ## 1.c

# In[ ]:

# Z-score the training and test data 

trainZScored = np.copy(train)
testZScored = np.copy(test)

for i in range(train.shape[1]): 
    m = np.mean(train[:,i])
    s = np.std(train[:,i])
    
    for x in range(len(trainZScored[:,i])):
        
        trainZScored[x,i] = (trainZScored[x,i] - m) / s

    for x in range(len(testZScored[:,i])): 
        testZScored[x,i] = (testZScored[x,i] - m) / s
        
        
        
learning_rate = 0.1

print "learning rate = ",learning_rate
wZScored = trainPerceptron(learning_rate, trainZScored, train_labels_perceptron)        
perceptronAccuracy(testZScored, test_labels_perceptron, wZScored)

    
    
    


# # Problem 2

# In[ ]:

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_samples = []
training_labels = []
for input,label in training_data: 
    training_samples.append(input)
    training_labels.append(label)
    
test_samples = []
test_labels = []

for input,label in test_data: 
    test_samples.append(input)
    test_labels.append(label)
    
training_samples = np.array(training_samples).reshape((50000,1,784))
training_labels = np.array(training_labels).reshape((50000,10))
test_samples = np.array(test_samples).reshape((10000,1,784))
test_labels = np.array(test_labels)


# In[ ]:



class FeedForwardNeuralNetwork: 
    def __init__(self, size_input, size_hidden, size_output):
        
        self.input_size = size_input
        self.hidden_size = size_hidden
        self.output_size = size_output
        
        self.hidden_weight = np.random.randn(self.input_size, self.hidden_size)
        self.output_weight = np.random.randn(self.hidden_size, self.output_size)
        
        self.hidden_bias = np.random.randn(1,self.hidden_size) 
        self.output_bias = np.random.randn(1,self.output_size)
        
        self.output_velocity = np.zeros(self.output_weight.shape)
        self.hidden_velocity = np.zeros(self.hidden_weight.shape)
        
        self.output_bias_velocity = np.zeros(self.output_bias.shape)
        self.hidden_bias_velocity = np.zeros(self.hidden_bias.shape)
        
        self.training_x = [0]
        self.training_y = [1]
        
        self.test_x = [0]
        self.test_y = [1]
        
    def feed_forward(self, input): 
        
        self.current_inputs = input
        
        # calculate hidden output -> output = (Inputs * weights) + biases
        # apply activation function -> hidden activation = sigmoid(output)
        
        # (1,5)
        self.hidden_activations = self.sigmoid(np.dot(input,self.hidden_weight) + self.hidden_bias)
        
        
        # calculate output output -> output = (hidden activations * weights) + biases
        # apply activation function -> output activation = sigmoid(output)
        self.output_activations = self.sigmoid(np.dot(self.hidden_activations,self.output_weight) + self.output_bias)
        
        return self.output_activations 
    
    def feed_forward_regularization(self, input, p): 
        
        self.current_inputs = input
        
        # calculate hidden output -> output = (Inputs * weights) + biases
        # apply activation function -> hidden activation = sigmoid(output)
        
        # (1,5)
        self.hidden_activations = self.sigmoid(np.dot(input,self.hidden_weight) + self.hidden_bias)
        
        
        drop_hidden = (np.random.rand(*self.hidden_activations.shape) < p ) 
        
        self.hidden_activations = self.hidden_activations * drop_hidden
        
        
        # calculate output output -> output = (hidden activations * weights) + biases
        # apply activation function -> output activation = sigmoid(output)
        self.output_activations = self.sigmoid(np.dot(self.hidden_activations,self.output_weight) + self.output_bias)
        
        return self.output_activations 
        
    # alpha = Learning rate 
    def backpropagation_mom(self, output, alpha, mu):

        target = output
        
        #(1,10)
        output_delta = (self.output_activations - target) * self.derivative_sigmoid(self.output_activations)
        
        #(1,5)
        hidden_delta = (np.dot(output_delta,self.output_weight.T)) * self.derivative_sigmoid(self.hidden_activations)
        
        #(1,784) 
        hidden_gradient = np.dot(self.current_inputs.T,hidden_delta)
        
        # (1,5)(1,10)
        output_gradient = np.dot(self.hidden_activations.T, output_delta)
        
        output_bias_gradient = output_delta
        hidden_bias_gradient = hidden_delta
        
        self.output_velocity = self.output_velocity * mu + (-1 * alpha * output_gradient)
        self.output_weight = self.output_weight + self.output_velocity
        
        self.hidden_velocity = self.hidden_velocity * mu + (-1 * alpha * hidden_gradient)
        self.hidden_weight = self.hidden_weight + self.hidden_velocity
        
        #self.output_weight = self.output_weight + (-1 * alpha * output_gradient)
        #self.hidden_weight = self.hidden_weight + (-1 * alpha * hidden_gradient)
        
        self.output_bias_velocity = self.output_bias_velocity * mu + (-1 * alpha * output_bias_gradient)
        self.output_bias = self.output_bias + self.output_bias_velocity
        
        self.hidden_bias_velocity = self.hidden_bias_velocity * mu + (-1 * alpha * hidden_bias_gradient)
        self.hidden_bias = self.hidden_bias + self.hidden_bias_velocity
        
        #self.output_bias = self.output_bias + (-1 * alpha * output_bias_gradient)
        #self.hidden_bias = self.hidden_bias + (-1 * alpha * hidden_bias_gradient)
        
    # alpha = Learning rate 
    def backpropagation(self, output, alpha):
        target = output
        
        #(1,10)
        output_delta = (self.output_activations - target) * self.derivative_sigmoid(self.output_activations)
        
        #(1,5)
        hidden_delta = (np.dot(output_delta,self.output_weight.T)) * self.derivative_sigmoid(self.hidden_activations)
        
        #(1,784) 
        hidden_gradient = np.dot(self.current_inputs.T,hidden_delta)
        
        # (1,5)(1,10)
        output_gradient = np.dot(self.hidden_activations.T, output_delta)
        
        output_bias_gradient = output_delta
        hidden_bias_gradient = hidden_delta
        
        self.output_weight = self.output_weight + (-1 * alpha * output_gradient)
        self.hidden_weight = self.hidden_weight + (-1 * alpha * hidden_gradient)
        
        self.output_bias = self.output_bias + (-1 * alpha * output_bias_gradient)
        self.hidden_bias = self.hidden_bias + (-1 * alpha * hidden_bias_gradient)
    
    def predict(self,input, withRegMom, p): 
        
        predictions = []
        for i in input: 
            if withRegMom: 
                predictions.append(self.feed_forward_regularization(i,p))
            else: 
                predictions.append(self.feed_forward(i))
        return np.array(predictions)
            
    def train(self, training_samples, training_labels, test_samples, test_labels, alpha, epochs, withRegMom, p, mu):
        
        iterations = 0
        for i in range(epochs): 
            
            for input,label in zip(training_samples, training_labels):
                if withRegMom: 
                    self.feed_forward_regularization(input,p)
                    self.backpropagation_mom(label,alpha,mu)
                else: 
                    self.feed_forward(input)
                    self.backpropagation(label, alpha)
                iterations = iterations + 1
            
        
            #predictions = np.argmax(np.asarray(self.predict(training_samples)), axis=1)
           
            targets = np.argmax(np.asarray(training_labels), axis=1)
            predictions = self.predict(training_samples, withRegMom, p)
            
            
            errors = 0
            for target,predicted in zip(targets,predictions): 
                predicted = np.argmax(np.asarray(predicted), axis=1)
                if target != predicted: 
                    errors = errors + 1
            error_rate = 1.0 * errors / len(training_labels)
            
            print iterations,' iterations, training error rate = ',error_rate
            
            self.training_x.append(self.training_x[-1]+1)
            self.training_y.append(error_rate)
            
            
            
            targets = test_labels
            predictions = self.predict(test_samples, withRegMom, p)
            
            
            errors = 0
            for target,predicted in zip(targets,predictions): 
                predicted = np.argmax(np.asarray(predicted), axis=1)
                if target != predicted: 
                    errors = errors + 1
            error_rate = 1.0 * errors / len(test_labels)
            
            self.test_x.append(self.test_x[-1]+1)
            self.test_y.append(error_rate)
            
            print iterations,' iterations, test error rate = ',error_rate
            
    
                
    def sigmoid(self, t):
        
        return 1.0/(1.0+np.exp(-t))

    def derivative_sigmoid(self, t): 
        return t * (1.0 - t)


# ## 2.1

# In[ ]:



ffnet = FeedForwardNeuralNetwork(784, 20, 10)
ffnet.train(training_samples,training_labels,test_samples,test_labels,0.1,60, False, 0,0)
#ffnetRegMom.train(training_samples,training_labels,test_samples,test_labels,0.1,100, True, 0.95,0.7)


# In[ ]:

plt.figure(figsize=(10,5)),plt.subplot(121),
plt.plot(ffnet.training_x[1:],ffnet.training_y[1:],'b', label='training error')
plt.plot(ffnet.test_x[1:],ffnet.test_y[1:],'r', label='test error')
plt.title('Training and test error rate on feedforward network')
plt.xlabel('epochs')
plt.ylabel('error rate')
plt.legend(loc='upper center', shadow=True)
plt.ylim(min(min(ffnet.training_y[1:]),min(ffnet.test_y[1:])),max(max(ffnet.training_y[1:]),max(ffnet.test_y[1:])))

plt.show()


# ## 2.3

# In[ ]:

ffnetRegMom = FeedForwardNeuralNetwork(784, 20, 10)
#ffnet.train(training_samples,training_labels,test_samples,test_labels,0.1,60, False, 0,0)
ffnetRegMom.train(training_samples,training_labels,test_samples,test_labels,0.1,60, True, 0.95,0.7)


# In[ ]:

plt.figure(figsize=(10,5)),plt.subplot(121),
plt.plot(ffnetRegMom.training_x[1:],ffnetRegMom.training_y[1:],'b', label='training error')
plt.plot(ffnetRegMom.test_x[1:],ffnetRegMom.test_y[1:],'r', label='test error')
plt.title('Training and test error rate on network with regularization and momentum')
plt.xlabel('epochs')
plt.ylabel('error rate')
plt.legend(loc='upper center', shadow=True)
plt.ylim(min(min(ffnetRegMom.training_y[1:]),min(ffnetRegMom.test_y[1:])),max(max(ffnetRegMom.training_y[1:]),max(ffnetRegMom.test_y[1:])))

plt.show()


# In[ ]:

class FeedForwardNeuralNetwork2Hidden: 
    def __init__(self, size_input, size_hidden1, size_hidden2, size_output):
        
        self.input_size = size_input
        self.hidden1_size = size_hidden1
        self.hidden2_size = size_hidden2
        self.output_size = size_output
        
        self.hidden1_weight = np.random.randn(self.input_size, self.hidden1_size)
        self.hidden2_weight = np.random.randn(self.hidden1_size, self.hidden2_size)
        self.output_weight = np.random.randn(self.hidden2_size, self.output_size)
        
        self.hidden1_bias = np.random.randn(1,self.hidden1_size) 
        self.hidden2_bias = np.random.randn(1,self.hidden2_size) 
        self.output_bias = np.random.randn(1,self.output_size)

        self.training_x = [0]
        self.training_y = [1]
        
        self.test_x = [0]
        self.test_y = [1]

        
       
    def feed_forward(self, input): 
        
        self.current_inputs = input
        
        # calculate hidden output -> output = (Inputs * weights) + biases
        # apply activation function -> hidden activation = sigmoid(output)
        
        # (1,5)
        self.hidden1_activations = self.sigmoid(np.dot(input,self.hidden1_weight) + self.hidden1_bias)
        
        
        self.hidden2_activations = self.sigmoid(np.dot(self.hidden1_activations,self.hidden2_weight) + self.hidden2_bias)
        
        
        # calculate output output -> output = (hidden activations * weights) + biases
        # apply activation function -> output activation = sigmoid(output)
        self.output_activations = self.sigmoid(np.dot(self.hidden2_activations,self.output_weight) + self.output_bias)
        
        return self.output_activations 
    
   
        
    # alpha = Learning rate 
    def backpropagation(self, output, alpha):
        target = output
        
        #(1,10)
        output_delta = (self.output_activations - target) * self.derivative_sigmoid(self.output_activations)
        
        #(1,5)
        hidden2_delta = (np.dot(output_delta,self.output_weight.T)) * self.derivative_sigmoid(self.hidden2_activations)
        
        hidden1_delta = (np.dot(hidden2_delta,self.hidden2_weight.T)) * self.derivative_sigmoid(self.hidden1_activations)
        
        # (1,5)(1,10)
        output_gradient = np.dot(self.hidden2_activations.T, output_delta)
        
        #(1,784) 
        hidden2_gradient = np.dot(self.hidden1_activations.T,hidden2_delta)
        
        hidden1_gradient = np.dot(self.current_inputs.T,hidden1_delta)
        
       
        
        output_bias_gradient = output_delta
        hidden2_bias_gradient = hidden2_delta
        hidden1_bias_gradient = hidden1_delta
        
        self.output_weight = self.output_weight + (-1 * alpha * output_gradient)
        self.hidden2_weight = self.hidden2_weight + (-1 * alpha * hidden2_gradient)
        self.hidden1_weight = self.hidden1_weight + (-1 * alpha * hidden1_gradient)
        
        self.output_bias = self.output_bias + (-1 * alpha * output_bias_gradient)
        self.hidden2_bias = self.hidden2_bias + (-1 * alpha * hidden2_bias_gradient)
        self.hidden1_bias = self.hidden1_bias + (-1 * alpha * hidden1_bias_gradient)
    
    def predict(self,input): 
        
        predictions = []
        for i in input: 
            predictions.append(self.feed_forward(i))
        return np.array(predictions)
            
    def train(self, training_samples, training_labels, test_samples, test_labels, alpha, epochs):
        
        iterations = 0
        for i in range(epochs): 
            
            for input,label in zip(training_samples, training_labels):
                self.feed_forward(input)
                self.backpropagation(label, alpha)
                iterations = iterations + 1
            
        
            #predictions = np.argmax(np.asarray(self.predict(training_samples)), axis=1)
           
            targets = np.argmax(np.asarray(training_labels), axis=1)
            predictions = self.predict(training_samples)
            
            
            errors = 0
            for target,predicted in zip(targets,predictions): 
                predicted = np.argmax(np.asarray(predicted), axis=1)
                if target != predicted: 
                    errors = errors + 1
            error_rate = 1.0 * errors / len(training_labels)
            self.training_x.append(self.training_x[-1]+1)
            self.training_y.append(error_rate)
            
            
            print iterations,' iterations, training error rate = ',error_rate
            
            
            targets = test_labels
            predictions = self.predict(test_samples)
            
            
            errors = 0
            for target,predicted in zip(targets,predictions): 
                predicted = np.argmax(np.asarray(predicted), axis=1)
                if target != predicted: 
                    errors = errors + 1
            error_rate = 1.0 * errors / len(test_labels)
            
            self.test_x.append(self.test_x[-1]+1)
            self.test_y.append(error_rate)
            
            print iterations,' iterations, test error rate = ',error_rate
            
            
            


    
                
    def sigmoid(self, t):
        
        return 1.0/(1.0+np.exp(-t))

    def derivative_sigmoid(self, t): 
        return t * (1.0 - t)


# ## 2.2

# In[ ]:


ffnet = FeedForwardNeuralNetwork2Hidden(784, 20, 20, 10)
ffnet.train(training_samples,training_labels,test_samples,test_labels,0.1,60)


# In[ ]:


plt.figure(figsize=(10,5)),plt.subplot(121),
plt.plot(ffnet.training_x[1:],ffnet.training_y[1:],'b', label='training error')
plt.plot(ffnet.test_x[1:],ffnet.test_y[1:],'r', label='test error')
plt.title('Training and test error rate on network with 2 hidden layers')
plt.xlabel('epochs')
plt.ylabel('error rate')
plt.legend(loc='upper center', shadow=True)
plt.ylim(min(min(ffnet.training_y[1:]),min(ffnet.test_y[1:])),max(max(ffnet.training_y[1:]),max(ffnet.test_y[1:])))

plt.show()


# # Problem 3

# In[ ]:

from keras.datasets import mnist
mnist.load_data()


# ## CNN with stochastic gradient descent

# In[42]:

# modified from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

num_classes = 10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# first convolutional layer
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))

# second convolutional layer
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

# first max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# third convolutional layer
#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(Activation('relu'))

# fourth convolutional layer
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))

# second max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True)

lrate = 0.01
epochs = 30
decay = lrate/epochs
sgd = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

batch_size = 32




results = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[43]:

plt.figure(figsize=(10,5)),plt.subplot(121),
x1 = range(epochs)
train_loss = results.history['loss']
x2 = range(epochs)
test_loss = results.history['val_loss']
plt.plot(x1,train_loss,'b', label='training loss')
plt.plot(x2,test_loss,'r', label='test loss')
plt.title('Training and test loss for convolutional neural network with SGD')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper center', shadow=True)
plt.ylim(min(min(train_loss),min(test_loss)),max(max(train_loss),max(test_loss)))

plt.show()


# ## CNN with bath normalization 

# In[ ]:

# modified from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


num_classes = 10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# first convolutional layer
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))

# second convolutional layer
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# first max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))

# third convolutional layer
#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))

# fourth convolutional layer
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))

# second max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True)

lrate = 0.01
epochs = 30
decay = lrate/epochs
sgd = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

batch_size = 32




results2 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:

plt.figure(figsize=(10,5)),plt.subplot(121),
x1 = range(epochs)
train_loss = results2.history['loss']
x2 = range(epochs)
test_loss = results2.history['val_loss']
plt.plot(x1,train_loss,'b', label='training loss')
plt.plot(x2,test_loss,'r', label='test loss')
plt.title('Training and test loss for convolutional neural network with batch normalization')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper center', shadow=True)
plt.ylim(min(min(train_loss),min(test_loss)),max(max(train_loss),max(test_loss)))

plt.show()


# ## CNN with global average pooling

# In[44]:

# modified from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Embedding, LSTM
from keras.layers.normalization import BatchNormalization


num_classes = 10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# first convolutional layer
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
#model.add(BatchNormalization())
model.add(Activation('relu'))

# second convolutional layer
model.add(Conv2D(32, (3, 3)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

# first max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(BatchNormalization())
model.add(Dropout(0.25))

# third convolutional layer
#model.add(Conv2D(64, (3, 3), padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))

# fourth convolutional layer
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))

# second max pooling
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# fully connected
#model.add(Flatten())
#model.add(Dense(512))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes))
#model.add(BatchNormalization())
model.add(Activation('softmax'))


# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True)

lrate = 0.01
epochs = 50
decay = lrate/epochs
sgd = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

batch_size = 32


results2 = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[45]:

plt.figure(figsize=(10,5)),plt.subplot(121),
x1 = range(epochs)
train_loss = results2.history['loss']
x2 = range(epochs)
test_loss = results2.history['val_loss']
plt.plot(x1,train_loss,'b', label='training loss')
plt.plot(x2,test_loss,'r', label='test loss')
plt.title('Training and test loss for CNN with Global Average Pooling ')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper center', shadow=True)
plt.ylim(min(min(train_loss),min(test_loss)),max(max(train_loss),max(test_loss)))

plt.show()

