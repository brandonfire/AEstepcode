#Autoencoder visulization of MINST data
#build for testing our attack
#Chengbin Hu

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn import preprocessing 
from tensorflow.keras import regularizers
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

class Mnistencoder(Model):
  def __init__(self):
    super(Mnistencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
    
      # layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
      # layers.MaxPooling2D((2, 2), padding='same'),
      # layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
      # layers.MaxPooling2D((2, 2), padding='same')
      
      layers.Dense(128, activation="relu"),#,kernel_regularizer=regularizers.l2(0.001)),
      tf.keras.layers.Dropout(rate=0.2),
      layers.Dense(64, activation="relu"),#,kernel_regularizer=regularizers.l2(0.001)),
      layers.Dense(36, activation="relu")#,kernel_regularizer=regularizers.l2(0.001))
      ])
      #layers.Dense(1, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      # layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
      # layers.UpSampling2D((2, 2)),
      # layers.Dense(64, activation="relu"),
      # layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
      # layers.UpSampling2D((2, 2)),
      # layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
      layers.Dense(64, activation="relu"),#,kernel_regularizer=regularizers.l2(0.001)),
      tf.keras.layers.Dropout(rate=0.2),
      layers.Dense(128, activation="relu"),#,kernel_regularizer=regularizers.l2(0.001)),
      
      layers.Dense(784, activation="relu")#,kernel_regularizer=regularizers.l2(0.001))
      
      ])

  

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



def showsample(img):
    plt.figure()
    plt.imshow(img,cmap='Greys')
    plt.title('sample mnist')
    plt.show()

def lossfunction(x,y):
    return abs(tf.keras.losses.KLD(x,y))

def main():
    mnist = tf.keras.datasets.mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    # x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 
    # noise = np.random.rand(x_train.shape[0],28,28,1) 
    # noise_rate = 0.2
    # noise = (noise - 0.5) * noise_rate
    # noisetrain = (x_train - 0.5) + noise + 0.5 + 0.5*noise_rate #add noise and make value non-negative
    # #noisetrain = preprocessing.normalize(noisetrain)
    # noise = np.random.rand(x_test.shape[0],28,28,1) 
    # noise_rate = 0.2
    # noise = (noise - 0.5) * noise_rate
    # noisetest = (x_test - 0.5) + noise + 0.5 + 0.5*noise_rate #add noise and make value non-negative
    #noisetest = preprocessing.normalize(noisetest)
    
    
    
    
    
    
    autoencoder = Mnistencoder()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=lossfunction)
    
    history = autoencoder.fit(x_train, x_train,epochs=20,batch_size=128,validation_data=(x_test, x_test),shuffle=True)
    #history = autoencoder.fit(noisetrain, x_train,epochs=3,batch_size=128,validation_data=(noisetest, x_test),shuffle=True)
    #autoencoder.save("trainedencoder")
    
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    
    
    #print(encoded_imgs.shape)
    showsample(x_test[555].reshape(28,28))
    showsample(encoded_imgs[555].reshape(6,6))
    
    showsample(decoded_imgs[555].reshape(28,28))
    
    
def loadandshow():
    mnist = tf.keras.datasets.mnist
    autoencoder = tf.keras.models.load_model('./trainedencoder')
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    
    
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 
    noise = np.random.rand(x_train.shape[0],28,28,1) 
    noise_rate = 1
    noise = (noise - 0.5) * noise_rate
    noisetrain = (x_train - 0.5) + noise + 0.5 + 0.5*noise_rate #add noise and make value non-negative
    #noisetrain = preprocessing.normalize(noisetrain)
    noise = np.random.rand(x_test.shape[0],28,28,1) 
    #noise_rate = 0.2
    noise = (noise - 0.5) * noise_rate
    noisetest = (x_test - 0.5) + noise + 0.5 + 0.5*noise_rate #add noise and make value 
    
    
    
    
    
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    
    
    encoded_imgs1 = autoencoder.encoder(noisetest).numpy()
    decoded_imgs1 = autoencoder.decoder(encoded_imgs1).numpy()
    
    plt.figure(figsize=(20, 4))
    n = 10
    for i in range(n):
        # display original
        ax = plt.subplot(5, n, i + 1)
        plt.imshow(x_test[550+i].reshape(28,28),cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # # display original encode
        # ax = plt.subplot(7, n, i + 1 + n)
        # plt.imshow(encoded_imgs[550+i].reshape(6, 6),cmap='Greys')
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        # display original reconstruction
        ax = plt.subplot(5, n, i + 1 + n)
        plt.imshow(decoded_imgs[550+i].reshape(28,28),cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        
        #display noise
        
        ax = plt.subplot(5, n, i + 1 + 2*n)
        plt.imshow(noise[550+i].reshape(28,28),cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #display noisetest
        ax = plt.subplot(5, n, i + 1 + 3*n)
        plt.imshow(noisetest[550+i].reshape(28,28),cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # # display noise encode
        # ax = plt.subplot(7, n, i + 1 + 5*n)
        # plt.imshow(encoded_imgs1[550+i].reshape(6, 6),cmap='Greys')
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        
        # display noise reconstruction
        ax = plt.subplot(5, n, i + 1 + 4*n)
        plt.imshow(decoded_imgs1[550+i].reshape(28,28),cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        
    plt.show()

if __name__ == "__main__":
    main()
    #loadandshow()
