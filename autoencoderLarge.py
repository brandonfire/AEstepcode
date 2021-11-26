#Autoencoder visulization of lung image data
import itertools
import os
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn import preprocessing 
from tensorflow.keras import regularizers
from random import random
class LungXrayEncoder(Model):
  def __init__(self):
    super(LungXrayEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
    
      layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Flatten(),
      layers.Dense(256, activation="relu"),
      
      ])
    self.decoder = tf.keras.Sequential([
          
          layers.Dense(4332, activation="relu"),
          layers.Reshape((38,38,3),input_shape=(4332,)),
          layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
          layers.UpSampling2D((2, 2)),
          layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
          layers.UpSampling2D((2, 2)),
          #layers.Dense(64, activation="relu"),
          layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
          layers.UpSampling2D((2, 2)),
          layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
          
          
          ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
      #layers.Dense(1, activation="relu")])
    #layers.dense(256), activation='relu'),
    #layers.Dense(4)
  
  #def softmaxloss(self,truth):
  #     return tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=self.call())
#generate goodfellow adversial image.
#loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad    
    
    
def showsample(img):
    plt.figure()
    plt.imshow(img,cmap='Greys')
    plt.title('Lung Xray')
    plt.show()

def randomselect (p):
    select = random()
    if select < p:
        return True
    return False



def main():
    
    pixels = 304
    IMAGE_SIZE = (pixels, pixels)

    BATCH_SIZE = 32 
    data_dir = '..\\data\\chest_xray\\train'

    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear", class_mode='input')
                       
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=False, **dataflow_kwargs)
    
    x_train = train_generator.next()[0]
    # count = 0
    # x_test = False
    # for imgs in train_generator:
        # count += 1
        # print(count)
        # if count <4:
            # x_train = np.concatenate((x_train, imgs[0]), axis=0)
        # elif count == 4:
            # x_test = train_generator.next()[0]
        # elif count <= 5:
            # x_test = np.concatenate((x_test, imgs[0]), axis=0)
        # else:
            # break
        
    
    autoencoder = LungXrayEncoder()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    history = autoencoder.fit(train_generator,epochs=20,batch_size=32,validation_data=valid_generator,shuffle=True)
        #history = autoencoder.fit(noisetrain, x_train,epochs=3,batch_size=128,validation_data=(noisetest, x_test),shuffle=True)
        #autoencoder.save("trainedencoder")
    autoencoder.save("LungencoderBiggermodel")
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    showsample(x_train[5])
    
    #showsample(encoded_imgs[5].reshape(16,16))
    showsample(decoded_imgs[5])
    loss = sum(sum(tf.keras.losses.MSE(x_train[5],decoded_imgs[5]))).numpy()
    
    print('loss after all data training', loss)



def largetrainmodeltest(x_train,autoencoder):
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    
    
    plt.figure(figsize=(20, 4))
    n = 4
    for i in range(n):
        print("loss: ",i)
        loss = sum(sum(tf.keras.losses.MSE(x_train[5+i],decoded_imgs[5+i]))).numpy()
        print(loss)
    
    for i in range(n):
        #display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_train[5+i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #display original encode
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[5+i].reshape(16, 16),cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #display original reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_imgs[5+i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        
        # #display noise
        
        # ax = plt.subplot(7, n, i + 1 + 3*n)
        # plt.imshow(shownoise[i],cmap='Greys')
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        # #display noisetest
        # ax = plt.subplot(3, n, i + 1)
        # plt.imshow(ntrain[i],cmap='Greys')
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        # # display noise encode
        # ax = plt.subplot(3, n, i + 1 + n)
        # plt.imshow(encoded_imgs1[i].reshape(16, 16),cmap='Greys')
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        
        # # display noise reconstruction
        # ax = plt.subplot(3, n, i + 1 + 2*n)
        # plt.imshow(decoded_imgs1[i])
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        
        
    plt.show()



def loadandtest():
    #autoencoder = tf.keras.models.load_model('./Lungencoder')
    autoencoder = tf.keras.models.load_model('./LungencoderBiggermodel') 
    photonumber = 6
    
    pixels = 304
    IMAGE_SIZE = (pixels, pixels)

    BATCH_SIZE = 32 
    data_dir = '..\\data\\chest_xray\\train'
    #data_dir = '.\\FlowerData\\flower_photos'
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")
                       
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=False, **dataflow_kwargs)
    
    x_train0 = train_generator.next()
    x_train = x_train0[0] #This is training input
    x_train1 = x_train0[1] #This is the training label.
    count = 0
    x_test = False
    
    #print("This is x train",x_train1)
    #print("This is the shape", x_train.shape,x_train1.shape, len(x_train0))
    
    for imgs in train_generator:
        count += 1
        print(count)
        if count <2:
            x_train = np.concatenate((x_train, imgs[0]), axis=0)
        else:
            break
    
    #print(x_train.shape)
    #exit()
    
    #This part is for demonstration of large trained model
    #
    #largetrainmodeltest(x_train,autoencoder)
    #exit()
    #
    #
    
    
    
    
    #x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    #x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 
    noise = np.random.rand(x_train.shape[0],304,304,3) 
    #import noise_rate to modify for steps
    noise_rate = 0.10
    #noise = (noise - 0.5) * noise_rate
    shownoisetrain = x_train + noise*noise_rate
    noisetrain = x_train + noise*noise_rate #+ 0.5*noise_rate #add noise and make value non-negative
    #noisetrain = preprocessing.normalize(noisetrain)
    #noise = np.random.rand(x_test.shape[0],28,28,1) 
    #noise_rate = 0.2
    #noise = (noise - 0.5) * noise_rate
    #noisetest = (x_test - 0.5) + noise + 0.5 + 0.5*noise_rate #add noise and make value 
    
    shownoise = np.random.rand(x_train.shape[0],304,304)
    #noiserate2 = noise * 0.2
    #nt2 = x_train + noiserate2 + 0.5*0.2
    for i in range(4):
        for x in range(len(x_train[photonumber+i])):
            for y in range(len(x_train[photonumber+i][0])):
                for z in range(len(x_train[photonumber+i][0][0])):
                    if randomselect(0):
                        noisetrain[photonumber+i][x][y][z] = noise[photonumber+i][x][y][z] + x_train[photonumber+i][x][y][z]
                    else:
                        pass
                        #noisetrain[photonumber+i][x][y][z] = x_train[photonumber+i][x][y][z]
    
    #minmax = [noisetrain[5],noisetrain[5]]
    
    # why design. attack details. Math details.
    # challenges. Justification 1, 2, 3.
    # stepping algorithm. zenme rao guo qu.
    # Attack extend 3 pages.
    encoded_imgs = autoencoder.encoder(np.asarray([x_train[photonumber]])).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    encoded_imgs1 = autoencoder.encoder(np.asarray([noisetrain[photonumber]])).numpy()
    decoded_imgs1 = autoencoder.decoder(encoded_imgs1).numpy()
    startloss = sum(sum(tf.keras.losses.MSE(x_train[photonumber],decoded_imgs1[0]))).numpy()
    #loss = [startloss,startloss]
    
    normalloss = sum(sum(tf.keras.losses.MSE(x_train[photonumber],decoded_imgs[0]))).numpy()
    print(normalloss)
    #exit()
    minloss = startloss
    minimage = noisetrain[photonumber]
    maxloss = startloss
    maximage = noisetrain[photonumber]
    
    minencoded = maxencoded = encoded_imgs1[0]
    mindecoded = maxdecoded = decoded_imgs1[0]
    count = 0
    steps = 1
    for step in range(steps):
        baseimage = maximage
        for i in range(5):
            noise = np.random.rand(304,304,3) 
            tmptrain = np.zeros((304,304,3))
            for x in range(len(x_train[photonumber])):
                for y in range(len(x_train[photonumber][0])):
                    for z in range(len(x_train[photonumber][0][0])):
                        if randomselect(noise_rate/steps):
                            tmptrain[x][y][z] = noise[x][y][z] + baseimage[x][y][z]
                        else:
                            tmptrain[x][y][z] = baseimage[x][y][z]
            count += 1
            print(count, "images generated")
            encoded_imgs1 = autoencoder.encoder(np.asarray([tmptrain])).numpy()
            decoded_imgs1 = autoencoder.decoder(encoded_imgs1).numpy()
            tmploss = sum(sum(tf.keras.losses.MSE(x_train[photonumber],decoded_imgs1[0]))).numpy()
            print("Current loss:", tmploss)
            if tmploss < minloss:
                minloss = tmploss
                minimage = tmptrain
                minencoded = encoded_imgs1[0]
                mindecoded = decoded_imgs1[0]
            elif tmploss > maxloss:
                maxloss = tmploss
                maximage = tmptrain
                maxencoded = encoded_imgs1[0]
                maxdecoded = decoded_imgs1[0]
        
    
    
    
    train = [x_train[photonumber],x_train[photonumber]]
    encoded = [encoded_imgs[0],encoded_imgs[0]]
    decoded = [decoded_imgs[0],decoded_imgs[0]]
    
    ntrain = [x_train[photonumber],noisetrain[photonumber]]
    encoded_imgs1 = [encoded_imgs[0],maxencoded]
    decoded_imgs1 = [decoded_imgs[0],maxdecoded]
    
    print(normalloss,minloss,maxloss)
    
    plt.figure(figsize=(20, 4))
    n = 2
    
    
    
    for i in range(n):
        # display original
        # ax = plt.subplot(7, n, i + 1)
        # plt.imshow(train[i])
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        # # display original encode
        # ax = plt.subplot(7, n, i + 1 + n)
        # plt.imshow(encoded[i].reshape(16, 16),cmap='Greys')
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        # #display original reconstruction
        # ax = plt.subplot(7, n, i + 1 + 2*n)
        # plt.imshow(decoded[i])
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        
        
        # #display noise
        
        # ax = plt.subplot(7, n, i + 1 + 3*n)
        # plt.imshow(shownoise[i],cmap='Greys')
        
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        
        #display noisetest
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(ntrain[i],cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #This is only for bigger model to select encode data.
        encode = np.zeros((256,))
        count = 0
        for x in range(len(encoded_imgs1[i])):
            for y in range(len(encoded_imgs1[i][x])):
                for z in range(len(encoded_imgs1[i][x][y])):
                    encode[count] = encoded_imgs1[i][x][y][z]
                    count += 1
                    if count == 256: break
                if count == 256: break
            if count == 256: break
        
        # display noise encode
        ax = plt.subplot(3, n, i + 1 + n)
        #plt.imshow(encoded_imgs1[i].reshape(16, 16),cmap='Greys')
        plt.imshow(encode.reshape(16, 16),cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        # display noise reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_imgs1[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        
    plt.show()
    
    
    #showsample(x_train[5])
    #showsample(encoded_imgs[5].reshape(16,16))
    #showsample(decoded_imgs[5])
    

if __name__ == "__main__":
    #main()
    loadandtest()