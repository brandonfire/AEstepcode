#Autoencoder for ImageNet
import itertools
import os
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import tensorflow_datasets as tfds
#import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn import preprocessing 
from tensorflow.keras import regularizers
from random import random

class ImageNetEncoder(Model):
  def __init__(self):
    super(ImageNetEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
    
      layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2), padding='same'),
      layers.Flatten(),
      layers.Dense(256, activation="relu")
      
      ])
      #layers.Dense(1, activation="relu")])
    
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
    data_dir = "G:\\Chengbinhu\\Desktop\\ScienceAdvancedSubmission\\Autoencoder\\FlowerData\\flower_photos"
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
    autoencoder = ImageNetEncoder()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    history = autoencoder.fit(train_generator,epochs=10,batch_size=32,validation_data=valid_generator,shuffle=True)
    #autoencoder.save("ClassicalEncodermodel")
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    showsample(x_train[5])
    
    showsample(encoded_imgs[5].reshape(16,16))
    showsample(decoded_imgs[5])
    loss = sum(sum(tf.keras.losses.MSE(x_train[5],decoded_imgs[5]))).numpy()
    
    print('loss after all data training', loss)

def trainclassification():
    pixels = 304
    IMAGE_SIZE = (pixels, pixels)

    BATCH_SIZE = 32 
    data_dir = "G:\\Chengbinhu\\Desktop\\ScienceAdvancedSubmission\\Autoencoder\\FlowerData\\flower_photos"
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")
                       
                       
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=True, **dataflow_kwargs)
    autoencoder = tf.keras.models.load_model('./ClassicalEncodermodel')
    
    x_train = False
    y_train = False
    x_test = False
    y_test = False
    
    
    count = 0
    
    for imgs in train_generator:
        count += 1
        print(count)
        if count == 1:
            x_train = autoencoder.encoder(imgs[0])
            y_train = imgs[1]
        elif count < 90:
            x_train = np.concatenate((x_train, autoencoder.encoder(imgs[0])), axis=0)
            y_train = np.concatenate((y_train, imgs[1]), axis=0)
        # elif count == 4:
            # x_test = train_generator.next()[0]
        # elif count <= 5:
            # x_test = np.concatenate((x_test, imgs[0]), axis=0)
        else:
            break
    
    count = 0
    for imgs in valid_generator:
        count += 1
        print(count)
        if count == 1:
            x_test = autoencoder.encoder(imgs[0])
            y_test = imgs[1]
        elif count < 20:
            y_test = np.concatenate((y_test, imgs[1]), axis=0)
            x_test = np.concatenate((x_test, autoencoder.encoder(imgs[0])), axis=0)
        # elif count == 4:
            # x_test = train_generator.next()[0]
        # elif count <= 5:
            # x_test = np.concatenate((x_test, imgs[0]), axis=0)
        else:
            break
    
    
    
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input = autoencoder.encoded(),input_shape=(256,)),#224,224,3 image
    tf.keras.layers.Dropout(rate=0.2),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(5,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary()
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
      metrics=['accuracy'])
    hist = model.fit(
    x_train,y_train,
    epochs=100, batch_size=32,
    validation_data=(x_test,y_test)).history
    
    model.save("FirstEnThenClassModel")
    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,1])
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    plt.show()
    
def testaccuracywithinception():
    import tensorflow_hub as hub
    module_selection = ("mobilenet_v2", 224) 
    handle_base, pixels = module_selection
    MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
    IMAGE_SIZE = (pixels, pixels)
    #print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

    BATCH_SIZE = 32 

    data_dir = ".\\FlowerData\\flower_photos"

    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    do_data_augmentation = False 
    if do_data_augmentation:
      train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rotation_range=40,
          horizontal_flip=True,
          width_shift_range=0.2, height_shift_range=0.2,
          shear_range=0.2, zoom_range=0.2,
          **datagen_kwargs)
    else:
      train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=True, **dataflow_kwargs)
    #print(type(train_generator),train_generator)




    do_fine_tuning = False #At first we want to fix the feature extractor. We can impove feature extractor later.



    #Transfer Learning
    print("Building model with", MODULE_HANDLE)
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),#224,224,3 image
        hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(2,
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary()
    exit()
    model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
      metrics=['accuracy'])
    log_dir="./logs/fit"
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    #print("This is model",model,type(model))
    #exit()
    hist = model.fit(
        train_generator,
        epochs=50, steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps).history
      
    # saved_model_path = "./Lunginception_v3"
    # tf.saved_model.save(model, saved_model_path)
    model.save("ImageNetClassModel")
    
    plt.figure()
    plt.ylabel("Loss (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,2])
    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])

    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,1])
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    plt.show()
def generateENDE(Image, AEmodel, threshold):
    EN, IN = [],[]
    encoded_img = AEmodel.encoder(np.asarray([Image])).numpy()
    decoded_img = AEmodel.decoder(encoded_img).numpy()
    baselose = sum(sum(tf.keras.losses.MSE(Image,decoded_img[0]))).numpy()
    print(baselose,"This is baseloss")

    for x in range(len(Image)):
        for y in range(len(Image[0])):
            for z in range(len(Image[0][0])):
                Image[x][y][z] += 1
                encoded_img = AEmodel.encoder(np.asarray([Image])).numpy()
                decoded_img = AEmodel.decoder(encoded_img).numpy()
                tmploss = sum(sum(tf.keras.losses.MSE(Image,decoded_img[0]))).numpy()
                if tmploss - baselose > threshold:
                    EN.append([x,y,z])
    for x in range(len(Image)):
        for y in range(len(Image[0])):
            for z in range(len(Image[0][0])):
                Image[x][y][z] -= 1
                encoded_img = AEmodel.encoder(np.asarray([Image])).numpy()
                decoded_img = AEmodel.decoder(encoded_img).numpy()
                tmploss = sum(sum(tf.keras.losses.MSE(Image,decoded_img[0]))).numpy()
                if tmploss - baselose > threshold:
                    DE.append([x,y,z])
    return EN,DE

def selectfrompool(EN,DE,rate):
    chooseEN = []
    chooseDE = []
    for point in EN:
        if randomselect(rate):
            chooseEN.append(point)
    for point in DE:
        if randomselect(rate):
            chooseDE.append(point)
    return chooseEN, chooseDE



def testwithnoiseattack(Autoencoder_model,Class_model,data_dir,steps,randomselectedsamples ,noise_rate, targetclass):
    autoencoder = tf.keras.models.load_model(Autoencoder_model) #load autoencoder
    classfymodel = tf.keras.models.load_model(Class_model) #load downstream classification model
    pixels = 304
    IMAGE_SIZE = (pixels, pixels)
    BATCH_SIZE = 32 
    #data_dir = "G:\\Chengbinhu\\Desktop\\ScienceAdvancedSubmission\\Autoencoder\\FlowerData\\flower_photos"

    datagen_kwargs = dict(rescale=1./255, validation_split=.20)#data normalize argument, 20% validation 
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")#Bilinear upsample for resize the image to 304 X 304
    
    #the following code generate all the training and validation data as datagenerator.
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=False, **dataflow_kwargs)
    


    for imgs in train_generator:
        print(type(imgs))
        print(imgs[0].shape,imgs[1].shape)
        x_train = imgs[0]
        y_train = imgs[1]
        break



    #exit() #This is for debug.
    #In our experiment we define x_train, y_train first, because we want to test smaller sample
    x_train = False
    y_train = False
    x_test = False
    y_test = False
    
    count = 0
    
    for imgs in train_generator:
        count += 1
        print(count)
        if count == 1:
            x_train = imgs[0]
            y_train = imgs[1]
        #elif count < 2:
        #    x_train = np.concatenate((x_train, imgs[0]), axis=0)
        #    y_train = np.concatenate((y_train, imgs[1]), axis=0)
        # elif count == 4:
            # x_test = train_generator.next()[0]
        # elif count <= 5:
            # x_test = np.concatenate((x_test, imgs[0]), axis=0)
        else:
            break
    
    count = 0
    for imgs in valid_generator:
        count += 1
        print(count)
        if count == 1:
            x_test = imgs[0]
            y_test = imgs[1]
        #elif count < 2:
        #    y_test = np.concatenate((y_test, imgs[1]), axis=0)
        #    x_test = np.concatenate((x_test, imgs[0]), axis=0)
        # elif count == 4:
            # x_test = train_generator.next()[0]
        # elif count <= 5:
            # x_test = np.concatenate((x_test, imgs[0]), axis=0)
        else:
            break
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    success_attack = 0
    total = 0
    for imgs in train_generator:
        x_train = imgs[0] #depends on the data size, this loop takes hours to days to be done.
        y_train = imgs[1]
        for photonumber in range(len(imgs[0])):
            noise = np.random.rand(x_train.shape[0],304,304,3) 
            #import noise_rate to modify for steps
            #noise = (noise - 0.5) * noise_rate
            noisetrain = x_train + noise*noise_rate
            shownoise = np.random.rand(x_train.shape[0],304,304)
            
            #for x in range(len(x_train[photonumber])):
            #    for y in range(len(x_train[photonumber][0])):
            #        for z in range(len(x_train[photonumber][0][0])):


            #                if randomselect(noise_rate):
            #                    noisetrain[photonumber+i][x][y][z] = noise[photonumber+i][x][y][z] + x_train[photonumber+i][x][y][z]
            #                else:
            #                    noisetrain[photonumber+i][x][y][z] = x_train[photonumber+i][x][y][z]
            encoded_imgs = autoencoder.encoder(np.asarray([x_train[photonumber]])).numpy()
            decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
            encoded_imgs1 = autoencoder.encoder(np.asarray([noisetrain[photonumber]])).numpy()
            decoded_imgs1 = autoencoder.decoder(encoded_imgs1).numpy()
            
            classpredict = classfymodel(encoded_imgs)
            
            startloss = sum(sum(tf.keras.losses.MSE(x_train[photonumber],decoded_imgs1[0]))).numpy()
            #loss = [startloss,startloss]
            
            normalloss = sum(sum(tf.keras.losses.MSE(x_train[photonumber],decoded_imgs[0]))).numpy()
            
            normalcrossloss = tf.keras.losses.binary_crossentropy(classpredict,y_train[photonumber]).numpy()
            #showsample(x_train[4])
            
            #showsample(encoded_imgs[0].reshape(16,16))
            #showsample(decoded_imgs[0])
            
            print(normalloss,normalcrossloss,classpredict)
            
            minloss = startloss
            minimage = noisetrain[photonumber]
            maxloss = startloss
            maximage = noisetrain[photonumber]
            
            minclassprobability = classpredict[0][targetclass]
            maxclassprobability = classpredict[0][targetclass]

            minencoded = maxencoded = encoded_imgs1[0]
            mindecoded = maxdecoded = decoded_imgs1[0]
            count = 0
            EN,DE = generateENDE(x_train[photonumber], autoencoder, 0.1)

            
            for step in range(steps):
                baseimage = maximage
                for i in range(randomselectedsamples):
                    noise = np.random.rand(304,304,3) 
                    tmptrain = list(x_train[photonumber])#np.zeros((304,304,3))
                    chooseEN, chooseDE = selectfrompool(EN,DE, noise_rate/steps)
                    for e in chooseEN:
                        tmptrain[e[0]][e[1]][e[2]] += 1
                    for d in chooseDE:
                        tmptrain[d[0]][d[1]][d[2]] -= 1

                    #for x in range(len(x_train[photonumber])):
                    #    for y in range(len(x_train[photonumber][0])):
                    #        for z in range(len(x_train[photonumber][0][0])):
                    #            for e in chooseEN:


                                #if randomselect(noise_rate/steps):
                                #    tmptrain[x][y][z] = noise[x][y][z] + baseimage[x][y][z]
                                #else:
                                #    tmptrain[x][y][z] = baseimage[x][y][z]
                    count += 1
                    print(count, "images generated")
                    encoded_imgs1 = autoencoder.encoder(np.asarray([tmptrain])).numpy()
                    decoded_imgs1 = autoencoder.decoder(encoded_imgs1).numpy()
                    tmploss = sum(sum(tf.keras.losses.MSE(x_train[photonumber],decoded_imgs1[0]))).numpy()
                    classpredict = classfymodel(encoded_imgs)
                    tmpclassprobability = classpredict[0][targetclass]
                    print("Current loss:", tmploss)
                    print("Current classification probability", classpredict)
                    
                    #if tmploss < minloss:
                    #    minloss = tmploss
                    #   minimage = tmptrain
                    #    minencoded = encoded_imgs1[0]
                    #    mindecoded = decoded_imgs1[0]
                    if tmpclassprobability > maxclassprobability:
                        maxclassprobability = tmpclassprobability
                        maximage = tmptrain
                        maxencoded = encoded_imgs1[0]
                        maxclasssample = encoded_imgs1
                        maxdecoded = decoded_imgs1[0]
                
            
            
            
            train = [x_train[photonumber],x_train[photonumber]]
            encoded = [encoded_imgs[0],encoded_imgs[0]]
            decoded = [decoded_imgs[0],decoded_imgs[0]]
            
            ntrain = [minimage,maximage]
            encoded_imgs1 = [minencoded,maxencoded]
            decoded_imgs1 = [mindecoded,maxdecoded]
            
            #print("maxencoded",maxencoded,maxencoded.shape)
            maxclass = classfymodel(maxclasssample)
            print("This is maxclass", maxclass, type(maxclasssample))
            maxcrossloss = tf.keras.losses.binary_crossentropy(maxclass,y_train[photonumber]).numpy()
            
            print(minloss,maxloss,maxcrossloss,maxclass)
        if max(classpredict[0]) == classpredict[0][targetclass]:
            success_attack += 1
        total += 1
    attack_succss_rate = success_attack/total
    print("Attack succss rate is", attack_succss_rate)

    plt.figure(figsize=(20, 4))
    n = 2
    
    
    
    for i in range(n):
        # display original
        ax = plt.subplot(7, n, i + 1)
        plt.imshow(train[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display original encode
        ax = plt.subplot(7, n, i + 1 + n)
        plt.imshow(encoded[i].reshape(16, 16),cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #display original reconstruction
        ax = plt.subplot(7, n, i + 1 + 2*n)
        plt.imshow(decoded[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        
        #display noise
        
        ax = plt.subplot(7, n, i + 1 + 3*n)
        plt.imshow(shownoise[i],cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #display noisetest
        ax = plt.subplot(7, n, i + 1 + 4*n)
        plt.imshow(ntrain[i],cmap='Greys')
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        #This is only for bigger model to select encode data.
        # encode = np.zeros((256,))
        # count = 0
        # for x in range(len(encoded_imgs1[i])):
            # for y in range(len(encoded_imgs1[i][x])):
                # for z in range(len(encoded_imgs1[i][x][y])):
                    # encode[count] = encoded_imgs1[i][x][y][z]
                    # count += 1
                    # if count == 256: break
                # if count == 256: break
            # if count == 256: break
        
        # display noise encode
        ax = plt.subplot(7, n, i + 1 + 5*n)
        plt.imshow(encoded_imgs1[i].reshape(16, 16),cmap='Greys')
        #plt.imshow(encode.reshape(16, 16),cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        # display noise reconstruction
        ax = plt.subplot(7, n, i + 1 + 6*n)
        plt.imshow(decoded_imgs1[i])
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        
        
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    #main()
    #trainclassification()
    #testaccuracywithinception()
    Autoencoder_model,Class_model = "./ClassicalEncodermodel","./FirstEnThenClassModel"
    data_dir = '.\\FlowerData\\flower_photos'
    steps = 500
    randomselectedsamples = 100
    noise_rate = 0.08
    targetclass = 2
    testwithnoiseattack(Autoencoder_model,Class_model,data_dir,steps,randomselectedsamples, noise_rate , targetclass)#Attack test.
