import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d,lib.cnmem=0"%(random.randint(0,3))
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D, Deconvolution2D, Convolution3D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
#from IPython import display
from keras.utils import np_utils
from tqdm import tqdm
import keras.backend as K
import pickle
import sys
import math
import tensorflow as tf

K.set_image_dim_ordering('th')
"""
class CustomizedWeightRegularizer(WeightRegularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.variable(l1)
        self.l2 = K.variable(l2)
        self.uses_learning_phase = True
        self.p = None


def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
"""
#define two optimizer with different learning rates
#opt = Adam(lr=1e-1)
opt = Adagrad(lr=0.005, epsilon=1e-08)
#dopt = Adam(lr=1e-4)
dopt = Adagrad(lr=0.0005, epsilon=1e-08)
dopt2 = Adagrad(lr=0.0005, epsilon=1e-08)
opt2 = Adagrad(lr=0.005, epsilon=1e-08)
#opt2 = Adam(lr=1e-3)

#dopt2 = Adam(lr=1e-4)

#dis_temp_opt = Adam(lr=1e-4)
dis_temp_opt = Adagrad(lr=0.001, epsilon=1e-08)
opt_enc_frozen = Adagrad(lr=0.008, epsilon=1e-08)
#opt =RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
#dopt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#dis_temp_opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
"""#define two optimizer with different learning rates
#opt = Adam(lr=1e-1)
opt = Adagrad(lr=0.005, epsilon=1e-08)
#dopt = Adam(lr=1e-4)
dopt = Adagrad(lr=0.0001, epsilon=1e-08)
#dis_temp_opt = Adam(lr=1e-4)
dis_temp_opt = Adagrad(lr=0.001, epsilon=1e-08)
opt_enc_frozen = Adagrad(lr=0.0001, epsilon=1e-08)
#opt =RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
#dopt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#dis_temp_opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
"""
############################ Load data and preprocessing #######################

# load data

##########  N = 40   #############################
print("start loading data")
"""
file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T0.9.pickle', 'rb')
x_train9 = pickle.load(file)
file.close() 
file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.0.pickle', 'rb')
x_train10 = pickle.load(file)
file.close() 
"""
file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.02.pickle', 'rb')
x_train102 = pickle.load(file)
file.close() 
file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.04.pickle', 'rb')
x_train104 = pickle.load(file)
file.close() 
"""
file = open('/fs1/users/mstieffenhofer/data_heisenberg/N40/data_N40_T1.06.pickle', 'rb')
x_train106 = pickle.load(file)
file.close() 
"""
print("loading data successfull")



X_train = x_train104


x_train = np.concatenate((x_train102, x_train104), axis = 0)
#x_train = np.concatenate((x_train, x_train102), axis = 0)
#x_train = np.concatenate((x_train, x_train104), axis = 0)
#x_train = np.concatenate((x_train, x_train106), axis = 0)



(n_samples_b,dim, num_row_b, num_col_b) = X_train.shape

#

print("Shape of X_train:")
print(X_train.shape)





labels = ["T = 0.1", "T = 0.2", "T = 0.3","T = 0.4","T = 0.5","T = 0.6","T = 0.7","T = 0.8","T = 0.9","T = 1.0","T = 1.04","T = 1.1","T = 1.2","T = 1.3","T = 1.4","T = 1.5","T = 1.6","T = 1.7","T = 1.8","T = 1.9","T = 2.0"]





##############data preprocessing #############
"""
def make_strides(samples):
    n = samples.shape(2)
    samples = samples.repeat(2, axis=2).repeat(2, axis=3)
	mult_mat = np.array([n*[0,1],n*[0,0]]*n)
"""	

################################################################################################################################################################################################################################
##################################################################################################  DECODER  (RENORMALIZATION)   #######################################################################################################
################################################################################################################################################################################################################################

	

def reshaper(layers):
    layers = tf.reshape(layers, (32,3,40,40))
    return layers

def normalize_features(layers):
    norm = tf.square(layers)
    norm = tf.reduce_sum(norm, 1, keep_dims=True)
    norm = tf.sqrt(norm) 
    #norm = tf.concat(1, [norm,norm, norm])
    layers= tf.div(layers, norm)

    return layers

def normalize(layers):
    norm = tf.square(layers)
    norm = tf.reduce_sum(norm, 1, keep_dims=True)
    norm = tf.sqrt(norm) 
    #norm = tf.concat(1, [norm,norm, norm])
    norm = tf.tile(norm, [1,100,1,1])
    layers= tf.div(layers, norm)

    return layers
    
    
def normalize(layers):
    norm = tf.square(layers)
    norm = tf.reduce_sum(norm, 1, keep_dims=True)
    norm = tf.sqrt(norm) 
    norm = tf.concat(1, [norm,norm, norm])
    layers= tf.div(layers, norm)

    return layers



def noise_out_shape(input_shape):
    shape = list(input_shape)
    #shape[1] = 15
    return tuple([shape[0],2*shape[1],shape[2],shape[3]])
	
def noise(layers):
    print(K.shape(layers))
    noise = K.random_normal(shape=K.shape(layers),mean=0.,std=0.2)
    #noise = normalize(noise)
    layers = tf.concat(1, [layers, noise])

    return layers	
    


    
 
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val	
        
        
BS = 8



def takepart(layers):	
    part = tf.random_crop(layers, [BS,3,40,40])
    return part    
    
def takepart2(layers):	
    part = tf.random_crop(layers, [BS,3,80,80])
    return part 

def takepart_set_out_shape(input_shape):
    shape = list(input_shape)
    return tuple([shape[0],16,3,40,40])   
    
def takepart_set(layers):	
    part = tf.random_crop(layers, [BS,3,160,160])
    part = tf.reshape(part,(BS,1,3,160,160))
    for i in range(0,4):
        for j in range(0,4):
            p = part[:,:,:,i*40:(i+1)*40,j*40:(j+1)*40] 
            if (i+j) != 0:
                r = tf.concat(1, [r, p])	
            else:
                r = p
    return r 
    
#########################################----------------------ENCODER---------------------################################################
encoder_inp = Input(shape=(3,None,None))
H = AveragePooling2D(pool_size=(2, 2))(encoder_inp)
encoder_out = Lambda(normalize)(H)

encoder = Model(encoder_inp, encoder_out)
encoder.compile(loss='categorical_crossentropy', optimizer=opt)
encoder.summary()
##########################################################################################################################################

        
        



#########################################----------------------DECODER---------------------################################################
n_ch = 100

decoder_inp = Input(shape=(3,40,40))
H = Convolution2D(n_ch,4,4, border_mode='same', init='glorot_uniform', name='decoder11')(decoder_inp)
H = Activation('tanh')(H)
H = Deconvolution2D(n_ch, 4, 4, output_shape=(BS,n_ch, 80, 80), subsample=(2, 2), border_mode='same', name='decoder12')(H)
H = Activation('tanh')(H)
H = Convolution2D(3,4,4, border_mode='same', init='glorot_uniform', name='decoder13')(H)
H = Activation('tanh')(H)

decoder_out = Lambda(normalize)(H)

decoder = Model(decoder_inp, decoder_out)
decoder.compile(loss='categorical_crossentropy', optimizer=opt)
decoder.summary()
###########################################################################################################################################

#########################################----------------------DECODER 2---------------------################################################
n_ch = 100

decoder_inp2 = Input(shape=(3,80,80))
H = Convolution2D(n_ch,4,4, border_mode='same', init='glorot_uniform', name='decoder21')(decoder_inp2)
H = Activation('tanh')(H)
H = Deconvolution2D(n_ch, 4, 4, output_shape=(BS,n_ch, 160, 160), subsample=(2, 2), border_mode='same', name='decoder22')(H)
H = Activation('tanh')(H)
H = Convolution2D(3,4,4, border_mode='same', init='glorot_uniform', name='decoder23')(H)
H = Activation('tanh')(H)

decoder_out2 = Lambda(normalize)(H)

decoder2 = Model(decoder_inp2, decoder_out2)
decoder2.compile(loss='categorical_crossentropy', optimizer=opt)
decoder2.summary()
###########################################################################################################################################

#########################################----------------------DECODER 3---------------------################################################
n_ch = 100

decoder_inp3 = Input(shape=(3,160,160))
H = Convolution2D(n_ch,4,4, border_mode='same', init='glorot_uniform', name='decoder31')(decoder_inp3)
H = Activation('tanh')(H)
H = Deconvolution2D(n_ch, 4, 4, output_shape=(BS,n_ch, 320, 320), subsample=(2, 2), border_mode='same', name='decoder32')(H)
H = Activation('tanh')(H)
H = Convolution2D(3,4,4, border_mode='same', init='glorot_uniform', name='decoder33')(H)
H = Activation('tanh')(H)

decoder_out3 = Lambda(normalize)(H)

decoder3 = Model(decoder_inp3, decoder_out3)
decoder3.compile(loss='categorical_crossentropy', optimizer=opt)
decoder3.summary()
###########################################################################################################################################

#########################################----------------------DECODER 4--------------------################################################
n_ch = 100

decoder_inp4 = Input(shape=(3,320,320))
H = Convolution2D(n_ch,4,4, border_mode='same', init='glorot_uniform', name='decoder41')(decoder_inp4)
H = Activation('tanh')(H)
H = Deconvolution2D(n_ch, 4, 4, output_shape=(BS,n_ch, 640, 640), subsample=(2, 2), border_mode='same', name='decoder42')(H)
H = Activation('tanh')(H)
H = Convolution2D(3,4,4, border_mode='same', init='glorot_uniform', name='decoder43')(H)
H = Activation('tanh')(H)

decoder_out4 = Lambda(normalize)(H)

decoder4 = Model(decoder_inp4, decoder_out4)
decoder4.compile(loss='categorical_crossentropy', optimizer=opt)
decoder4.summary()
###########################################################################################################################################



#########################################----------------------DISCRIMINATOR ---------------------################################################

dr_dis_en = 0.3
#define input tensor
dis_decoder_inp = Input(shape=(3,40,40))
H = Convolution2D(512,3,3,  subsample=(2,2), border_mode ='same',name='dis_de1_1')(dis_decoder_inp)
H = LeakyReLU(0.2)(H)
H = Dropout(dr_dis_en)(H)
H = Convolution2D(256,3,3, subsample=(2,2), border_mode ='same',name='dis_de1_2')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dr_dis_en)(H)
H = Flatten()(H)
H = Dense(256,name='dis_de1_3')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dr_dis_en)(H)
dis_decoder_out = Dense(2, activation='softmax',name='dis_de1_4')(H)	
	
dis_decoder = Model(dis_decoder_inp, dis_decoder_out)
dis_decoder.compile(loss='categorical_crossentropy', optimizer=dopt, metrics=['accuracy'])
dis_decoder.summary()	





################################### take random part #####################
inp = Input(shape=(3,None,None))
H = Lambda(takepart)(inp)

taker = Model(inp, H)
taker.compile(loss='categorical_crossentropy', optimizer=opt_enc_frozen, metrics=['accuracy'])
#print summary representation of the model
taker.summary()



#########################################----------------------GAN - 1 ---------------------################################################
make_trainable(dis_decoder,False)
make_trainable(decoder,True)

gan_comb_inp = Input(shape=(3,40,40))

B = decoder(gan_comb_inp)
H = Lambda(takepart)(B)
dis_out1 = dis_decoder(H)
D = encoder(B)


gan_comb = Model([gan_comb_inp], [dis_out1, D])
gan_comb.compile(loss=['categorical_crossentropy', 'mse'], optimizer=opt_enc_frozen, metrics=['accuracy'],loss_weights=[1.0,10.0])
gan_comb.summary()
###########################################################################################################################################

#########################################----------------------GAN - 2 ---------------------################################################
make_trainable(dis_decoder,False)
make_trainable(dis_decoder2,False)
make_trainable(decoder,False)
make_trainable(decoder2,True)

gan_comb_inp2 = Input(shape=(3,40,40))

B = decoder(gan_comb_inp2)
B = decoder2(B)

H = Lambda(takepart)(B)
dis_out21 = dis_decoder(H)

E = encoder(B)

gan_comb2 = Model([gan_comb_inp2], [dis_out21, E])
gan_comb2.compile(loss=['categorical_crossentropy', 'mse'], optimizer=opt_enc_frozen, metrics=['accuracy'],loss_weights=[1.0,10.0])
gan_comb2.summary()


###########################################################################################################################################
#########################################----------------------GAN - 3 ---------------------################################################
make_trainable(dis_decoder,False)
make_trainable(dis_decoder2,False)
make_trainable(decoder,False)
make_trainable(decoder2,False)
make_trainable(decoder3,True)

gan_comb_inp3 = Input(shape=(3,40,40))

B = decoder(gan_comb_inp3)
B = decoder2(B)
B = decoder3(B)

H = Lambda(takepart_set, output_shape=takepart_set_out_shape)(B)
H = TimeDistributed(dis_decoder,input_shape=(16,3,40,40))(H)

E = encoder(B)


gan_comb3 = Model([gan_comb_inp3], [H, E])
gan_comb3.compile(loss=['categorical_crossentropy','mse'], optimizer=opt_enc_frozen, metrics=['accuracy'],loss_weights=[1.0,8.0])
gan_comb3.summary()





#########################################----------------------GAN - 4 ---------------------################################################
make_trainable(dis_decoder,False)
make_trainable(dis_decoder2,False)
make_trainable(decoder,False)
make_trainable(decoder2,False)
make_trainable(decoder3,False)
make_trainable(decoder4,True)

gan_comb_inp4 = Input(shape=(3,40,40))

B = decoder(gan_comb_inp4)
B = decoder2(B)
B = decoder3(B)
B = decoder4(B)

#H = Lambda(takepart)(B)
H = Lambda(takepart_set, output_shape=takepart_set_out_shape)(B)
H = TimeDistributed(dis_decoder,input_shape=(16,3,40,40))(H)


E = encoder(B)


gan_comb4 = Model([gan_comb_inp4], [H, E])
gan_comb4.compile(loss=['categorical_crossentropy','mse'], optimizer=opt_enc_frozen, metrics=['accuracy'],loss_weights=[1.0,7.0])
gan_comb4.summary()





losses = {"g":[],"g2":[],"g3":[],"g4":[], "d1": [], "d2": [], "d3": [], "d4": []}  	
def training(nb_epoch=5000, BATCH_SIZE=BS):
    #display the progess of the learning process    
    for e in tqdm(range(nb_epoch)):  

        idx = np.random.choice(np.arange(X_train.shape[0]), BATCH_SIZE, replace=False)
        X_sample = X_train[idx]
                
        idx = np.random.choice(np.arange(X_train.shape[0]),BATCH_SIZE, replace=False)
        train_img_b = X_train[idx]    
        decode_img = decoder.predict(train_img_b)
        dec_img1 = taker.predict(decode_img) 

        X_sample = np.concatenate((X_sample, dec_img1),axis = 0)

        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(dis_decoder,True)
        
        d1_loss  = dis_decoder.train_on_batch(X_sample,y)
        losses["d1"].append(d1_loss)          
#######################################################################################

        idx = np.random.choice(np.arange(X_train.shape[0]), BATCH_SIZE, replace=False)
        X_sample = X_train[idx]
                
        idx = np.random.choice(np.arange(X_train.shape[0]),BATCH_SIZE, replace=False)
        train_img_b = X_train[idx]    
        decode_img = decoder2.predict(decoder.predict(train_img_b))
        dec_img1 = taker.predict(decode_img) 
        
        X_sample = np.concatenate((X_sample, dec_img1),axis = 0)
        
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(dis_decoder,True)
        
        d2_loss  = dis_decoder.train_on_batch(X_sample,y)
        losses["d2"].append(d2_loss)              
      
#######################################################################################

        idx = np.random.choice(np.arange(X_train.shape[0]), BATCH_SIZE, replace=False)
        X_sample = X_train[idx]
                
        idx = np.random.choice(np.arange(X_train.shape[0]),BATCH_SIZE, replace=False)
        train_img_b = X_train[idx]    
        decode_img = decoder3.predict(decoder2.predict(decoder.predict(train_img_b)))
        dec_img1 = taker.predict(decode_img) 
        
        X_sample = np.concatenate((X_sample, dec_img1),axis = 0)
        
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(dis_decoder,True)
        
        d3_loss  = dis_decoder.train_on_batch(X_sample,y)
        losses["d3"].append(d3_loss)       

#######################################################################################

        idx = np.random.choice(np.arange(X_train.shape[0]), BATCH_SIZE, replace=False)
        X_sample = X_train[idx]
                
        idx = np.random.choice(np.arange(X_train.shape[0]),BATCH_SIZE, replace=False)
        train_img_b = X_train[idx]    
        decode_img = decoder4.predict(decoder3.predict(decoder2.predict(decoder.predict(train_img_b))))
        dec_img1 = taker.predict(decode_img) 

        X_sample = np.concatenate((X_sample, dec_img1),axis = 0)

        
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(dis_decoder,True)
        
        d4_loss  = dis_decoder.train_on_batch(X_sample,y)
        losses["d4"].append(d4_loss)           
        
 ######################################################################################
 

        idx = np.random.choice(np.arange(X_train.shape[0]), BATCH_SIZE, replace=False)
        X_sample = X_train[idx]
        make_trainable(dis_decoder,False)    
        y = np.zeros([BATCH_SIZE,2])
        y[:,1] = 1

        g_loss = gan_comb.train_on_batch(X_sample,[y, X_sample])
        losses["g"].append(g_loss)
        
        dec_img = decoder.predict(X_sample)
        g_loss2 = gan_comb2.train_on_batch(X_sample,[y, dec_img])
        losses["g2"].append(g_loss2)

        y = np.zeros([BATCH_SIZE,16,2])
        y[:,:,1] = 1
        dec_img2 = decoder2.predict(dec_img)
        g_loss3 = gan_comb3.train_on_batch(X_sample,[y, dec_img2])
        losses["g3"].append(g_loss3)
        

        dec_img3 = decoder3.predict(dec_img2)        
        g_loss4 = gan_comb4.train_on_batch(X_sample,[y, dec_img3])
        losses["g4"].append(g_loss4)
	
        print("Gen-Loss: "+str(g_loss)+" --- "+"Gen-Loss2: "+str(g_loss2)+" --- "+"Gen-Loss3: "+str(g_loss3)+" --- "+"Gen-Loss4: "+str(g_loss4)+" --- "+"Dis-Dec1: "+str(d1_loss)+" --- "+"Dis-Dec: "+str(d2_loss))


for n in range(0,20):
    print("############################  "+str(n)+"  ##########################################")
    training(nb_epoch=2500, BATCH_SIZE=BS)	
    with open('loss.pickle', 'wb') as f:
        pickle.dump(losses, f)
    
    decoder.save('decoder.h5', overwrite=True)
    decoder2.save('decoder2.h5', overwrite=True)
    decoder3.save('decoder3.h5', overwrite=True)
    decoder4.save('decoder4.h5', overwrite=True)
    dis_decoder.save('dis_decoder.h5', overwrite=True)
    decoder.save_weights('decoder_weights.h5', overwrite=True)		
    decoder2.save_weights('decoder2_weights.h5', overwrite=True)	
    decoder3.save_weights('decoder3_weights.h5', overwrite=True)
    decoder4.save_weights('decoder4_weights.h5', overwrite=True)

    dis_decoder.save_weights('dis_decoder_weights.h5', overwrite=True)
    
