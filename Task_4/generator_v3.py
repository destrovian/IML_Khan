# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import keras.losses
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image

#from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.xception import preprocess_input

from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import img_to_array
from keras import layers, Model
from keras.regularizers import l2
import numpy as np
import pandas as pd
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential, Input, load_model
from sklearn.utils import class_weight
import datetime
import zipfile


# %%
train_triplets = pd.read_csv("train_triplets.txt", sep=' ', header=None, dtype=str)
test_triplets = pd.read_csv("test_triplets.txt", sep=' ', header=None, dtype=str)

with zipfile.ZipFile("food.zip","r") as zip_ref:
    zip_ref.extractall("food_porn")


# %%
def append_ext(index):
    return index + '.jpg'


# %%
ONE_SET_SIZE = 59515
ABC = ['A', 'B', 'C']

train_triplets_jpg = train_triplets.apply(append_ext)
#data_y1 = pd.DataFrame(np.ones((ONE_SET_SIZE)), dtype='str', columns=['label'])
#train_triplets_jpg = pd.concat([train_triplets_jpg, data_y1], axis=1)
train_triplets_jpg = train_triplets_jpg.rename(columns={0:'A', 1:'B', 2:'C'}, inplace=False)

test_triplets_jpg = test_triplets.apply(append_ext)
test_triplets_jpg = test_triplets_jpg.rename(columns={0:'A', 1:'B', 2:'C'}, inplace=False)


# %%
get_ipython().run_cell_magic('time', '', "\n\n# There are 4 different permutations of ABC that we can know the answer of. (CAB and CBA we don't know)\nONE_SET_SIZE = 59515\n\n#create the base dataset with the 2 sets of labels\ndata_y1 = pd.DataFrame(np.ones((ONE_SET_SIZE * 2,1)), dtype='str', columns=['label'])\ndata_y0 = pd.DataFrame(np.zeros((ONE_SET_SIZE * 2,1)), dtype='str', columns=['label'])\n\n#create the inverse dataset and a set where B is connected to A (and inv)\n\n# One needs 3 different assignments since the arrays would be linked otherwise.\nACB0 = train_triplets_jpg[['A', 'C', 'B']].copy()\nBAC1 = train_triplets_jpg[['B', 'A', 'C']].copy()\nBCA0 = train_triplets_jpg[['B', 'C', 'A']].copy()\n\nACB0 = ACB0.rename(columns={'C':'B', 'B':'C'}, inplace=False)\nBAC1 = BAC1.rename(columns={'B':'A', 'A':'B'}, inplace=False)\nBCA0 = BCA0.rename(columns={'B':'A', 'C':'B', 'A':'C'}, inplace=False)\n\nid_fin = pd.concat((train_triplets_jpg, BAC1, ACB0, BCA0),axis=0, ignore_index=True)\nlabels = pd.concat((data_y1,data_y0),axis=0, ignore_index=True)\n\ndata_nn_x = pd.concat([id_fin,labels], join='outer', axis=1)\n\n# This only shuffels the set which might be unneccessary since ImageGen also shuffles. Still for redundanacy. Costs only 70ms on my machiene.\ndata_nn_x = data_nn_x.sample(frac=1).reset_index(drop=True) ")


# %%



# %%
#print(data_nn_x.loc[3,'label'].dtype)


# %%
batch_size = 48
pixels = 144


# %%
datagen=image.ImageDataGenerator(
    rescale=1./255.,
    validation_split=0.2,
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

test_datagen=image.ImageDataGenerator(
    rescale=1./255.,
    preprocessing_function=preprocess_input,
    horizontal_flip=True
    )


# %%
def generate_generator_multiple(generator, triplet_file, batch_size, pixels, subset):

    if subset == 'training':
        shuffle = True
        y_col = 'label'
        class_mode = 'binary'
    elif subset == 'validation':
        shuffle = False
        y_col = 'label'
        class_mode = 'binary'
    elif subset == None:
        shuffle = False
        y_col = None
        class_mode = 'input'
    else: raise Exception('Subset needs to be either training, validation or None.')

    generator_A = datagen.flow_from_dataframe(
        triplet_file,
        directory='./food_porn/food',
        x_col= 'A',
        y_col=y_col,
        subset=subset ,
        shuffle=shuffle,
        class_mode=class_mode,
        target_size=(pixels, pixels),
        batch_size=batch_size,
        seed=69,
        )

    generator_B = datagen.flow_from_dataframe(
        triplet_file,
        directory='./food_porn/food',
        x_col= 'B',
        y_col=y_col,
        subset=subset,
        shuffle=shuffle,
        class_mode=class_mode,
        target_size=(pixels, pixels),
        batch_size=batch_size,
        seed=69,
        )


    generator_C = datagen.flow_from_dataframe(
        triplet_file,
        directory='./food_porn/food',
        x_col= 'C',
        y_col=y_col,
        subset=subset,
        shuffle=shuffle,
        class_mode=class_mode,
        target_size=(pixels, pixels),
        batch_size=batch_size,
        seed=69,
    )

    while True:
            X1i = generator_A.next()
            X2i = generator_B.next()
            X3i = generator_C.next()

            yield [X1i[0], X2i[0], X3i[0]], X3i[1]  #Yield all images and their mutual label


# %%
input_generator = generate_generator_multiple(generator=datagen,
                                            triplet_file=data_nn_x,
                                            batch_size=batch_size,
                                            pixels=pixels,
                                            subset='training')       
     
validation_generator = generate_generator_multiple(generator=datagen,
                                            triplet_file=data_nn_x,
                                            batch_size=batch_size,
                                            pixels=pixels,
                                            subset='validation') 

test_generator = generate_generator_multiple(generator=test_datagen,
                                            triplet_file=test_triplets_jpg,
                                            batch_size=batch_size,
                                            pixels=pixels,
                                            subset=None)


# %%



# %%



# %%



# %%
""" vgg16 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg') """


# %%
""" vgg16.summary() """


# %%
""" abc.summary() """


# %%

""" 
inA = Input(shape=(244, 244, 3), dtype='float32')
inB = Input(shape=(244, 244, 3), dtype='float32')
inC = Input(shape=(244, 244, 3), dtype='float32')

outA = vgg16(inA)
outB = vgg16(inB)
outC = vgg16(inC)

combined = layers.Concatenate()([outA, outB, outC])


combined = Dense(units = 1024, activation='relu')(combined)
combined = Dense(units = 1, activation='sigmoid')(combined)

abc = Model(inputs=[inA, inB, inC], outputs=combined) """


# %%
""" abc.compile(loss = keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

#fit the network to (for now the un-sparse matrix)
abc.fit(train_generator, epochs=128, batch_size=32, verbose=1) """


# %%
cnn = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
cnn.save('ResNetV2')  # creates a HDF5 file 'my_model.h5'
del cnn

""" Comment this when uploading to euler. """


# %%
cnn.summary()


# %%
from keras import layers, models, applications

# Multiple inputs
in1 = layers.Input(shape=(pixels,pixels,3))
in2 = layers.Input(shape=(pixels,pixels,3))
in3 = layers.Input(shape=(pixels,pixels,3))

# CNN output
#cnn = applications.xception.Xception(include_top=False)
#cnn = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
cnn = load_model('ResNetV2')

#cnn.summary()

for layer in cnn.layers[:-4]:
    layer.trainable = False

out1 = cnn(in1)
out2 = cnn(in2)
out3 = cnn(in3)

# Flattening the output for the dense layer
fout1 = layers.Flatten()(out1)
fout2 = layers.Flatten()(out2)
fout3 = layers.Flatten()(out3)

# Getting the dense output
dense = layers.Dense(512, activation='relu')

dout1 = dense(fout1)
dout2 = dense(fout2)
dout3 = dense(fout3)

# Concatenating the final output
out = layers.Concatenate(axis=-1)([dout1, dout2, dout3])
out = Dropout(0.5)(out)
combined = Dense(units = 1024, activation='relu')(out)
combined = Dropout(0.5)(combined)
combined = Dense(units = 1, activation='sigmoid')(combined)

# Creating the model
model = Model(inputs=[in1, in2, in3], outputs=combined)
model.summary()


# %%
val_step = np.floor(59515*4*0.2/batch_size)
ep_step = np.floor(59515*4*0.8/batch_size)


# %%
model.compile(loss = keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

#fit the network to (for now the un-sparse matrix)
model.fit(input_generator, validation_data=validation_generator, epochs= 64,
          batch_size=batch_size, verbose=1, steps_per_epoch = ep_step,
          validation_steps= val_step)


# %%
predict_test_nn = model.predict(test_generator, batch_size=batch_size)


# %%
export = np.array(predict_test_nn>0.5).astype('int')
np.savetxt("submission_generator.txt", np.round(export,decimals=0), fmt='%i')


