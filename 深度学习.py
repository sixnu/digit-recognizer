
import numpy as np
import pandas as pd


data_train=pd.read_csv('.../digit-recognizer/train.csv')
data_test=pd.read_csv('.../digit-recognizer/test.csv')

data_train.shape
data_test.shape

from sklearn.model_selection import cross_val_score,train_test_split
x_train, x_valid, y_train, y_valid =train_test_split(X,Y,train_size=0.8,random_state=10)
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding,BatchNormalization
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#
X_train = data_train.iloc[:,1:]
y_train = data_train.iloc[:,0]

# Reshape and normalize image
X_train = X_train.values.reshape(-1, 28, 28, 1)/255.
test = data_test.values.reshape(-1, 28, 28, 1)/255.
# One Hot encoding the label
y_train = to_categorical(y_train, 10)
random_seed = 0
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=random_seed)

model=models.Sequential()
model.add(layers.Conv2D(32,(5,5), padding='same',activation='relu',input_shape=X_train.shape[1:]))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.15))
model.add(layers.Conv2D(64,(5,5), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Conv2D(64,(3,3), padding='same',activation='relu'))
model.add(layers.Dropout(0.15))
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

callback_list = [
    ReduceLROnPlateau(monitor='val_loss', patience=1, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001),
    EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=4)    
]

#ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.5)
                  
model.fit(X_train, y_train,epochs=25,batch_size=20,validation_data=(X_val, y_val),
          steps_per_epoch=X_train.shape[0] // 20,callbacks=callback_list)

results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1,28001), name='ImageID'), results], axis=1)
submission.to_csv('digit_mod_predictions.csv', index=False)
