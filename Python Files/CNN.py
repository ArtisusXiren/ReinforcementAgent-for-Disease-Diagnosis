import glob
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model  
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from Preprocess import train_df,val_df
from keras.utils import load_img, img_to_array


output_dir = r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized'
batch_size = 32
image_size = (128, 128)

# Use ImageDataGenerator to handle augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.15,
    zoom_range=0.15,
     
)
def custom_data_generator(data,batch_size,output_dir,datagen):
    while True:
        for start in range(0,len(data),batch_size):
            end=min(start+batch_size,len(data))
            batch_data=data[start:end]
            FVC=[]
            images_batch=[]
            for i,row in batch_data.iterrows():
                    patient_dir=os.path.join(output_dir,str(row['SmokingStatus']))
                    imgs_path=glob.glob(os.path.join(patient_dir,'*jpg'))
                    images=[]
                    for img_path in imgs_path:
                        img=load_img(img_path,target_size=(128,128),color_mode='grayscale')
                        img=img_to_array(img)
                        images.append(img)
                    if len(images)>0:
                        images=np.stack(images)
                        augumented_images=[datagen.random_transform(image) for image in images ]
                        images_batch.extend(augumented_images)
                        FVC.extend([row['SmokingStatus']]*len(augumented_images))
                        
                    if len(images)>=batch_size:
                        yield tf.convert_to_tensor(images_batch[:batch_size]), tf.convert_to_tensor(FVC[:batch_size])
                        images_batch=[]
                        FVC=[]
            if len(images_batch)>0 and len(FVC)>0:
                images_batch=np.array(images_batch)
                FVC=np.array(FVC)
                yield tf.convert_to_tensor(images_batch),tf.convert_to_tensor(FVC)
        
train_generator = custom_data_generator(
    train_df,batch_size,output_dir,datagen
)

val_generator = custom_data_generator(
val_df,batch_size,output_dir,datagen
)
                                    

model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.02)),  
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.02)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.02)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.02)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),  
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.save(r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized\model.keras')


early_stopping= EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)
checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)


model_checkpoint=load_model(r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\model_checkpoint.keras')

history=model.fit(
    train_generator,
    steps_per_epoch = len(train_df) // batch_size,
    validation_data=val_generator,
    validation_steps = len(val_df) // batch_size,
    epochs=10,
    verbose=1,
    callbacks=[checkpoint_callback,early_stopping]
)

