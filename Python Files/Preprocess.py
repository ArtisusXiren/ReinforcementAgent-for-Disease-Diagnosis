import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pydicom
from PIL import Image
import numpy as np

data=pd.read_csv(r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\train.csv')
categorical=data.iloc[:, 1:].select_dtypes(include='object').columns
for column in categorical:
    mapping={name:idx for idx,name in enumerate(data[column].unique())}
    data[column]=data[column].map(mapping)
image_to_label=dict(zip(data['Patient'],data['SmokingStatus']))
path=r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\train'
output_dir=r'C:\Users\ArtisusXiren\Desktop\pulmonaryfibrosis\orgnaized'

for img_name,label in image_to_label.items():
    img_path=os.path.join(path,img_name)
    if os.path.exists(img_path):
        class_directory=os.path.join(output_dir,str(label))
        os.makedirs(class_directory,exist_ok=True)
        for dicom_file in os.listdir(img_path):
            if dicom_file.endswith('.dcm'):  
                src_file = os.path.join(img_path, dicom_file)
                try:
                    dcm = pydicom.dcmread(src_file)
                    img_array = dcm.pixel_array.astype(float)
                    img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0  
                    img = Image.fromarray(img_array.astype(np.uint8))
                    dst_file = os.path.join(class_directory, f"{img_name}_{dicom_file}.jpg")
                    img.save(dst_file)

                except Exception as e:
                    print(f"Failed to process {src_file}: {e}")

print('Conversion and organization complete.')

train_df,val_df=train_test_split(data,test_size=0.2,random_state=42)

