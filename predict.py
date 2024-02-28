from keras.preprocessing import image 
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Creating list for mapping 
list_ =  ['B-1', 'B-2', 'B-52', 'BareLand', 'C-130', 'C-135', 'C-5', 'E-3', 'KC-10']
model = tf.keras.models.load_model('recognition.model')
#Input image 
base_dir = 'Planes'

img_size = 224
batch = 64
test_image = image.load_img(r'testing\e3real.jpg',target_size=(224,224)) 
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, 
								validation_split=0.2) 
test_datagen = test_datagen.flow_from_directory(base_dir, 
												target_size=( 
													img_size, img_size), 
												subset='validation', 
												batch_size=batch) 

#For show image 
plt.imshow(test_image) 
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image,axis=0) 

#Result array 
result = model.predict(test_image) 
print(result) 
#model.evaluate(test_datagen)
#Mapping result array with the main name list 
i=0
for i in range(len(result[0])): 
    if(result[0][i]==1): 
	    print(list_[i])
    
