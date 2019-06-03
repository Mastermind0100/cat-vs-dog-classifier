"""
Created on Mon Jun  3 11:45:05 2019

@author: AtharvaHudlikar
"""
from keras.preprocessing import image

model=load_model('modelwts.h5')

test_image=image.load_img('directory/sample_image.jpg', target_size = (64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis = 0)
result=model.predict(test_image)
training_set.class_indices

if result[0][0] >= 0.5:
    prediction='Dog'
else:
    prediction='Cat'
print(prediction)
