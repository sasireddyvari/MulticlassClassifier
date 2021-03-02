import numpy as np
from keras.models import load_model
from keras_preprocessing import image

class multiClass_Classifier:
    def __init__(self,filename):
        self.filename=filename

    def predictMultiClass(self):
        model=load_model('multiClassClassifierCNN_Model.h5')
        #model.summary()
        imagename=self.filename
        test_image = image.load_img(imagename, target_size=(64, 64))
        test_image= image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result=model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'BlackPlum Tree'
            return [{"image": prediction}]
        elif result[0][1] == 1:
            prediction = 'Coconut Tree'
            return [{"image": prediction}]
        elif result[0][2] == 1:
            prediction = 'MS Dhoni'
            return [{"image": prediction}]
        elif result[0][3] == 1:
            prediction = 'Rohit Sharma'
            return [{"image": prediction}]
        elif result[0][4] == 1:
            prediction = 'Sasi Kumar'
            return [{"image": prediction}]
        else:
            prediction = "Picture Not Identified"
            return [{"image": prediction}]