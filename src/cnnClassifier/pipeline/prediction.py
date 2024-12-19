import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        ## load model
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model","model.h5"))
       

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image / 255.0 
        
        output = model.predict(test_image)
        print(output)  
        result = np.argmax(output, axis=1)

        print(result)

        if result[0] == 0:
          prediction = 'Adenocarcinoma Cancer'
        elif result[0] == 1:
          prediction = 'Large cell carcinoma'
        elif result[0] == 2:
          prediction = 'Normal'
        elif result[0] == 3:
           prediction = 'Squamous cell carcinoma'

        return [{"image": prediction}]

    
        
       
