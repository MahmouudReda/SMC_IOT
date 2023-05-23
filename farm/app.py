from flask import Flask, request, render_template, url_for, jsonify
from PIL import Image
import numpy as np
###################################
IMG_DIM=224
############################
app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    #image_arr.shape = (1, 150, 150, 3)
    return image_arr

CLASSES=['Apple scab',
 'Apple Black_rot',
 'Apple Cedar apple rust',
 'Apple healthy',
 'Blueberry healthy',
 'Cherry (including_sour) Powdery mildew',
 'Cherry (including_sour) healthy',
 'Corn (maize) Cercospora leaf spot Gray leaf spot',
 'Corn (maize) Common rust',
 'Corn (maize) Northern Leaf Blight',
 'Corn (maize) healthy',
 'Grape Black rot',
 'Grape Esca (Black Measles)',
 'Grape Leaf_blight (Isariopsis Leaf Spot)',
 'Grape healthy',
 'Orange Haunglongbing (Citrus greening)',
 'Peach Bacterial spot',
 'Peach healthy',
 'Pepper bell Bacterial spot',
 'Pepper bell healthy',
 'Potato Early blight',
 'Potato Late blight',
 'Potato healthy',
 'Raspberry healthy',
 'Soybean healthy',
 'Squash Powdery mildew',
 'Strawberry Leaf scorch',
 'Strawberry healthy',
 'Tomato Bacterial spot',
 'Tomato Early blight',
 'Tomato Late blight',
 'Tomato Leaf Mold',
 'Tomato Septoria leaf spot',
 'Tomato Spider mites Two-spotted spider mite',
 'Tomato Target Spot',
 'Tomato Yellow Leaf Curl Virus',
 'Tomato mosaic virus',
 'Tomato healthy']


################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAvgPool2D, Dense, Flatten,Dropout
model= Sequential()

model.add(Conv2D(64, 3, input_shape=(IMG_DIM, IMG_DIM, 3), activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(96, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(256, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(512, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
#Fully C L
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(38, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
MODEL_PATH = 'saved-models/pdcnn-best'
############################################################################3

print(MODEL_PATH)
model.load_weights(MODEL_PATH)
#####################################################################

@app.route('/')
def index():

    return render_template('index.html', appName="plant diseases classify")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(np.expand_dims(image_arr, 0))
        print("Model predicted")
        ind = np.argmax(result)
        prediction = CLASSES[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(np.expand_dims(image_arr, 0))
        print("predicted ...")
        ind = np.argmax(result)
        prediction = CLASSES[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="plant diseases classify")
    else:
        return render_template('index.html',appName="plant diseases classify")


if __name__ == '__main__':
    app.run(debug=True)