from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import cv2
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == "POST":
        img = request.files['img']
        model = request.form['model']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(file_path)
        predict = predictImage(file_path, model)
        os.remove(file_path)
        predict = predict.capitalize()
        return render_template('search.html', predict=predict)
    return render_template('search.html')

def get_class_names(data_dir):
    class_names = os.listdir(data_dir)
    return class_names

def predictImage(img, model):
    
    model_path = f"Model/{model}.hdf5"  
    model = load_model(model_path)
    image = cv2.imread(img)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1, 224, 224, 3)

    image_size = (224, 224)
    batch_size = 64
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.4,
                                zoom_range = 0.4,
                                horizontal_flip = True,
                                vertical_flip = True,
                                validation_split = 0.2)
    train_ds = train_datagen.flow_from_directory('Train',
                                      target_size = image_size,
                                      batch_size = batch_size,
                                      class_mode = 'categorical',
                                      subset = 'training',
                                      color_mode="rgb",)
    label_names = train_ds.class_indices
    dict_class = dict(zip(list(range(len(label_names))), label_names))
    clas = model.predict(image).argmax()
    name = dict_class[clas]
    return name

if __name__ == '__main__':
    app.run(debug=True)
