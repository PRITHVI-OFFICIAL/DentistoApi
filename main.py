from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

np.set_printoptions(suppress=True)

model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
# Load the Keras model during app startup
model_path = "keras_model.h5"
loaded_model = tf.keras.models.load_model(model_path)

@app.get("/")
async def root():
    return "Dentisto Api"


@app.get("/check")
async def root():
    return "Calling Render.com for every 14 min...."

# Define the prediction route
@app.post("/predict")
async def predict_teeth_decay(photo: UploadFile = File(...)):
    # image_bytes = await photo.read()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    contents = photo.file.read()
    path="teeth.jpg"
    with open(path, "wb") as f:
        f.write(contents)



    image = Image.open(path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print(prediction)

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    result= f"result : {class_name[2:]} - accuracy score : {confidence_score}"

    return result
