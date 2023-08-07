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


    # # Read and process the image
    # image_bytes = await image.read()
    # image_pil = Image.open(io.BytesIO(image_bytes))

    # # Preprocess the image to make it compatible with the model
    # # For example, resize the image, normalize pixel values, etc.
    # # Depending on how the model was trained, you may need to preprocess the image accordingly.

    # # Convert the image to a NumPy array
    # # processed_image = ...

    # # Make predictions using the loaded Keras model
    # # predictions = loaded_model.predict(processed_image)

    # # Replace the above line with actual prediction code based on your model and preprocessing steps

    # # For demonstration purposes, we'll return a dummy result
    # # You should replace this with the actual result based on your model predictions
    # result = {
    #     "filename": image.filename,
    #     "result": "decayed" if "decay" in image.filename.lower() else "healthy",
    # }

    # return result
