from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import render_template

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

# For home page return index.html
@app.route("/",methods=["GET"])
def home():
	return render_template("index.html")

@app.route("/success", methods=["POST"])
def success():
	if 'image' not in request.files:
		return jsonify({"error": "Missing file"}), 400
	else:
		# Get the image file from the request
		image_file = request.files["image"]
		print("Received image File")
		#   Get the file extension (file name suffix)
		extension = image_file.filename.split(".")[-1]
		# Resize to 128 by 128 size
		im=Image.open(image_file)
		im=im.resize((128,128))
		# Save the image in the uploads folder
		im.save("static/input." + extension)
		# Open the image file and convert it to a NumPy array
		img = Image.open(image_file).convert("RGB")
		img = np.array(img)
		if img is None:
			return jsonify({"error": "Failed to read image file"}), 400
		img = tf.image.resize(img, [128, 128])
		img = tf.cast(img, tf.float32) / 255.0
		# Make a prediction using the pre-trained model
		pred = model.predict(np.array([img]))

		# Convert the prediction to a binary mask
		mask = np.argmax(pred, axis=-1)
		mask = np.expand_dims(mask, axis=-1)
		mask = tf.image.resize(mask, [img.shape[0], img.shape[1]])

		# Convert the binary mask to a PNG image
		mask = tf.keras.preprocessing.image.array_to_img(mask[0])
		mask.save("static/output.png")
		
		# Render the output HTML and original image
		return render_template("display_image.html", image_name="input." + extension)


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
