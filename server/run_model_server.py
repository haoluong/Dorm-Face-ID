# import the necessary packages
import numpy as np
import settings
import helpers
import redis
import base64
import time
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

def classify_process():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	print("* Loading model...")
	base_model = tf.keras.applications.MobileNetV2(input_shape=(settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANS),
                                               include_top=False,
                                               weights='imagenet')
	model = Sequential([base_model, GlobalAveragePooling2D(), Dense(497, activation='softmax')])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.load_weights(settings.CHECKPOINT_PATH)
	print("* Model loaded")

	# continually pool for new images to classify
	while True:
		# attempt to grab a batch of images from the database, then
		# initialize the image IDs and batch of images themselves
		queue = db.lrange(settings.IMAGE_QUEUE, 0,
			settings.BATCH_SIZE - 1)
		imageIDs = []
		batch = None

		# loop over the queue
		for q in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
			image = helpers.base64_decode_image(q["image"],
				settings.IMAGE_DTYPE,
				(1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
					settings.IMAGE_CHANS))
			# check to see if the batch list is None
			if batch is None:
				batch = image

			# otherwise, stack the data
			else:
				batch = np.vstack([batch, image])

			# update the list of image IDs
			imageIDs.append(q["id"])

		# check to see if we need to process the batch
		if len(imageIDs) > 0:
			# classify the batch
			print("* Batch size: {}".format(batch.shape))
			preds = model.predict(batch)
			results = helpers.decode_predictions(preds)
			
			# loop over the image IDs and their corresponding set of
			# results from our model
			for (imageID, resultSet) in zip(imageIDs, results):
				# initialize the list of output predictions
				output = []

				# loop over the results and add them to the list of
				# output predictions
				label, prob = resultSet
				r = {"label": label, "probability": float(prob)}
				output.append(r)

				# store the output predictions in the database, using
				# the image ID as the key so we can fetch the results
				db.set(imageID, json.dumps(output))

			# remove the set of images from our queue
			db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

		# sleep for a small amount
		time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
	classify_process()
