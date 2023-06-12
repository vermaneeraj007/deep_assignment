


import fasttext

model = fasttext.train_supervised('training_data.txt', lr=1.0, epoch=25, wordNgrams=2)


# Test the model on a test sample
test_sample = "This is a test text in English"

# Predict the language of the test sample
prediction = model.predict(test_sample)
predicted_language = prediction[0][0].split('__')[-1]

print("Predicted language:", predicted_language)


# Save the trained model
model.save_model('language_detection_model.bin')

# Load the trained model
loaded_model = fasttext.load_model('language_detection_model.bin')
