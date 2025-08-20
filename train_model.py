import tensorflow as tf
from tensorflow.keras.datasets import mnist  #Tensorflow , keras used for building traning and save the neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2  #handles image preprocessing task
import numpy as np #handles numerical operation such as array manipulation

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() #Mnist contains 600000 image and 10000 testing image

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0  #{Reshape the data to match the shape, normalize pixel value to the range [0 to 1]}
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0  #""
y_train = to_categorical(y_train, 10)  #Each label represented as 10 dimensional binary vector
y_test = to_categorical(y_test, 10)

# Adjusted data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,  # Slightly reduced rotation
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,              #Augmentation helps prevent overfitting
    fill_mode='nearest'
)
datagen.fit(x_train)

# Build an improved model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  #Conv is Convolutional layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),  #MaxPooling2D: A pooling layer that reduces the spatial dimensions of the feature maps.
                              #(2, 2): The pooling window size (2x2 pixels).
                             #Purpose: It down-samples the feature maps by taking the maximum value from each 2x2 region. This reduces computational complexity and helps make the model invariant to small translations in the input image.
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),    #3,3 is the size of each filter (3*3 pixel)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'), #Rectified linear unit (Relu) Activation function of hidden layers
    Dropout(0.4),
    Dense(10, activation='softmax')  #Softmax is activation function to generate probabilities for multi class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  #Adaptive Moment Estimation (Adam) is used for minimize the loss

# Add EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model using the data generator
model.fit(datagen.flow(x_train, y_train, batch_size=32), 
          validation_data=(x_test, y_test),
          epochs=30,                   #evaluates the model on the test set after each epoch
          callbacks=[early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")  #calculate the loss and accuracy on the test set

# Save the model
model.save('mnist_model.h5')
print("Model saved as 'mnist_model.h5'")  #save for further use

# Updated preprocessing function
def preprocessing_image(img):
    """Improved function to preprocess the drawn image for prediction."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to enhance digit contrast
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Apply Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

    # Find contours and select the largest one
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit = blurred[y:y+h, x:x+w]
    else:
        digit = blurred

    # Resize the digit to 20x20 while maintaining aspect ratio
    rows, cols = digit.shape
    aspect_ratio = rows / cols
    if aspect_ratio > 1:
        new_rows = 20
        new_cols = int(round(new_rows / aspect_ratio))
    else:
        new_cols = 20
        new_rows = int(round(new_cols * aspect_ratio))

    digit_resized = cv2.resize(digit, (new_cols, new_rows), interpolation=cv2.INTER_AREA)

    # Pad the image to make it 28x28
    padding_top = (28 - new_rows) // 2
    padding_bottom = 28 - new_rows - padding_top
    padding_left = (28 - new_cols) // 2
    padding_right = 28 - new_cols - padding_left
    padded_digit = cv2.copyMakeBorder(digit_resized, padding_top, padding_bottom, 
                                      padding_left, padding_right, 
                                      cv2.BORDER_CONSTANT, value=0)

    # Normalize the image to a 0-1 range
    normalized_digit = padded_digit / 255.0

    # Reshape for model input
    return normalized_digit.reshape(1, 28, 28, 1)

# Function to predict digit with improved feedback
def predict_digit(img):
    """Predicts the digit from the preprocessed image."""
    preprocessed_image = preprocessing_image(img)
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Return the prediction and confidence
    return predicted_digit, confidence
