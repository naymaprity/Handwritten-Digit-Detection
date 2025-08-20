import tkinter as tk #Creating the GUI
import numpy as np #Provides support for array manipulation
import cv2 #openCV library for image processing
from tensorflow.keras.models import load_model #used to load pretrained model(mnist_model.h5)
from PIL import ImageGrab, Image #(PLI=Pillow) Handles image capturing and processing of the canvas

# Load the pre-trained MNIST model
model = load_model('mnist_model.h5') #Loads the pre trained model which is trained on the Mnist dataset. the model expect 28*28 pixel image size as input

def preprocessing_image(img): #process the raw input image for prediction
    """Enhanced function to preprocess the drawn image for prediction."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY) #The image is converted to grayscale to simplify the processing
    
    # Invert and threshold the image to get a binary image (digit vs. background)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV) #Convert the grayscale image into binary(black and white)for clear digit background separation

    # Apply GaussianBlur for noise reduction
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0) #Smooth the image to reduce noise

    # Use morphological operations to better define the digit
    kernel = np.ones((3, 3), np.uint8) 
    morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel) #enhanced the structure of the digit by filling small gaps

    # Detect contours to isolate the digit
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Finds the outlines of the digit
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit = morph[y:y+h, x:x+w]
    else:
        digit = morph  # Fallback if no contours are found

    # Resize to 20x20 while maintaining aspect ratio, then pad to 28x28
    rows, cols = digit.shape
    aspect_ratio = rows / cols
    if aspect_ratio > 1:
        new_rows = 20
        new_cols = int(round(new_rows / aspect_ratio))
    else:
        new_cols = 20
        new_rows = int(round(new_cols * aspect_ratio))
    digit_resized = cv2.resize(digit, (new_cols, new_rows), interpolation=cv2.INTER_AREA) #Crop the digit and resize
    
    # Calculate padding to make it 28x28
    padding_top = (28 - new_rows) // 2
    padding_bottom = 28 - new_rows - padding_top
    padding_left = (28 - new_cols) // 2
    padding_right = 28 - new_cols - padding_left
    padded_digit = cv2.copyMakeBorder(digit_resized, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=0)

    # Normalize the image to a 0-1 range
    normalized_digit = padded_digit / 255.0  #Pixel value range 0 to 255, thats why divided by 255.0

    # Reshape the image to match the model's expected input (1, 28, 28, 1)
    return normalized_digit.reshape(1, 28, 28, 1) #Batch size, Height, Width, Channel

def predict_digit(img):
    """Function to predict the digit from the image."""
    preprocessed_image = preprocessing_image(img) #Calls preprocess the image
    prediction = model.predict(preprocessed_image) #{Predict the digit and calculate the confidence}
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Print the confidence directly without thresholding so predictions are always shown
    return predicted_digit, confidence  #Return the predicted digit and confidence

class App(tk.Tk):   #Defining the GUI
    def __init__(self):
        super().__init__()
        self.x = self.y = 0
        self.title("Handwritten Digit Recognition")
        
        # Create canvas and buttons
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross") #Canvas allow user to draw digit
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 24)) #label display the predicted digit and confidence score
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting) #Recognize classify the drawn digit
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all) #Clears the canvas
        
        # Place elements in the grid
        self.canvas.grid(row=0, column=0, pady=2, sticky="W")
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        
        # Bind the drawing event to the canvas
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        """Clear the canvas."""      #Clear the canvas by deleting all drawing and reset the label text
        self.canvas.delete("all")
        self.label.configure(text="Draw a digit")

    def classify_handwriting(self): #Recognize the handwriting digit
        """Capture the canvas, save as image, and classify the digit."""
        # Capture the canvas as a screenshot
        x1 = self.winfo_rootx() + self.canvas.winfo_x()
        y1 = self.winfo_rooty() + self.canvas.winfo_y()
        x2 = x1 + self.canvas.winfo_width()
        y2 = y1 + self.canvas.winfo_height()

        # Grab the screenshot from the canvas area
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2)) #Capture the canvas as an image using ImageGrab
        
        # Predict the digit
        digit, confidence = predict_digit(img) #Calls predict digit to predict the digit and confidence
        
        # Display the result without a confidence threshold
        self.label.configure(text=f"Digit: {digit}, Confidence: {int(confidence * 100)}%")

    def draw_lines(self, event):
        """Draw lines on the canvas."""    #Enables user to draw on the canvas by tracking the mouse motion
        self.x = event.x
        self.y = event.y
        r = 8  # Radius of the drawing point
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

# Run the application
app = App()
app.mainloop()  #initiates the APP class and runs the tkinter event loop
