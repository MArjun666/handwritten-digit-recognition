from keras.models import load_model
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import win32gui

# Load the trained model from file
try:
    model = load_model('model.h5')
    print("Model loaded successfully from model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'model.h5' is in the same directory.")
    model = None

def predict_digit(img):
    if model is None:
        return "Model not loaded", 0

    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert the image to grayscale
    img = img.convert('L')
    # Convert the image to a NumPy array
    img = np.array(img)
    # Reshape the array to match the model input shape (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)
    # Normalize the pixel values to the range [0, 1]
    img = img / 255.0

    # Predict the digit using the trained model
    res = model.predict(img)[0]
    
    # Return the predicted digit and the confidence score
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.x = self.y = 0
        
        # --- UI Elements ---
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize Digit", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear Canvas", command=self.clear_all)

        # --- Grid Layout ---
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W, columnspan=2)
        self.label.grid(row=0, column=2, pady=2, padx=10, columnspan=2)
        self.button_clear.grid(row=1, column=0, pady=5, padx=5, sticky=tk.W)
        self.classify_btn.grid(row=1, column=1, pady=5, padx=5, sticky=tk.E)

        # Bind drawing event to the canvas
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        # Clear the canvas and reset the label
        self.canvas.delete("all")
        self.label.configure(text="Draw a digit...")

    def classify_handwriting(self):
        # Get the handle of the canvas widget
        hwnd = self.canvas.winfo_id()
        # Get the bounding box of the canvas
        rect = win32gui.GetWindowRect(hwnd)
        
        # Add a small margin to ensure the whole drawing is captured
        x1, y1, x2, y2 = rect
        rect_with_margin = (x1 + 4, y1 + 4, x2 - 4, y2 - 4)
        
        try:
            # Capture the drawn image from the canvas area
            im = ImageGrab.grab(rect_with_margin)
            
            # Predict the digit and get the confidence score
            digit, acc = predict_digit(im)
            self.label.configure(text=f'Predicted: {digit}\nConfidence: {int(acc*100)}%')
        except Exception as e:
            print(f"Error during classification: {e}")
            self.label.configure(text="Error")

    def draw_lines(self, event):
        # Draw on the canvas
        r = 8 # Brush radius
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='black', outline='black')

if __name__ == '__main__':
    app = App()
    app.mainloop()
