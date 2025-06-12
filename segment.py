import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# ==================== SEGMENTATION CODE ====================
class BrailleImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        self._preprocess()
    
    def _preprocess(self):
        _, self.binary_image = cv2.threshold(self.original_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        self.edged_image = cv2.Canny(self.binary_image, 50, 150)
    
    def get_binary_image(self): return self.binary_image
    def get_edged_binary_image(self): return self.edged_image
    def get_original_image(self): return self.original_image

class BrailleCharacter:
    def __init__(self, dots, diameter, radius, image):
        self.dots = dots
        self.diameter = diameter
        self.radius = radius
        self.image = image
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0
    
    def get_dots(self): return self.dots
    def get_bounding_box(self): return (self.left, self.right, self.top, self.bottom)

class SegmentationEngine: 
    def __init__(self, image=None):
        self.image = image
        self.initialized = False
        self.dots = []
        self.diameter = 0.0
        self.radius = 0.0
        self.next_epoch = 0
        self.characters = []

    def __iter__(self): return self
    def __next__(self): return self.next()

    def next(self):
        if not self.initialized:
            self.initialized = True
            contours = self.__process_contours()
            if len(contours) == 0: self.__clear(); raise StopIteration()
            enclosingCircles = self.__get_min_enclosing_circles(contours)
            if len(enclosingCircles) == 0: self.__clear(); raise StopIteration()

            diameter,dots,radius = self.__get_valid_dots(enclosingCircles)
            if len(dots) == 0: self.__clear(); raise StopIteration()
            self.diameter = diameter
            self.dots = dots
            self.radius = radius
            self.next_epoch = 0
            self.characters = []

        if len(self.characters) > 0:
            r = self.characters[0]
            del self.characters[0]
            return r

        cor = self.__get_row_cor(self.dots, epoch=self.next_epoch)
        if cor is None: self.__clear(); raise StopIteration()

        top = int(cor[1] - int(self.radius*1.5))
        self.next_epoch = int(cor[1] + self.radius)

        cor = self.__get_row_cor(self.dots,self.next_epoch,self.diameter,True)
        if cor is None: self.next_epoch = int(self.next_epoch + (2*self.diameter))
        else: self.next_epoch = int(cor[1] + self.radius)

        cor = self.__get_row_cor(self.dots,self.next_epoch,self.diameter,True)
        if cor is None: self.next_epoch = int(self.next_epoch + (2*self.diameter))
        else: self.next_epoch = int(cor[1] + self.radius)
        
        bottom = self.next_epoch
        self.next_epoch += int(2*self.diameter)

        DOI = self.__get_dots_from_region(self.dots, top, bottom)
        xnextEpoch = 0
        while True:
            xcor = self.__get_col_cor(DOI, xnextEpoch)
            if xcor is None: break

            left = int(xcor[0] - self.radius)
            xnextEpoch = int(xcor[0] + self.radius)
            xcor = self.__get_col_cor(DOI,xnextEpoch,self.diameter,True)
            if xcor is None: xnextEpoch += int(self.diameter*1.5)
            else: xnextEpoch = int(xcor[0]) + int(self.radius)
            right = xnextEpoch
            box = (left, right, top, bottom)
            dts = self.__get_dots_from_box(DOI, box)
            char = BrailleCharacter(dts, self.diameter, self.radius, self.image)
            char.left = left
            char.right = right
            char.top = top
            char.bottom = bottom
            self.characters.append(char)

        if len(self.characters) < 1: self.__clear(); raise StopIteration()
        r = self.characters[0]
        del self.characters[0]
        return r

    def __clear(self):
        self.image = None
        self.initialized = False
        self.dots = []
        self.diameter = 0.0
        self.radius = 0.0
        self.next_epoch = 0
        self.characters = []

    def __get_row_cor(self, dots, epoch=0, diameter=0, respectBreakpoint=False):
        if len(dots) == 0: return None
        minDot = None
        for dot in dots:
            x,y = dot[0]
            if y < epoch: continue
            if minDot is None: minDot = dot
            else:
                v = int(y - epoch)
                minV = int(minDot[0][1] - epoch)
                if minV > v: minDot = dot
        if minDot is None: return None
        if respectBreakpoint:
            v = int(minDot[0][1] - epoch)
            if v > (2*diameter): return None
        return minDot[0]

    def __get_col_cor(self, dots, epoch=0, diameter=0, respectBreakpoint=False):
        if len(dots) == 0: return None
        minDot = None
        for dot in dots:
            x,y = dot[0]
            if x < epoch: continue
            if minDot is None: minDot = dot
            else:
                v = int(x - epoch)
                minV = int(minDot[0][0] - epoch)
                if minV > v: minDot = dot
        if minDot is None: return None
        if respectBreakpoint:
            v = int(minDot[0][0] - epoch)
            if v > (2*diameter): return None
        return minDot[0]

    def __get_dots_from_box(self, dots, box):
        left,right,top,bottom = box
        return [dot for dot in dots if left <= dot[0][0] <= right and top <= dot[0][1] <= bottom]

    def __get_dots_from_region(self, dots, y1, y2):
        return [dot for dot in dots if y1 < dot[0][1] < y2] if y2 > y1 else []

    def __get_valid_dots(self, circles):
        tolerance = 0.45
        radii = []
        consider = []
        bin_img = self.image.get_binary_image()
        for circle in circles:
            x,y = circle[0]
            rad = circle[1]
            it = 0
            while it < int(rad):
                if bin_img[y,x+it] > 0 and bin_img[y+it,x] > 0: it += 1
                else: break
            else:
                if bin_img[y,x] > 0:
                    consider.append(circle)
                    radii.append(rad)

        baserad = Counter(radii).most_common(1)[0][0]
        dots = [circle for circle in consider if int(baserad*(1-tolerance)) <= circle[1] <= int(baserad*(1+tolerance))]

        for dot in dots:
            X1,Y1 = dot[0]
            C1 = dot[1]
            for sdot in dots:
                if dot == sdot: continue
                X2,Y2 = sdot[0]
                C2 = sdot[1]
                D = sqrt(((X2-X1)**2)+((Y2-Y1)**2))
                if C1 > (D + C2): dots.remove(sdot)
        
        radii = [dot[1] for dot in dots]
        baserad = Counter(radii).most_common(1)[0][0] 
        return 2*(baserad), dots, baserad
            
    def __get_min_enclosing_circles(self, contours):
        return [( (int(x),int(y)), int(radius) ) for contour in contours for (x,y),radius in [cv2.minEnclosingCircle(contour)]]

    def __process_contours(self):
        edg_bin_img = self.image.get_edged_binary_image()
        contours = cv2.findContours(edg_bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0] if len(contours) == 2 else contours[1]

class BrailleSegmenter:
    def __init__(self, image_path):
        self.image = BrailleImage(image_path)
        self.segmentation_engine = SegmentationEngine(self.image)
    
    def process(self):
        return [char for char in self.segmentation_engine]

# ==================== RECOGNITION CODE ====================
def load_model_and_encoder():
    # Load your trained model
    model = keras.models.load_model("braille_model.h5")
    
    # Create label encoder (should match your training data)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z', ' '
    ])
    
    return model, label_encoder

def preprocess_char_image(img, size=64):
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)

def recognize_braille(image_path, model, label_encoder, min_confidence=0.7):
    # Segment the image
    segmenter = BrailleSegmenter(image_path)
    
    characters = segmenter.process()
    
    # Load original image for visualization
    original_img = cv2.imread(image_path)
    vis_img = original_img.copy()
    
    # Process each character with space detection
    recognized_text = []
    prev_right = 0
    space_threshold = None  # Will be calculated based on character spacing
    
    for i, char in enumerate(characters):
        left, right, top, bottom = char.get_bounding_box()
        
        # Calculate space threshold based on first two characters
        if i == 1 and len(characters) > 1:
            space_threshold = (characters[1].left - characters[0].right) * 1.5
        
        # Detect space between words
        if i > 0 and space_threshold is not None:
            space_width = left - prev_right
            if space_width > space_threshold:
                recognized_text.append(' ')
        
        # Get character image and predict
        char_img = original_img[top:bottom, left:right]
        processed_img = preprocess_char_image(char_img)
        
        # First check if it's an empty cell (space)
        if np.mean(char_img) > 240:  # Empty cell detection (mostly white)
            recognized_text.append(' ')
        else:
            predictions = model.predict(processed_img)
            predicted_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            if confidence >= min_confidence:
                predicted_char = label_encoder.inverse_transform([predicted_idx])[0]
                recognized_text.append(predicted_char)
            else:
                recognized_text.append('?')
        
        # Update previous right position
        prev_right = right
        
        # Visualization markup
        cv2.rectangle(vis_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(vis_img, f"{recognized_text[-1]}", 
                   (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Post-process spaces
    braille_text = ''.join(recognized_text)
    
    # Convert Braille space indicators to actual spaces
    english_text = braille_text.replace(' ', ' ')
    
    # Remove duplicate spaces
    english_text = ' '.join(english_text.split())
    
    return {
        'braille': braille_text,
        'english': english_text,
        'visualization': vis_img
    }

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Load model and label encoder
    try:
        model, label_encoder = load_model_and_encoder()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Hardcoded image path
    image_path = r"C:/Users/jayad/Downloads/sample6.png"
    
    try:
        # Process the image
        results = recognize_braille(image_path, model, label_encoder)
        
        # Display results
        print("\n=== BRAILLE RECOGNITION RESULTS ===")
        # print(f"Braille Sequence: {results['braille']}")
        print(f"English Text: {results['english']}")
        
        # Show visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB))
        plt.title("Braille Recognition Results")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error processing image: {e}")