"""----------------------------------------------------------------------------
TITLE       : digit_recog_norm.py
BY          : Sang Yoon Byu
DESCRIPTION : A program that can recognize and identify handwritten 
              digits (from 0 to 9) by a user.
----------------------------------------------------------------------------"""

import sys
import numpy as np
import cv2

"""============================================================================
PROCEDURE:
    onMouse
PARAMETERS:
    event, any cv2.MouseEventTypes events
    x, x-coordinate of the event
    y, y-coordinate of the event
    flags, any cv2.MouseEventFlags flags
    param, any additional parameters 
PURPOSE:
    a callback function for mouse events (writes digits on screen)
PRODUCES:
    None - a void function
============================================================================"""


def on_mouse(event, x, y, flags, param):

    global old_x, old_y

    # Get the blank screen from parameter
    screen = param

    if event == cv2.EVENT_LBUTTONDOWN:  # Left button pressed
        old_x, old_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # Left button released
        old_x, old_y = -1, -1

    elif event == cv2.EVENT_MOUSEMOVE:  # Moving / dragging
        if flags & cv2.EVENT_FLAG_LBUTTON:
            # Draw white lines as the cursor moves - writing the digit
            cv2.line(screen, (old_x, old_y), (x, y), (255, 255, 255), 20,
                     cv2.LINE_AA)
            old_x, old_y = x, y
            cv2.imshow('Screen', screen)


"""============================================================================
PROCEDURE:
    norm_digits
PARAMETERS:
    img, an image
PURPOSE:
    Calculate the centroid of an image (the current position of the digit), 
    and place it at the center of the image
PRODUCES:
    dst, a new image with a digit at the center of the image
============================================================================"""


def norm_digit(img):

    # Dictionary of all the moments within the given image
    m = cv2.moments(img)

    # Centroid information (by definition)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']

    # Attain height and width information of image
    h, w = img.shape[:2]

    # Matrix holding transformation information
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)

    # Transform using warpAffine()
    dst = cv2.warpAffine(img, aff, (0, 0))

    return dst


"""============================================================================
PROCEDURE:
    display_text
PARAMETERS:
    img, an image
    text, a string value
PURPOSE:
    displays given text at the top-center of an image 
PRODUCES:
    None - a void function
============================================================================"""


def display_text(img, text):

    # Set up text elements
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    text_scale = 0.5
    text_thickness = 1
    text_size = cv2.getTextSize(text, text_font, text_scale, text_thickness)

    # Calculate the text display position
    window_x = img.shape[0]
    window_y = img.shape[1]

    text_x = int((window_y - text_size[0][0]) / 2)
    text_y = window_x - 30   # Giving some top-margin

    # Display text
    cv2.putText(img, text, (text_x, text_y), text_font, text_scale,
                text_color, text_thickness, cv2.LINE_AA)


"""============================================================================
PROCEDURE:
    write_and_classify_digit
PARAMETERS:
    hog, 
    svm, a trained SVM model
PURPOSE:
    Open a blank screen for the user to write a digit. Then use the 
    trained SVM to classify user-written digit 
PRODUCES:
    
============================================================================"""


def write_and_classify_digit(hog, svm):

    # Initialized x, y values for writing digits
    old_x, old_y = -1, -1

    # Create a blank screen for users to write on
    screen = np.zeros((400, 400), np.uint8)

    display_text(screen, "Please write a number (0-9)")

    cv2.imshow('Screen', screen)
    cv2.setMouseCallback('Screen', on_mouse, screen)

    while True:
        key = cv2.waitKey()

        if key == 27:  # ESC key
            break
        elif key == ord(' '):  # Space key
            # Resize user-drawn image to 20x20
            test_image = cv2.resize(screen, (20, 20),
                                    interpolation=cv2.INTER_AREA)

            # Normalization
            test_image = norm_digit(test_image)

            # Compute Hog descriptors
            test_desc = hog.compute(test_image).T

            # Determine the user-written digit
            _, res = svm.predict(test_desc)
            print("The digit is:", int(res[0, 0]))

            # Reset black input screen
            screen.fill(0)
            display_text(screen, "Please write a number (0-9)")
            cv2.imshow('Screen', screen)

    cv2.destroyAllWindows()


"""============================================================================
                                     MAIN
============================================================================"""


def main():

    # Attain the image file containing 5000 digit samples
    digits = cv2.imread('./Projects/digit_recognition/digits.png',
                        cv2.IMREAD_GRAYSCALE)

    # Check for opening image
    if digits is None:
        print("Error: Image not found.")
        sys.exit()

    # Attain height and width information of image
    h, w = digits.shape[:2]

    # Create a HOG descriptor
    # cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)
    # print('Descriptor Size:', hog.getDescriptorSize()) # should be 324

    # Cutting/reshaping the image file into 5000 pieces (each 20pix by 20pix)
    # 500 samples for each digit (0-9), hence 5000 samples in total
    cells = [np.hsplit(row, w // 20) for row in np.vsplit(digits, h // 20)]
    cells = np.array(cells)
    cells = cells.reshape(-1, 20, 20)  # shape: (5000, 20, 20)

    # Normalize, compute HOG descriptors, and arrange them in a list
    # desc will thus have 5000 of (324, 1) ndarrays
    desc = []
    for img in cells:
        img = norm_digit(img)
        desc.append(hog.compute(img))

    # Train descriptors
    train_desc = np.array(desc)                          # shape:(5000, 324, 1)
    train_desc = train_desc.squeeze().astype(np.float32)  # shape:(5000, 324)

    # Make training labels: 500 0s, 500 1s, ..., 500 9s
    train_labels = np.repeat(np.arange(10), len(train_desc) / 10)

    # Train the SVM - Support Vector Machine
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)

    # Set C and Gamma: Attained after using svm.trainAuto()
    svm.setC(2.5)
    svm.setGamma(0.50625)

    svm.train(train_desc, cv2.ml.ROW_SAMPLE, train_labels)
    svm.save('digit_recog_norm.yml')

    # Use the trained SVM to classify user-written digits
    write_and_classify_digit(hog, svm)


if __name__ == '__main__':
    main()
