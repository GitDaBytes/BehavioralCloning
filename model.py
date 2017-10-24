import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import rotate, shift, zoom
from random import randint

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization


# base folder where training data is located
base_dir = "./training_data/"

# sub folders to use for training
tracks_to_process = ["Track1_4", "Track1_5R"]#, "Track2"]

def add_flipped_images(images, measurements):
    
    for idx in range(len(images)):
        image_flipped = np.fliplr(images[idx])
        measurement_flipped = -measurements[idx]
        images.append(image_flipped)
        measurements.append(measurement_flipped)

def Add_Shadow(image):
    
    r = randint(1,3)
    
    im_width = image.shape[1]
    im_height = image.shape[0]
    
    shadow_width = int(im_width * .4)
    shadow_height = int(im_height * .4)
    
    x=0
    y=0
    
    if r == 1: # left of image
        x = 0
        y = 0
        shadow_height = im_height
        
    elif r == 2: # right of image
        x = im_width - shadow_width
        y = 0
        shadow_height = im_height
        
    elif r == 3: # bottom of image
        x = 0
        y = im_height - shadow_height
        shadow_width = im_width
        
    elif r == 4: # top of image
        x = 0
        y = 0
        shadow_width = im_width
    
    overlay = image.copy()
    output = image.copy()
    
    cv2.rectangle(overlay, (x, y), (x+shadow_width, y+shadow_height), (20, 20, 20), -1)
    alpha = 0.7;

    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

def read_lines_from_csv(filename):
    lines = []
    
    csv_filename = base_dir + track_number + '/driving_log.csv'
    
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    return lines

def trim_repeated_lines(lines, max_repeat_count=5):
    
    repeated_count = 0
    
    last_measurement = 0.0
    
    trimmed_lines = []
    
    for idx in range(len(lines)):
        
        line = lines[idx]
        
        # take current measurement from line
        measurement = float(line[3])
        
        # add line on to the trimmed lines list
        trimmed_lines.append(line)
        
        
        # keep track of repeated count
        if (idx > 0) and (measurement==0.0) and (measurement == last_measurement):
            repeated_count+=1
        
        # if we have x repeats in a row, truncate them to x-1 then start afresh
        if repeated_count >= max_repeat_count:
            trimmed_lines = trimmed_lines[:-1 * (max_repeat_count-1)]
            
            repeated_count=0 # reset trimmed count
        
        # record last measurement for next loop round
        last_measurement = measurement
        
        
    return trimmed_lines

def rand_jitter(temp):

    temp = shift(temp, shift=(np.random.randint(-20,20,1), 0, 0))

    return temp


# change brightness of image by random amount
def Augment_Brightness(image):

    #image = cv2.imread('test.jpg') #load rgb image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #convert it to hsv

    h, s, v = cv2.split(hsv)
    
    tweak_amount = randint(-75, 75)
    
    newV = np.clip(np.int32(v) + tweak_amount, 0, 255)
    
    v = np.uint8(newV)
    
    final_hsv = cv2.merge((h, s, v))

    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    
    return image

def basic_model(model):

    model.add(Flatten())
    
    return model

def LeNet_model(model):
    
    my_init = TruncatedNormal(mean = 0, stddev = 1e-01)
    
    #kernel_initializer='truncated_normal', bias_initializer='zeros'
    
    model.add(Conv2D(6,(5,5),activation="relu", kernel_initializer=my_init, bias_initializer='zeros'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(6,(5,5),activation="relu", kernel_initializer=my_init, bias_initializer='zeros'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu', kernel_initializer=my_init, bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu', kernel_initializer=my_init, bias_initializer='zeros'))
    
    return model

# tried to replicate the model at:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# https://arxiv.org/pdf/1704.07911.pdf

def NVIDIA_PilotNet_model(model):
    
    dropout_cnn = .2
    dropout = .5
    
 
    model.add(Conv2D(24,(5,5),strides=(2,2),activation="elu", kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(36,(5,5),strides=(2,2),activation="elu", kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(48,(5,5),strides=(2,2),activation="elu", kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64,(3,3),strides=(1,1),activation="elu", kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64,(3,3),strides=(1,1),activation="elu", kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='elu', kernel_initializer='he_normal', bias_initializer='zeros'))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='elu', kernel_initializer='he_normal', bias_initializer='zeros'))
    
    model.add(BatchNormalization())
    
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='elu', kernel_initializer='he_normal', bias_initializer='zeros'))
    
    return model


def main(argv):
    # load the images for processing


    images = []
    measurements = []
        
    for track_number in tracks_to_process:
        
        csv_filename = base_dir + track_number + '/driving_log.csv'
        
        lines = read_lines_from_csv(csv_filename)
        
        print(len(lines))
        lines = trim_repeated_lines(lines, 3)
        print(len(lines))
        
        drawn = 0
        
        for line in lines:
            for idx in range(0, 3): # loop through the center, left and right images on each line of the file
                source_path = line[idx]
                filename = source_path.split('\\')[-1]
                current_path = base_dir + track_number + '/IMG/' + filename

                image = cv2.imread(current_path)

                if not image is None:

                    #note drive.py reads image in RGB format, imread reads in RGB, 
                    #so we will flip it here
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if drawn == 0:
                        plt.imshow(image)
                        plt.show()

                    # add random jitter in y axis to some images
                    if np.random.random() > .9:
                        image = rand_jitter(image)

                    # Add random brightness change to some images
                    if np.random.random() > .6:
                        image = Augment_Brightness(image)

                    # Add random shadow effect to some images
                    if np.random.random() > .8:
                        image = Add_Shadow(image)

                    # plot first image so we can see all going ok.
                    if drawn == 0:
                        imgplot = plt.imshow(image)
                        plt.show()
                        
                        drawn = 1

                    # now we are done editing the image convert to YUV as this is what we will train the model with
                    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

                    # add the image to the list of images to use
                    images.append(yuv)                    

                    # read measurement for this picture
                    measurement = float(line[3])

                    steering_adjustment = 0.2 # constant we add / subtract for the left and right camera images
                    if idx==1: # left image
                        measurement += steering_adjustment
                    elif idx == 2: # right image
                        measurement -= steering_adjustment

                    # add the measurement to the list for processing
                    measurements.append(measurement)
                    

        # run through all images and generate a new image flipped in y, 
        # add flipped steering commands for those too! We are trying to remove bias 
        # for the mostly left turn track 1
        add_flipped_images(images, measurements)


    images, measurements = shuffle(images, measurements)

    X_train = np.array(images)
    y_train = np.array(measurements)

    print(X_train.shape)


    print("IMAGES ARE LOADED... STARTING TRAINING...)

    # Training parameters
    batch_size = 256
    epochs = 15
    learning_rate = 0.0005

    early_stop = 0 # not using

    # Build the model
    model = Sequential()

    # add cropping layer
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))

    # normalize the data centered on 0.5
    model.add(Lambda(lambda x: x / 255.0))

    # use PilotNet
    model_to_use = 3

    if model_to_use == 1:
        print("using basic model")
        model = basic_model(model)
        
    elif model_to_use == 2:
        print("using LeNet")
        model = LeNet_model(model)

    elif model_to_use == 3:
        print("using NVIDIA PilotNet")
        model = NVIDIA_PilotNet_model(model)

    # add the output layer on end (the steering angle)
    model.add(Dense(1))

    # setup learning rate with Adam Optimizer
    adam = Adam(lr=learning_rate)

    # configure early stopping so we are not waiting longer than needed and also reduces overfitting

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(loss='mse', optimizer=adam)

    if early_stop > 0:
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])
    else:
        model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=batch_size, epochs=epochs)

    model.save('model.h5') # save the model out

    print("Training complete!")

    

if __name__ == "__main__":
    main(sys.argv)
