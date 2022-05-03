# used to change filepaths
import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

labels = pd.read_csv("datasets/labels.csv", index_col=0)


def get_image(row_id, root="datasets/"):
    """
    Converts an image number into the file path where the image is located,
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)


def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to grayscale
    gray_image = rgb2gray(img)
    # get HOG features from grayscale image
    hog_features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack([color_features, hog_features])
    return flat_features


def create_feature_matrix(label_dataframe):
    features_list = []

    for img_id in label_dataframe.index:
        # load image
        img = get_image(img_id)
        # get features
        image_features = create_features(img)
        features_list.append(image_features)

    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix


feature_matrix = create_feature_matrix(labels)

X_train, X_test, y_train, y_test = train_test_split(feature_matrix,
                                                    labels.genus.values,
                                                    test_size=.1,
                                                    random_state=1234123)

# look at the distribution of labels in the train set
pd.Series(y_train).value_counts()
print(X_train.shape)

print('Training features matrix shape is: ', X_train.shape)

# define standard scaler
ss = StandardScaler()

# fit the scaler and transform the training features
train_stand = ss.fit_transform(X_train)

# use fit_transform to run PCA on our standardized training features
test_stand = ss.transform(X_test)

# look at the new shape of the standardized feature matrices
# print('Standardized training features matrix shape is: ', train_stand.shape)
# print('Standardized test features matrix shape is: ', test_stand.shape)
svm = None
pca = None

def bee_predict(file_path='/home/pi/Desktop/5642FinalProj/predict/datasets/520.jpg'):
    """ 100*100*3 -> predict bee """
    global svm
    global pca

    # 1. get input img
    #filename = "{}.jpg".format(img_id)
    #file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    # 100 * 100 * 3 input
    img_info = np.array(img)

    # 4. convert 3D img info into linear array
    flat_feature = create_features(img_info)
    flat_feature = flat_feature.reshape(1, -1)

    ts = ss.transform(flat_feature)
    # print(ts.shape)

    # 5. PCA process
    pca = PCA(n_components=450)

    # use fit_transform on our standardized training features
    X_train = pca.fit_transform(train_stand)

    # use transform on our standardized test features
    ts = pca.transform(ts)

    # look at the new shape of the transformed matrices
    #print('Training features matrix is: ', X_train.shape)
    #print('Test features matrix is: ', ts.shape)
    # 6. define support vector classifier
    svm = SVC(kernel='linear', probability=True, random_state=42)

    # fit model
    svm.fit(X_train, y_train)

    # generate predictions
    y_pred = svm.predict(ts)
    print('init')
    # calculate accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print('Model accuracy is: ', accuracy)
    #if y_pred:
        #print("bumble bee detected")
    #else:
        #print("honey bee detected")
    # print(y_pred)
bee_predict()

def quick_pred(fp, pca=pca, svm=svm):

    img = Image.open(fp)
    # 100 * 100 * 3 input
    img_info = np.array(img)

    flat_feature = create_features(img_info)
    flat_feature = flat_feature.reshape(1, -1)

    ts = ss.transform(flat_feature)

    # use transform on our standardized test features
    ts = pca.transform(ts)

    # generate predictions
    y_pred = svm.predict(ts)


    if y_pred:
        print("bumble bee detected")
    else:
        print("honey bee detected")
        

# bee_predict()