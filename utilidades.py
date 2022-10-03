import os
import cv2
import imutils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications import densenet
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from random import shuffle,seed
seed(29071999)

#######################################################################################
################################ CARGAR DATOS #########################################
#######################################################################################


def resize(img): 
  img=smart_resize(img,(224,224))
  return img

def resize_all(set):
  ret = []
  for img in set:
    ret.append(resize(img))
  return np.asarray(ret)

def leer_datos():
    
    # Cargamos las imágenes 
    directory_training = './Training/'
    directory_testing = './Testing/'

    mapping={'no_tumor':0, 'pituitary_tumor':1, 'meningioma_tumor':2, 'glioma_tumor':3}

    dataset=[]
    count=0
    for file in os.listdir(directory_training):
        path=os.path.join(directory_training,file)
        for im in os.listdir(path):
            image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(150,150))
            image=img_to_array(image)
            image=image/255.0
            dataset.append([image,count])     
        count=count+1

    count=0
    testset=[]
    for file in os.listdir(directory_testing):
        path=os.path.join(directory_testing,file)
        for im in os.listdir(path):
            image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(150,150))
            image=img_to_array(image)
            image=image/255.0
            testset.append([image,count])         
        count=count+1
        
    
    #Separamos en conjuntos
    training_set,training_labels = zip(*dataset)
    test_set,test_labels         = zip(*testset)
        
    # Desordenamos los conjuntos
    ind_train = list(range(len(training_set)))
    ind_test  = list(range(len(test_set)))
    shuffle(ind_train)
    shuffle(ind_test)
    
    training_labels_visu = training_labels
    test_labels_visu     = test_labels
    
    training_labels = to_categorical(training_labels,4)
    test_labels     = to_categorical(test_labels,4)
        
    training_set    = np.asarray(training_set)[ind_train]
    training_labels = np.asarray(training_labels)[ind_train]
    test_set        = np.asarray(test_set)[ind_test]    
    test_labels     = np.asarray(test_labels)[ind_test]
    
    #training_set  = resize_all(training_set)
    #test_set      = resize_all(test_set)
       
    return training_set, training_labels, test_set, test_labels, training_labels_visu, test_labels_visu


#######################################################################################
########################CARGAR DATOS CON OPENCV########################################
#######################################################################################

def carga_datos_openCV(dir_path):
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} imágenes cargadas de la ruta {dir_path}.')
    return X, y, labels


#######################################################################################
################################# PREPROCESADO ########################################
#######################################################################################


def recortar_imagen(set_name, add_pixels_value=0):
    """
    Encuentra los extremos en la imagen y recorta en regiones rectangulares
    """
    set_new = []
    for img in set_name:
        # Pasamos a escala de grises y aplicamos filtros gaussianos
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Aplicamos threshold a la imagen, hacemos luego una serie de
        # erosiones y dilataciones para quitar cualquier región con ruido 
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Buscamos contornos y nos quedamos con el grande
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # fBuscamos extremos 
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)


def preprocess_imgs(set_name, img_size):
    """
    Redimensionamos y aplicamos el preprocesado de DenseNet
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(densenet.preprocess_input(img))
    return np.array(set_new)


#######################################################################################
################################## VISUALIZAR #########################################
#######################################################################################

def normalizar(mat):
      return (mat-mat.min())/(mat-mat.min()).max()
        
def title_from_label(label):
    
    if label == 0:
        title = 'no_tumor'
    elif label == 1:
        title = 'pituitary_tumor'
    elif label == 2:
        title = 'meningioma_tumor'
    else:
        title = 'glioma_tumor'
    
    return title

def show_image(img, label, figsize = (7,7) ):
    '''
        Imprime imagen por pantalla
        - img:     imagen.
        - label:   etiqueta 
        - figsize: (width, height)
    '''

    im  = normalizar(img) 
    
    plt.figure(figsize = figsize)
    plt.imshow(im, interpolation = None, cmap = 'gray')
    plt.xticks([]), plt.yticks([])


    title = title_from_label(label)
    
    plt.title(title)
        
    plt.show()
    

def show_n_images(vim, vlab, ncols = 2, tam = (5, 5)):
    
    '''
        Muestra una sucesión de imágenes en la misma ventana, eventualmente con sus títulos.
        - vim:    sucesión de imágenes a mostrar.
        - vlab:   vector de etiquetas
        - ncols:  número de columnas del multiplot.
        - tam = (width, height): 
    '''


    nrows = len(vim) // ncols + (0 if len(vim) % ncols == 0 else 1)
    plt.figure(figsize = tam)

    for i in range(len(vim)):
        plt.subplot(nrows, ncols, i + 1)
        show_image(vim[i], vlab[i],  tam)

    plt.show()
    
    
def show_some_images(number_images_per_row,training_set,training_labels): 
    '''
        Muestra un conjunto de imágenes en un marco cuadradao:
        - number_images_per_row: Lado del cuadrado
        - training_set:          Conjunto de imagenes
        - training_labels:       Conjunto de etiquetas correspondiente
    '''
    
    result = training_labels
    fig = plt.figure()
    for i in range(number_images_per_row*number_images_per_row):
        plt.subplot(number_images_per_row,number_images_per_row,i+1).set_title(title_from_label(training_labels[i]))
        plt.imshow(training_set[i].squeeze(), cmap='gray',vmin=0,vmax=1)

    fig.set_size_inches(np.array(fig.get_size_inches()) * number_images_per_row)
    plt.show()
    
    
    
    
#######################################################################################
#######################################################################################
#######################################################################################


def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()
    
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.show()
