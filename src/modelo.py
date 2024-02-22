import os
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from IPython.display import Image, display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from datetime import datetime
import gradio as gr

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from ipywidgets import interact, widgets, Button
from IPython.display import display

class Modelo:
    # Classe python para...
    
    ##############################################################
    
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.train_images = None
        self.test_images = None
        self.model = None
        self.model_path = None
        self.history = None
              
    ##############################################################

    def preparar(self):
        # Esplicar o que faz essa função
        
        # List to store image filenames and corresponding artist labels
        image_filenames = []
        artist_labels = []

        # Iterate through image files in the directory
        for filename in os.listdir(self.config.images_path):
            if filename.endswith(tuple(self.config.image_extensions)):
                # Split the filename by '.' to get the artist name
                artist_name = filename.split('_')[0]
                if artist_name in self.dataset.selected_artists:
                    image_filenames.append(os.path.join(self.config.images_path, filename))
                    artist_labels.append(artist_name)

        # Encode artist labels as integers
        label_to_id = {artist: idx for idx, artist in enumerate(set(artist_labels))}
        artist_ids = [str(label_to_id[artist]) for artist in artist_labels]

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            image_filenames, artist_ids, test_size=0.15, random_state=42
        )
        
        self.train_images = X_train
        self.test_images = X_val
            
        # Imagens de treinamento com data augmentation para treinar a rede neural 
        train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,      # Rotate images up to 20 degrees
        width_shift_range=0.2,  # Shift width by 20% of the image width
        height_shift_range=0.2, # Shift height by 20% of the image height
        brightness_range=[0.8, 1.2], # Adjust brightness
        fill_mode='nearest')     # How to fill missing pixels after augmentation

        # Imagens de validação e teste sem data augmentation
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators for training and validation data
        self.train_generator = train_datagen.flow_from_dataframe(
            pd.DataFrame({'filename': X_train, 'label': y_train}),
            x_col='filename',
            y_col='label',
            target_size=(self.config.img_height, self.config.img_width),
            batch_size=self.config.batch_size,
            class_mode='categorical',  # Use categorical mode for multi-class classification
            shuffle=True
        )

        self.test_generator = test_datagen.flow_from_dataframe(
            pd.DataFrame({'filename': X_val, 'label': y_val}),
            x_col='filename',
            y_col='label',
            target_size=(self.config.img_height, self.config.img_width),
            batch_size=self.config.batch_size,
            class_mode='categorical',  # Use categorical mode for multi-class classification
            shuffle=False  # No need to shuffle validation data
        )
    
        #Definição da arquitetura da rede neural
        self.model = keras.models.Sequential([
           keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.config.img_height, self.config.img_width,3)),
           keras.layers.MaxPooling2D((2,2)),
           keras.layers.Conv2D(64, (3,3), activation='relu'),
           keras.layers.MaxPooling2D((2,2)),
           keras.layers.Conv2D(128, (3,3), activation='relu'),
           keras.layers.MaxPooling2D((2,2)),
           keras.layers.Conv2D(256, (3,3), activation='relu'),
           keras.layers.MaxPooling2D((2,2)),
           keras.layers.Flatten(),
           keras.layers.Dense(256, activation='relu'),
           keras.layers.Dense(len(self.dataset.selected_artists), activation='softmax')
        ])
            
        # Compilação do modelo
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ##############################################################
            
    def treinar(self):
        # Explicar o que faz essa função python        
        
        #history = self.model.fit(self.train_generator, epochs=self.config.epochs, validation_data=self.val_generator)

        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        #def lr_schedule(epoch):
        #     return 0.001 * 0.9 ** epoch  # Decrease learning rate by 10% each epoch
        # lr_scheduler = LearningRateScheduler(lr_schedule)

        # Treinando a rede neural
        self.history = self.model.fit(self.train_generator, epochs=self.config.epochs, callbacks=[early_stopping])

    ##############################################################
        
    def inferir(self):
        # Explicar o que faz esse método
        
        test_loss, test_acc = self.model.evaluate(self.test_generator)
        print('Test accuracy:', test_acc)
        
        count_images = 0
        
        self.y_pred = list()
        self.y_true = list()
        
        plt.figure(1, figsize=(24, 16))

        for image_path in self.test_images:
         
            if image_path.endswith(tuple(self.config.image_extensions)):
                
                count_images+=1
                split_path = image_path.replace('\\','/').split('/')
                split_path = split_path[3].split('_')
                label = split_path[0]
                self.y_true.append(label)

                display(Image(filename=image_path, width=300))

                img = image.load_img(image_path, target_size=(self.config.img_height, self.config.img_width))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = x.astype('float32') / 255.0

                # Make a prediction
                prediction = self.model.predict(x)

                predicted_class_index = np.argmax(prediction)
                predicted_class_probability = prediction[0][predicted_class_index]
                predicted_class = self.dataset.selected_artists[predicted_class_index]
                self.y_pred.append(predicted_class)

                print("Anotação:", label)
                print("Previsão:", predicted_class)
                print(f"Probabilidade: {predicted_class_probability:.4f}")
                print("\n")
               
    ##############################################################
                
    def gerar_metricas(self):
        # Explicar o que faz essa função python
        
        self.accuracy = skm.accuracy_score(self.y_true, self.y_pred)
        self.precision = skm.precision_score(self.y_true, self.y_pred, average='weighted', zero_division=1)
        self.recall = skm.recall_score(self.y_true, self.y_pred, average='weighted')
        self.f1score = skm.f1_score(self.y_true, self.y_pred, average='weighted')

        print("Acurácia: ", self.accuracy)
        print("Precisão: ", self.precision)
        print("Recall: ", self.recall)
        print("F1 Score: ", self.f1score)

        cnf_matrix = confusion_matrix(self.y_true, self.y_pred, labels=self.dataset.selected_artists)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.print_custom_cm(cnf_matrix, 
                              self.dataset.selected_artists,
                              normalize= False, 
                              title='Matriz real x predição')
        
    ##############################################################
    
    def salvar(self):
        # Explicar o que faz essa função python
        
        # Data e hora atual
        now = datetime.now()

        # Formato data-hora
        format_data = '%Y-%m-%dT%H%M'
        formatted_datetime = now.strftime(format_data)

        name_model = "model_{}_ep{}_bz{}_img{}_nc{}".format(formatted_datetime, self.config.epochs, self.config.batch_size, self.config.img_height, len(self.dataset.selected_artists))

        # Salvando o modelo
        self.model.save(os.path.join(self.config.models, name_model + '.h5'))
        print("Modelo salvo com o nome: ", name_model)

        # Method 1: Using json.dump() to write the dictionary to a JSON file
        with open('./src/models/labels/' + name_model + '.txt', 'w') as txt_file:
            for item in self.dataset.selected_artists:
                txt_file.write(str(item) + '\n')

    ##############################################################
    
    def select_model(self, selected_model):
        # Explicar o que faz essa função python 
        
        selected_model_path = os.path.join(self.config.models, selected_model)
        self.model_path = selected_model_path
        
        self.model = keras.models.load_model(self.model_path)
        
        # Specify the path to the JSON file you want to read
        labels_file_path = selected_model.split('.')
        labels_file_path = labels_file_path[0]
        labels_list = []

        # Method 1: Using json.load() to read the JSON file and parse it into a dictionary
        with open(f'./src/models/labels/{labels_file_path}.txt', 'r') as txt_file:
            for line in txt_file:
                labels_list.append(line.strip())
            self.dataset.selected_artists = labels_list
        
        return selected_model_path + ' carregado!!!' 
    
    ##############################################################  
         
    def escolher_modelo(self):
        # Explicar o que faz essa função python        

        # Get a list of model filenames from the folder
        model_filenames = [f for f in os.listdir(self.config.models) if f.endswith('.h5')]

        # Dropdown widget to select a model
        model_dropdown = widgets.Dropdown(
            options=model_filenames,
            description='Selecione um modelo:',
            disabled=False,
        )
        
        interact(self.select_model, selected_model=model_dropdown)

    ##############################################################
    
    def predict_author(self, input):
        # Explicar aqui o que faz essa função
        
        if input is None:
            return 'Please upload an image'
        
        x = image.img_to_array(input)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32') / 255.0
        
        prediction = self.model.predict(x)
        
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = self.dataset.selected_artists[predicted_class_index]
        class_probabilities = prediction[0] 

        confidences = {self.dataset.selected_artists[i]: float(class_probabilities[i]) for i in range(len(self.dataset.selected_artists))}
        
        return confidences
    
    ##############################################################

    
    def launch_app(self):
        # Explicar aqui o que faz essa função
        
        description = 'Este é um aplicativo de demonstração que tenta reconhecer autores de pinturas. Depois de enviar uma foto de uma pintura, o aplicativo exibirá o autor previsto junto com a probabilidade de previsão dos autores que o modelo tem conhecimento. O aplicativo usa uma rede neural convolucional como modelo base cujo classificador foi treinado com um conjunto limitado de pinturas. Dadas as limitações do conjunto de dados, o modelo reconhece apenas pinturas de ' + str(len(self.dataset.selected_artists)) +  ' artistas:'
        
        for artist in self.dataset.selected_artists:
            description+=f'\n- {artist}'

        demo = gr.Interface(
            title='Prevendo autores de pintura',
            description=description,
            fn=self.predict_author, 
            inputs=gr.Image(shape=(self.config.img_height, self.config.img_width)), 
            outputs=gr.Label(num_top_classes=len(self.dataset.selected_artists)),
            examples=['./image_example.png']
            )

        demo.launch()
        
    ##############################################################
   
    def print_custom_cm(self, cm, classes,
                          normalize=False,
                          title='Matriz de confusão',
                          cmap=plt.cm.Blues):

        """
        Esta função imprime e plota a matriz de confusão.
        A normalização pode ser aplicada definindo `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Matriz de confusão normalizada")
        else:
            print('Matriz de confusão sem normalização')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Label real')
        plt.xlabel('Label predito')
