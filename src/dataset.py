import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

class Dataset:
    # Classe python para 
    
    ##############################################################
    
    def __init__(self, config):
        self.config = config
        self.artists = None
        self.selected_artists = None
            
    def artistas_disponiveis(self):
        # Explicar aqui o que faz a função 
        
        artists = set()
        for filename in os.listdir(self.config.images_path):
            if filename.endswith(tuple(self.config.image_extensions)):
                artist = filename.split('_')[0]  # Assumes the label is the first part before underscore
                artists.add(artist)
        
        self.artists = list(artists)
        return sorted(list(artists))
        
    ###########################################################
    
    def visualizar_numero_artistas_e_obras(self):
        # Explicar aqui o que faz a função
        
        self.artistas_disponiveis()
        
        print('O número de artistas disponíveis para treinar um modelo é:', self.config.num_classes)
        print('\nO número de imagens por artista é:')

        # Initialize a dictionary to store counts
        artist_counts = {artist: 0 for artist in self.artists}

        # Loop through the image files in the directory
        for filename in os.listdir(self.config.images_path):
            if filename.endswith(tuple(self.config.image_extensions)):
                # Split the filename by '_' to get the artist name
                parts = filename.split('_')
                if len(parts) > 1:
                    artist_name = parts[0]
                    # Check if the artist is in the list of artists to consider
                    if artist_name in self.artists:
                        artist_counts[artist_name] += 1

        # Convert the artist counts to a Pandas DataFrame
        df = pd.DataFrame(list(artist_counts.items()), columns=['Artist', 'Count'])

        # Create a bar plot using Matplotlib
        plt.figure(figsize=(7, 5))
        plt.barh(df['Artist'], df['Count'])
        plt.xlabel('Número de imagens')
        plt.ylabel('Artista')
        plt.title('Número de imagens disponíveis por artista')
        plt.gca().invert_yaxis()  # Invert the y-axis for better visualization
        plt.show()
        
    ###########################################################
    
    def visualizar_imagens_random(self):
        # Explicar o que faz essa função
            
        # Create a 3x4 grid for displaying images
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))

        # List of image files in the directory
        image_files = [file for file in os.listdir(self.config.images_path) if file.endswith(tuple(self.config.image_extensions))]

        # Randomly select 12 images for display
        sample_images = random.sample(image_files, 12)

        # Iterate through the axes and sample images
        for ax, image_filename in zip(axes.flatten(), sample_images):
            artist_name = image_filename.split('_')[0]
            artist_name = artist_name.replace('.jpg', '')  # Remove the file extension
            img = mpimg.imread(os.path.join(self.config.images_path, image_filename))
            ax.imshow(img)
            ax.set_title(artist_name, fontsize=10)  # Set artist name as title above the image
            ax.axis('off')  # Turn off axis labels

        # Adjust spacing between subplots
        plt.tight_layout()
        plt.show()
     
        
    ###########################################################
    
    def selecionar_artistas(self):
        # Explicar o que faz essa função
        
        selected_artists = []
        print("Selecione os artistas que serão usados para treinar o modelo:")
        for i, artist in enumerate(self.artists, start=1):
            selection = input(f"[ ] {artist} (s/n): ").strip().lower()
            if selection == 's' or selection == 'S':
                selected_artists.append(artist)
                
        self.selected_artists = selected_artists
        
        print("\nOs artistas selecionados foram:\n")
        
        for artist in self.selected_artists:
            print(f'- {artist}')
    
    ###########################################################
  