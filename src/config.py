from src.dataset import Dataset

class Config:
    
    def __init__(self):
        
        self.dataset = Dataset(self)
        
        # Definição do caminho onde estão salvas as imagens de treinamento, validação e teste 
        self.images_path = './src/images'
        self.models = './src/models'
        self.inference_images = './src/images/inference'

        # Definição das configurações das imagens 
        self.batch_size = 32
        self.img_height = 224 
        self.img_width = 224
        
        self.image_extensions = ['.jpg', '.jpeg', '.png']
        
        # Definição do número de autores (depende do número de pastas relacionado aos autores)
        self.num_classes = len(self.dataset.artistas_disponiveis())
        
        # Definição do número de epocas para treinar a rede neural
        self.epochs = 35