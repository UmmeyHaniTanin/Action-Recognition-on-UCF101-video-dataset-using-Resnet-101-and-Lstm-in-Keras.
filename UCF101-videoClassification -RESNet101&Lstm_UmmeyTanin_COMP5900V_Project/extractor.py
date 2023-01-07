from keras.preprocessing import image
from keras.applications.resnet import ResNet101, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np 

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load  saved
        weights from  eaarlier training."""

        self.weights = weights  

        if weights is None:
            # Get model with pretrained weights.
            base_model = ResNet101(
                weights='imagenet',
                include_top=True
            )

            # extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so to get features not predictions.
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
