from os import name
from generate_data import GenerateData
from model import ReportModel, ModelTrainer
import torch
import logging

logger = logging.getLogger()


class TrainModel:
    def __init__(self):
        self.data = GenerateData()

    def train(self, model_name="updated_model", epochs=10, batch_size=32, lr=0.001, print_every=32):
        input_data, training_data = self.data.generate_training_data("plots_text.pickle")
        # set the vocab size to be the length of the number of unique phrases within the training data. Pulled from the token_word mapping
        self.model = ReportModel(len(self.data.token_word))
        self.model_trainer = ModelTrainer(self.data.input_int, self.data.target_int)
        print(torch.cuda.get_device_name(0))
        self.model_trainer.train(self.model, epochs=epochs, batch_size=batch_size, lr=lr, print_every=print_every)
        self.save_model(f"weights/{model_name}.pth")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved successfully to {path}")


if __name__ == "__main__":
    train = TrainModel()
    train.train(model_name="attempt1", epochs=1, print_every=256)