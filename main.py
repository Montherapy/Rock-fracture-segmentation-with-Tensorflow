# coding: utf-8

from network import Net
from trainer import Trainer
import numpy as np


network_type = 5
first_filter = 64
class_weight = [0.7, 0.3]
norm_type = "batch_norm"
epochs = 15
learning_rate = 0.001
batch_size = 5
keep_prob = 0.8
restore_path = "logs/model%d" %network_type
save_path = None


def train():
	x_tr = np.load('./x_train.npy')
	y_tr = np.load('./y_train.npy')
	network = Net(network_type, 224, first_filter, class_weights=class_weight, norm_kwargs={'norm_type': norm_type})
	trainer = Trainer(network, x_tr, y_tr)

	trainer.training(epochs, learning_rate, batch_size, keep_prob, restore_path, save_path)

def test():
	x_test = np.load("./x_test.npy")
	y_test = np.load("./y_test.npy")
	network = Net(network_type, 1024, first_filter, class_weights=class_weight, norm_kwargs={'norm_type': norm_type})
	trainer = Trainer(network)
	trainer.evaluate(x_test, y_test, restore_path)
	softmax, prediction = trainer.get_output(x_test, restore_path)
	np.save("softmax_output.npy", softmax)

if __name__ == "__main__":
	test()
