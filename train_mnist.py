import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from scratch_grad import Variable
import os

# Ensure the results_graphs directory exists
results_dir = 'results_graphs/acc_loss'
os.makedirs(results_dir, exist_ok=True)
comp_graphs_dir = 'results_graphs/computation_graphs'
os.makedirs(comp_graphs_dir, exist_ok=True)


from graphviz import Digraph

def visualize_computation_graph(final_var, filename):
    dot = Digraph(format='png')
    
    def add_nodes(var):
        if var not in seen:
            # Corrected from var.name to var._name
            dot.node(str(id(var)), label=var._name)  
            seen.add(var)
        if hasattr(var, 'grad_fn'):
            for child in var.grad_fn.inputs:
                if child not in seen:
                    add_nodes(child)
                dot.edge(str(id(child)), str(id(var)))
    
    seen = set()
    add_nodes(final_var)
    dot.render(filename)



if __name__ == '__main__':
    # Random seed
    np.random.seed(42)
    random.seed(42)

    # Training parameters
    num_epochs = 30
    learning_rate = 8e-2


    # Network
    # TODO creer votre reseau
    class MnistNeuralNet():
        def __init__(self):
            #Initialisation des paramètres
            self.w_1_data = np.random.randn(784, 25).astype(np.float32)
            self.b_1_data = np.random.randn(1, 25).astype(np.float32)
            self.w_2_data = np.random.randn(25, 25).astype(np.float32)
            self.b_2_data = np.random.randn(1, 25).astype(np.float32)
            self.w_3_data = np.random.randn(25, 10).astype(np.float32)
            self.b_3_data = np.random.randn(1, 10).astype(np.float32)

            #Nommage des variables gradients comme dans test_net_forward
            self.sg_w_1 = Variable(self.w_1_data, name='w1')
            self.sg_b_1 = Variable(self.b_1_data, name='b1')
            self.sg_w_2 = Variable(self.w_2_data, name='w2')
            self.sg_b_2 = Variable(self.b_2_data, name='b2')
            self.sg_w_3 = Variable(self.w_3_data, name='w3')
            self.sg_b_3 = Variable(self.b_3_data, name='b3')

            self.gradient_list = [self.sg_w_1, self.sg_b_1, self.sg_w_2, self.sg_b_2, self.sg_w_3, self.sg_b_3]

        def reset_gradients(self):
            for gradient in self.gradient_list:
                gradient.zero_grad()

        def update_weights(self):
            self.sg_w_1.data = self.sg_w_1.data - learning_rate*self.sg_w_1.grad 
            self.sg_w_2.data = self.sg_w_2.data - learning_rate*self.sg_w_2.grad 
            self.sg_w_3.data = self.sg_w_3.data - learning_rate*self.sg_w_3.grad 
            self.sg_b_1.data -= learning_rate * self.sg_b_1.grad
            self.sg_b_2.data -= learning_rate * self.sg_b_2.grad
            self.sg_b_3.data -= learning_rate * self.sg_b_3.grad

        def pass_data_through(self, x):
            sg_z_1 = x.__matmul__(self.sg_w_1).__add__(self.sg_b_1).relu()
            sg_z_2 = (sg_z_1.__matmul__(self.sg_w_2).__add__(self.sg_b_2)).relu()
            sg_z_3 = (sg_z_2.__matmul__(self.sg_w_3).__add__(self.sg_b_3 ))

            return sg_z_3
        
    mnist_neuralnet = MnistNeuralNet()


    # Dataset
    mnist = MNIST(root='data/', train=True, download=True)
    mnist.transform = ToTensor()

    mnist_test = MNIST(root='data/', train=False, download=True)
    mnist_test.transform = ToTensor()

    # Only take a small subset of MNIST
    mnist = Subset(mnist, range(len(mnist) // 16))
    mnist_test = Subset(mnist_test, range(32))

    # Dataloaders
    train_loader = DataLoader(mnist, batch_size=1, shuffle=True)
    val_loader = DataLoader(mnist_test, batch_size=1)

    # Logging
    train_loss_by_epoch = []
    train_acc_by_epoch = []
    val_loss_by_epoch = []
    val_acc_by_epoch = []

    epochs = list(range(num_epochs))
    for epoch in epochs:
        train_predictions = []
        train_losses = []
        # Training loop
        for x, y in tqdm.tqdm(train_loader):
            # Put the data into Variables
            x = x.numpy().reshape((1, 784))
            x = Variable(np.array(x), name='x')
            y = y.numpy().reshape((1, 1))
            y = Variable(np.array(y), name='y')

            # Pass the data through the network
            # TODO passer les valeurs dans le reseau
            z_3 = mnist_neuralnet.pass_data_through(x)

            # Compute the loss
            # TODO calculer la fonction de perte
            loss = z_3.nll(y)

            # Apply backprop
            # TODO appliquer la backprop sur la perte
            loss.backward()
# Placeholder for visualization function call
            # Assuming 'loss' is the output variable you want to visualize
            graph_filename = os.path.join(comp_graphs_dir, f'computation_graph_epoch_{epoch}.png')
            visualize_computation_graph(loss, graph_filename)  # Uncomment and implement this
            # Update the weights
            # TODO mettre-a-jour les poids avec un learning rate de `learning_rate`
            mnist_neuralnet.update_weights()

            # Reset gradients
            # TODO mettre les gradients a zero avec `variable.zero_grad()`
            mnist_neuralnet.reset_gradients()

            # Logging
            train_losses.append(loss.data)
            train_predictions.append(np.argmax(z_3.data, axis=1) == y.data)

        # Validation loop
        val_results = []
        val_losses = []
        for x, y in val_loader:
            # Put the data into Variables
            x = x.numpy().reshape((1, 784))
            x = Variable(np.array(x), name='x')
            y = y.numpy().reshape((1, 1))
            y = Variable(np.array(y), name='y')

            # Pass the data through the network
            # TODO passer les valeurs dans le reseau
            z_3 = mnist_neuralnet.pass_data_through(x)

            # Compute the loss
            # TODO calculer la fonction de perte
            loss = z_3.nll(y)

            # Logging
            val_losses.append(loss.data)
            val_results.append(np.argmax(z_3.data, axis=1) == y.data)

        # Compute epoch statistics
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_predictions)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_results)

        # Show progress
        print(f'Epoch {epoch}')
        print(f'\tTrain:\t\tLoss {train_loss},\tAcc {train_acc}')
        print(f'\tValidation:\tLoss {val_loss},\tAcc {val_acc}')

        # Logging
        train_loss_by_epoch.append(train_loss)
        train_acc_by_epoch.append(train_acc)
        val_loss_by_epoch.append(val_loss)
        val_acc_by_epoch.append(val_acc)

    # At the end of each epoch, plot and save graphs
        _, axes = plt.subplots(2, 1, sharex=True)
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs[:epoch + 1], train_acc_by_epoch[:epoch + 1], label='Train')
        axes[0].plot(epochs[:epoch + 1], val_acc_by_epoch[:epoch + 1], label='Validation')
        axes[0].legend()

        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs[:epoch + 1], train_loss_by_epoch[:epoch + 1], label='Train')
        axes[1].plot(epochs[:epoch + 1], val_loss_by_epoch[:epoch + 1], label='Validation')

        axes[1].set_xlabel('Epochs')
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the figure in the results_graphs folder
        plt.savefig(f'{results_dir}/epoch_{epoch}_graphs.png')

        # Optionally, display the plot inline (comment out if running in a non-interactive environment)
        plt.show()

        # Clear the current figure's data to prevent overlap with next epoch's data
        plt.clf()