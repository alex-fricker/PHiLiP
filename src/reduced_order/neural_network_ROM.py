import numpy as np
import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from scipy import linalg
from sklearn import preprocessing
import matplotlib.pyplot as plt


class POD:
    def __init__(self, snapshots, parameters, n_modes=None):
        self.snapshots = snapshots
        self.parameters = parameters
        self.n_modes = n_modes  # Leave as None to keep all singular values, or enter an positive
                                # int for a lower rank POD approximation
        self.ref_state = None
        self.POD_basis = None
        self.POD_coefficients = None

    def scale_parameters(self):
        scaler = preprocessing.MinMaxScaler()
        self.parameters = scaler.fit_transform(self.parameters.T)
        self.parameters = self.parameters.T

    def transform(self):
        self.scale_parameters()
        self.ref_state = np.mean(self.snapshots, axis=1)
        centered = self.snapshots - self.ref_state
        U, S, V = linalg.svd(centered, full_matrices=False)
        V = V.T  # To make computations clear, linalg.svd already returns transpose(V)
        if self.n_modes is not None:
            U, S, V = U[:, :self.n_modes], S[:self.n_modes], V[:, :self.n_modes]
        else:
            self.n_modes = len(S)

        self.POD_coefficients = np.diag(S) @ V.T
        self.POD_basis = U

    def inverse(self, coeffs):
        return self.POD_basis @ coeffs + self.ref_state


class SnapshotDataset(Dataset):
    def __init__(self, parameters, targets, device):
        self.parameters = parameters
        self.targets = targets
        self.device = device

    def __len__(self):
        return np.size(self.parameters, axis=0)

    def __getitem__(self, item):
        inputs = tc.from_numpy(self.parameters[: item])
        targets = tc.from_numpy(self.targets[: item])
        return inputs.to(self.device), targets.to(self.device)


class NeuralNetwork(nn.Module):
    def __init__(self, outSize, arch=1):
        super(NeuralNetwork, self).__init__()
        self.outSize = outSize
        self.arch = arch
        self.arch0 = nn.Sequential(
            nn.Linear(2, 50),
            nn.Sigmoid(),
            nn.Linear(50, self.outSize)
        )
        self.arch1 = nn.Sequential(
            nn.Linear(2, 25),
            nn.PReLU(),
            nn.Linear(25, 50),
            nn.PReLU(),
            nn.Linear(50, 100),
            nn.PReLU(),
            nn.Linear(100, 50),
            nn.PReLU(),
            nn.Linear(50, self.outSize)
        )
        self.arch2 = nn.Sequential(
            nn.Linear(2, 16),
            nn.PReLU(),
            nn.Linear(16, 32),
            nn.PReLU(),
            nn.Linear(32, 32),
            nn.PReLU(),
            nn.Linear(32, self.outSize)
        )
        self.arch3 = nn.Sequential(
            nn.Linear(2, 16),
            nn.PReLU(),
            nn.Linear(16, 16),
            nn.PReLU(),
            nn.Linear(16, 32),
            nn.PReLU(),
            nn.Linear(32, self.outSize)
        )
        self.arch4 = nn.Sequential(
            nn.Linear(2, 16),
            nn.PReLU(),
            nn.Linear(16, 16),
            nn.PReLU(),
            nn.Linear(16, 16),
            nn.PReLU(),
            nn.Linear(16, self.outSize)
        )
        self.arch5 = nn.Sequential(
            nn.Linear(2, 32),
            nn.PReLU(),
            nn.Linear(32, 32),
            nn.PReLU(),
            nn.Linear(32, self.outSize)
        )

    def forward(self, x):
        if self.arch == 1:
            x = self.arch1(x)
        elif self.arch == 2:
            x = self.arch2(x)
        elif self.arch == 3:
            x = self.arch3(x)
        elif self.arch == 4:
            x = self.arch4(x)
        elif self.arch == 5:
            x = self.arch5(x)
        return x


class PODNeuralNetworkROM:
    def __init__(self, POD, device):
        self.POD = POD
        self.device = device
        self.loss_function = None
        self.network = None
        self.optimizer = None
        tc.manual_seed(0)
    
    def initalize_network(self, architecture, learning_rate, weight_decay):
        self.loss_function = nn.MSELoss()
        self.network = NeuralNetwork(outSize=self.POD.n_modes, arch=architecture)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)

    def build_network(self, epochs, batch_size, print_plots):
        print('Training neural network.')
        self.network.apply(self.reset_weights())
        dataset = SnapshotDataset(parameters=
                                  self.POD.parameters,
                                  targets=self.POD.POD_coefficients,
                                  device=self.device)
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        training_losses = []

        for epoch in range(1, epochs + 1):
            training_losses = self.train(train_loader=dataset_loader, training_losses=training_losses)

        if print_plots:
            plt.plot(np.arange(0, epochs),
                     training_losses,
                     label=f'Training, final iteration loss: {training_losses[-1]:.4f}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Full dataset training losses')
            plt.legend(loc='upper right')
            plt.show()
        print('Done training neural network.')

    def evaluate_network(self, evaluation_points):
        self.network.eval()
        n_points = np.size(evaluation_points, axis=1)
        rom_solutions = np.zeros((np.size(self.POD.snapshots, axis=0), n_points))
        for i in range(n_points):
            params = evaluation_points[:, i].reshape(-1)
            print(f'Evaluating network at parameters: {params}')
            params = tc.from_numpy(params)
            coefficients = self.network(params.float())
            coefficients = coefficients.cpu().detatch().numpy()
            rom_solutions[:, i] = self.POD.inverse(coefficients)
        print('Done evaluating.')
        return rom_solutions

    def k_fold_cross_validation(self, epochs, training_batch_size, testing_batch_size,
                                number_splits, print_plots):
        print("Running k-fold cross validation.")
        fold_training_losses = []
        fold_testing_losses = []
        kf = KFold(n_splits=number_splits, shuffle=True, random_state=0)
        kf_enum = enumerate(kf.split(X=self.POD.parameters, y=self.POD.coefficients))
        for fold, (training_ids, testing_ids) in kf_enum:
            self.network.apply(self.reset_weights())
            print(f'Fold number: {fold}')
            training_losses = []
            testing_losses = []

            # Separating training and testing samples by the Ids for the current fold and generating dataloaders
            training_dataset = SnapshotDataset(parameters=self.POD.parameters[:, training_ids],
                                               targets=self.POD.coefficients[:, testing_ids],
                                               device=self.device)
            testing_dataset = SnapshotDataset(parameters=self.POD.parameters[:, testing_ids],
                                              targets=self.POD.coefficients[:, training_ids],
                                              device=self.device)
            training_loader = DataLoader(dataset=training_dataset, batch_size=training_batch_size, shuffle=True)
            testing_loader = DataLoader(dataset=testing_dataset, batch_size=testing_batch_size, shuffle=True)

            # Train the network
            for epoch in range(1, epochs + 1):
                training_losses = self.train(train_loader=training_loader, training_losses=training_losses)
                testing_losses = self.test(test_loader=testing_loader, testing_losses=testing_losses)

            fold_training_losses.append(training_losses)
            fold_testing_losses.append(testing_losses)

        fold_training_losses = np.array(fold_training_losses)
        fold_testing_losses = np.array(fold_testing_losses)
        avg_training_losses = np.mean(fold_training_losses, axis=0)
        avg_testing_losses = np.mean(fold_testing_losses, axis=0)

        if print_plots:
            fig1 = plt.figure()
            plt.plot(np.arange(0, epochs),
                     avg_training_losses,
                     label=f'Training, final iteration loss: {avg_training_losses[-1]:.4f}')
            plt.plot(np.arange(0, epochs),
                     avg_testing_losses,
                     label=f'Testing, final iteration loss: {avg_testing_losses[-1]:.4f}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('MSE Loss')
            plt.legend()
            fig1.show()
        print('Done k-fold cross validation.')

    def train(self, train_loader, training_losses):
        self.network.train()
        current_loss = 0
        for batchId, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)  # Send tensors to device
            self.optimizer.zero_grad()  # Ensures that the gradients are zeroed
            outputs = self.network(inputs.float())  # Calculates outputs of the network
            loss = self.loss_function(outputs.float(), targets.float())  # Calculates the loss from the batch
            loss.backward()  # Back-propagates the loss through the network
            self.optimizer.step()  # Optimizes the parameters
            current_loss += loss.item()  # Appends loss to keep track

        current_loss = current_loss / len(train_loader.dataset)
        training_losses.append(current_loss)
        return training_losses

    def test(self, test_loader, testing_losses):
        self.network.eval()
        current_loss = 0
        with tc.no_grad():  # Do not compute gradients to save memory and time
            for batchId, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # Send tensors to device
                outputs = self.network(inputs.float())  # Calculate the outputs of the network
                loss = self.loss_function(outputs, targets)  # Calculate the losses
                current_loss += loss.item()
        current_loss = current_loss / len(test_loader.dataset)
        testing_losses.append(current_loss)
        return testing_losses

    def reset_weights(self):
        # Used to reset model weights to avoid weight leakage in between folds
        for layer in self.network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def run_neural_network_rom(snapshots_path,
                           parameters_path,
                           evaulation_points_path,
                           n_modes=None,
                           architecture=1,
                           epochs=500,
                           learning_rate=5e-3,
                           training_batch_size=15,
                           testing_batch_size=2,
                           number_kf_splits=5,
                           weight_decay=1e-3,
                           solutions_save_name=None,
                           run_network=True,
                           run_k_fold_validation=False,
                           print_plots=False):
    snapshots_file = np.genfromtxt(snapshots_path, delimieter=" ")
    parameters_file = np.genfromtxt(parameters_path, delimiter=" ")
    evaluation_points = np.genfromtxt(evaulation_points_path, delimiter=" ")
    if solutions_save_name is None:
        solutions_save_name = f'{np.size(parameters_file, axis=1)}_nnrom_solutions.txt'

    pod = POD(snapshots=snapshots_file, parameters=parameters_file, n_modes=n_modes)
    pod.transform()
    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')  # Uses cuda if available, otherwise uses cpu
    rom = PODNeuralNetworkROM(POD=pod, device=device)
    rom.initalize_network(architecture=architecture, learning_rate=learning_rate, weight_decay=weight_decay)

    if run_k_fold_validation:
        rom.k_fold_cross_validation(epochs=epochs,
                                    training_batch_size=training_batch_size,
                                    testing_batch_size=testing_batch_size,
                                    number_splits=number_kf_splits,
                                    print_plots=print_plots)

    if run_network:
        rom.build_network(epochs=epochs, batch_size=training_batch_size, print_plots=print_plots)
        rom_solutions = rom.evaluate_network(evaluation_points=evaluation_points)
        np.savetxt(solutions_save_name, rom_solutions, delimiter=" ")

    return 0






