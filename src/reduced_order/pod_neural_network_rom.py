import numpy as np
import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from scipy import linalg
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt


class POD:
    def __init__(self, snapshots, parameters, num_pod_modes):
        # num_pod_modes: set to 0 to keep all singular values, or enter a
        # positive int for a lower rank POD approximation
        self.snapshots = snapshots
        self.parameters = parameters
        self.num_pod_modes = num_pod_modes  
        self.ref_state = None
        self.POD_basis = None
        self.POD_coefficients = None
        self.sort_pattern = None
        self.inverse_sort = None
        self.scaler = None

    def scale_parameters(self, params):
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(params.T)
        self.parameters = self.scaler.transform(params.T)
        self.parameters = self.parameters.T

    def transform(self):
        self.scale_parameters(self.parameters)
        self.ref_state = np.mean(self.snapshots, axis=1).reshape(-1, 1)
        self.sort_pattern = np.argsort(self.ref_state, axis=0)
        self.inverse_sort = np.argsort(self.sort_pattern, axis=0)
        self.ref_state = self.ref_state[self.sort_pattern, 0]
        for i in range(np.size(self.snapshots, axis=1)):
            self.snapshots[:, i] = self.snapshots[:, i][self.sort_pattern.reshape(-1)]
        centered = self.snapshots - self.ref_state

        U, S, V = linalg.svd(centered, full_matrices=False)
        V = V.T  # To make computations clear, linalg.svd already returns transpose(V)
        if self.num_pod_modes != 0:
            U, S, V = U[:, :self.num_pod_modes], S[:self.num_pod_modes], V[:, :self.num_pod_modes]
        else:
            self.num_pod_modes = len(S)

        # I realize this next snippet is dumb but np.diag(S) @ V.T segfaults when called via the A-API for some reason
        temp = np.ones((self.num_pod_modes, self.num_pod_modes))
        for i in range(temp.shape[1]):
            temp[i, :] = V.T[i, :] * S[i]

        self.POD_basis = U
        # self.POD_coefficients = np.diag(S) @ V.T
        self.POD_coefficients = temp
        return 0

    def inverse(self, coeffs):
        rows, cols = self.POD_basis.shape
        prediction = np.ones(self.ref_state.shape)
        for i in range(rows):
            value = 0
            for j in range(cols):
                value += self.POD_basis[i, j] * coeffs[j]
            prediction[i, 0] = value
        prediction = prediction + self.ref_state

        # prediction = (self.POD_basis @ coeffs).reshape(-1, 1) + self.ref_state
        prediction = prediction[self.inverse_sort, 0]
        return prediction


class SnapshotDataset(Dataset):
    def __init__(self, parameters, targets, device):
        self.parameters = parameters
        self.targets = targets
        self.device = device

    def __len__(self):
        L = np.size(self.parameters, axis=1)
        return L

    def __getitem__(self, item):
        inputs = tc.from_numpy(self.parameters[:, item])
        targets = tc.from_numpy(self.targets[:, item])
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets


class NeuralNetwork(nn.Module):
    def __init__(self, outSize, arch=1):
        super(NeuralNetwork, self).__init__()
        self.outSize = outSize
        self.arch = arch

        self.arch1 = nn.Sequential(
            nn.Linear(2, 25),
            nn.PReLU(),
            nn.Linear(25, 50),
            nn.PReLU(),
            nn.Linear(50, 100),
            nn.PReLU(),
            nn.Linear(100, 50),
            nn.PReLU(),
            nn.Linear(50, self.outSize))

        self.arch2 = nn.Sequential(
            nn.Linear(2, 75),
            nn.PReLU(),
            nn.Linear(75, 75),
            nn.PReLU(),
            nn.Linear(75, 75),
            nn.PReLU(),
            nn.Linear(75, 25),
            nn.PReLU(),
            nn.Linear(25, self.outSize))

        self.arch3 = nn.Sequential(
            nn.Linear(2, 75),
            nn.PReLU(),
            nn.Linear(75, 75),
            nn.PReLU(),
            nn.Linear(75, 25),
            nn.PReLU(),
            nn.Linear(25, self.outSize))

    def forward(self, x):
        if self.arch == 1:
            x = self.arch1(x)
        elif self.arch == 2:
            x = self.arch2(x)
        elif self.arch == 3:
            x = self.arch3(x)
        return x


class PODNeuralNetworkROM:
    def __init__(self, save_path):
        self.early_stopping_tol = 1e-6
        self.loss_function = None
        self.network = None
        self.optimizer = None
        self.epochs = None
        self.training_batch_size = None
        self.testing_batch_size = None
        self.POD = None
        self.device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
        self.save_path = save_path
        tc.manual_seed(0)

    def load_data(self, snapshots_path, parameters_path, residuals_path, num_pod_modes):
        snapshot_residuals = self.read_data(residuals_path).reshape(-1)
        snapshots_file = self.read_data(snapshots_path)
        parameters_file = self.read_data(parameters_path)

        for i in range(0, len(snapshot_residuals)):
            if snapshot_residuals[i] == -1:
                parameters_file = np.delete(arr=parameters_file, obj=i, axis=1)
                snapshots_file = np.delete(arr=snapshots_file, obj=i, axis=1)

        self.POD = POD(snapshots=snapshots_file, parameters=parameters_file,
                       num_pod_modes=num_pod_modes)
        self.POD.transform()

    def initialize_network(self, architecture, epochs, learning_rate, training_batch_size, weight_decay):
        self.loss_function = nn.MSELoss()
        self.network = NeuralNetwork(outSize=self.POD.num_pod_modes, arch=architecture)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
        self.epochs = epochs
        self.training_batch_size = training_batch_size

    def build_network(self):
        self.network.apply(self.reset_weights)
        dataset = SnapshotDataset(parameters=self.POD.parameters,
                                  targets=self.POD.POD_coefficients,
                                  device=self.device)
        dataset_loader = DataLoader(dataset, batch_size=self.training_batch_size, shuffle=True)

        training_losses = []
        loss = 1
        epoch = 1
        while epoch < self.epochs and loss > self.early_stopping_tol:
            training_losses = self.train(train_loader=dataset_loader, training_losses=training_losses)
            if not epoch % 500:
                print("\tEpoch number: ", epoch, ", current loss: ", training_losses[-1])
            loss = training_losses[-1]
            epoch += 1
        self.write_data(path=os.path.join(self.save_path) + "training_losses.txt", data=np.array(training_losses))

    def evaluate_network(self, test_points_path):
        params_array = self.read_data(test_points_path)
        self.network.eval()
        params_array = self.POD.scaler.transform(params_array.T)
        params_array = params_array.T
        rom_solutions = []

        for i in range(0, np.size(params_array, axis=1)):
            params = tc.from_numpy(params_array[:, i].reshape(-1))
            coefficients = self.network(params.float())
            coefficients = coefficients.cpu().detach().numpy()
            solution = self.POD.inverse(coefficients)
            rom_solutions.append(solution)

        rom_solutions = np.concatenate(rom_solutions, axis=1)
        filename = os.path.join(self.save_path, f"pod_nnrom_solution.txt")
        if os.path.exists(filename):
            os.remove(filename)
        self.write_data(filename, rom_solutions)
        return rom_solutions

    def k_fold_cross_validation(self, testing_batch_size, number_splits, print_plots=False):
        print("Running k-fold cross validation.")
        fold_training_losses = []
        fold_testing_losses = []
        kf = KFold(n_splits=number_splits, shuffle=True, random_state=0)
        kf_enum = enumerate(kf.split(X=self.POD.POD_coefficients.T, y=self.POD.parameters.T))
        for fold, (training_ids, testing_ids) in kf_enum:
            self.network.apply(self.reset_weights)
            print(f'Fold number: {fold}')
            training_losses = []
            testing_losses = []

            # Separating training and testing samples by the Ids for the current fold and generating dataloaders
            training_dataset = SnapshotDataset(parameters=self.POD.parameters[:, training_ids],
                                               targets=self.POD.POD_coefficients[:, training_ids],
                                               device=self.device)
            testing_dataset = SnapshotDataset(parameters=self.POD.parameters[:, testing_ids],
                                              targets=self.POD.POD_coefficients[:, testing_ids],
                                              device=self.device)
            training_loader = DataLoader(dataset=training_dataset,
                                         batch_size=self.training_batch_size, shuffle=True)
            testing_loader = DataLoader(dataset=testing_dataset,
                                        batch_size=testing_batch_size, shuffle=True)

            # Train the network
            for epoch in range(1, self.epochs + 1):
                if not epoch % 100:
                    print(f'\tEpoch number: {epoch}, current loss: {training_losses[-1]}')
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
            plt.plot(np.arange(0, self.epochs),
                     avg_training_losses,
                     label=f'Training, final iteration loss: {avg_training_losses[-1]:.2e}')
            plt.plot(np.arange(0, self.epochs),
                     avg_testing_losses,
                     label=f'Testing, final iteration loss: {avg_testing_losses[-1]:.2e}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('MSE Loss')
            plt.ylim(0, max(np.max(avg_testing_losses), np.max(avg_training_losses))*1.001)
            plt.yscale('log')
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

    @staticmethod
    def reset_weights(network):
        # Used to reset model weights to avoid weight leakage in between folds
        for layer in network.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @staticmethod
    def read_data(path):
        data = []
        with open(path) as file:
            for line in file:
                data.append([float(x) for x in line.strip().split()])
        file.close()
        return np.array(data)

    @staticmethod
    def write_data(path, data):
        data = np.squeeze(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        rows = data.shape[0]
        with open(path, 'w', newline="\n") as file:
            for line in range(rows):
                file.write("  ".join(data[line].astype(str)) + "\n")
        file.close()


if __name__ == "__main__":
    # snapshots_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
    #                   "50_volume_pressure_snapshots_training_matrix.txt")
    # residual_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
    #                   "50_volume_pressure_snapshots_training_residuals.txt")
    # parameters_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
    #                    "50_volume_pressure_snapshots_training_parameters.txt")
    # testing_points_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
    #                        "5_volume_pressure_snapshots_parameters.txt")
    # testing_snapshots_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/"
    #                           + "5_volume_pressure_snapshots_matrix.txt")

    snapshots_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
                      "25_surface_pressure_snapshots_matrix.txt")
    residual_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
                      "25_surface_pressure_snapshots_residuals.txt")
    parameters_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
                       "25_surface_pressure_snapshots_parameters.txt")
    testing_points_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/" +
                           "3_surface_pressure_testing_parameters.txt")
    testing_snapshots_path = ("/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/"
                              + "3_surface_pressure_testing_matrix.txt")
    savedir = r"/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/"
    points_path = r"/home/alex/Codes/PHiLiP/build_release/tests/integration_tests_control_files/reduced_order/point_locations.txt"

    num_pod_modes = 0
    architecture = 3
    learning_rate = 5e-4
    weight_decay = 1e-3
    epochs = 10000
    training_batch_size = 25
    early_stop = 1e-5

    ROM = PODNeuralNetworkROM(savedir)
    ROM.load_data(snapshots_path, parameters_path, residual_path, num_pod_modes)
    ROM.initialize_network(architecture, epochs, learning_rate, training_batch_size, weight_decay)
    ROM.build_network()
    # ROM.k_fold_cross_validation(testing_batch_size=2, number_splits=5, print_plots=True)

    testing_matrix = ROM.read_data(testing_snapshots_path)
    testing_parameters = ROM.read_data(testing_points_path)
    points = ROM.read_data(points_path)
    rom_solutions = ROM.evaluate_network(testing_points_path)

    for i in range(np.size(testing_parameters, axis=1)):
        # params = [testing_parameters[0, i], testing_parameters[1, i]]
        params = testing_parameters[:, i]
        rom_solution = rom_solutions[:, i].reshape(-1, 1)
        fom_solution = testing_matrix[:, i].reshape(-1, 1)
        diff = rom_solution - fom_solution
        L2_error = np.linalg.norm(diff)
        print(f'L2 error for parameters {testing_parameters[:, i]}: {L2_error}')
        ROM.visualize_solution(points, testing_parameters[:, i],
                               np.concatenate((rom_solution, fom_solution), axis=1),
                               ["ROM", "FOM"])

    print("Done.")
