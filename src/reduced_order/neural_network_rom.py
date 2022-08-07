import sys
import os
import numpy as np
import numpy.linalg as la
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy import linalg
import torch.optim as optim
from sklearn import preprocessing


class NetworkDataset(Dataset):
    def __init__(self, params, coeffs, device):
        self.coeffs = coeffs
        self.params = params
        self.device = device

    def __len__(self):
        return np.size(self.params, 0)

    def __getitem__(self, idx):
        inputs = tc.from_numpy(self.params[idx, :])
        targets = tc.from_numpy(self.coeffs[idx, :])
        return inputs.to(self.device), targets.to(self.device)


class Network(nn.Module):
    def __init__(self, outSize):
        super(Network, self).__init__()
        self.outSize = outSize
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

    def forward(self, x):
        return self.arch1


class NeuralNetROM:
    def __init__(self, snapsFile, paramsFile, kfstate, plots):
        self.snapshotsFile = snapsFile
        self.paramsFile = paramsFile
        self.kfState = kfstate
        self.plots = plots
        self.dimension = 0
        self.params = []
        self.coeffs = None
        self.basis = None
        self.network = None
        self.referenceState = None
        self.network = None

        # Getting snapshot matrix
        snapshots = []
        file = open(self.snapshotsFile)
        for line in file:
            snapshots.append(np.array(line.split()).astype(float))
        file.close()

        # Getting parameters associated with snapshot matrix
        file = open(self.paramsFile)
        for line in file:
            if ("mach_number" in line) or ("angle_of_attack" in line):
                continue
            else:
                line = line.replace("|", "").replace("\n", "")
                self.params.append(np.array(line.split()).astype(float))
        file.close()

        snapshots = np.array(snapshots)
        self.params = np.array(self.params)

        # Scaling parameters
        scaler = preprocessing.MinMaxScaler()
        self.params = scaler.fit_transform(self.params)

        # Computing basis and coefficients
        self.referenceState = la.norm(snapshots)
        for i in range(np.size(self.referenceState)):
            snapshots[:,  i] -= self.referenceState
        U, S, V = linalg.svd(snapshots, full_matrices=False)
        self.basis = V  # POD basis
        self.coeffs = U @ np.diag(S)  # Matrix of coefficients
        self.dimension = np.size(self.basis, axis=1)

    def inverse(self, param):
        """Compute the full order solution using the predicted coefficients."""
        self.network.eval()
        inputs = tc.from_numpy(param.reshape(-1))
        coeffs = self.network(inputs.float())
        coeffs = coeffs.cpu().detach().numpy()
        soln = coeffs @ self.basis + self.referenceState
        return soln

    def buildNeuralNetwork(self):
        """Function to train and return the neural network for predicting the POD coefficients. Inputs are the parameters
        used in the snapshots and the targets are the coefficients obtained from the snapshot matrix. dimension is the
        number of POD basis used. kfState is True/False to determine if the k-fold validation should be performed or not.
        Set kfState = True when tweaking the network and kfState = False when purely utilizing the network. Set plots
        parameter to false if training/testing error plots are not desired."""

        def train(net, trainLoader):
            """Trains the network"""
            net.train()
            currentLoss = 0
            for batchId, (inputs, targets) in enumerate(trainLoader):
                inputs, targets = inputs.to(device), targets.to(device)  # Send tensors to device
                optimizer.zero_grad()  # Ensures that the gradients are zeroed
                outputs = net(inputs.float())  # Calculates outputs of the network
                loss = lossFun(outputs.float(), targets.float())  # Calculates the loss from the batch
                loss.backward()  # Back-propagates the loss through the network
                optimizer.step()  # Optimizes the parameters
                currentLoss += loss.item()  # Appends loss to keep track

            currentLoss = currentLoss / len(trainLoader.dataset)
            trainLoss.append(currentLoss)

        def test(net, testLoader):
            """Tests the network during the training phase"""
            net.eval()
            currentLoss = 0
            with tc.no_grad():  # Do not compute gradients to save memory and time
                for batchId, (inputs, targets) in enumerate(testLoader):
                    inputs, targets = inputs.to(device), targets.to(device)  # Send tensors to device
                    outputs = net(inputs.float())  # Calculate the outputs of the network
                    loss = lossFun(outputs, targets)  # Calculate the losses
                    currentLoss += loss.item()
            currentLoss = currentLoss / len(testLoader.dataset)
            testLoss.append(currentLoss)

        def resetWeights(net):
            """Resets the model weights to avoid weight leakage in between folds"""
            for layer in net.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # misc setup and definitions
        device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')  # Uses cuda if available, otherwise uses cpu
        randomSeed = 0  # Sets random seed for repeatability
        tc.manual_seed(randomSeed)
        epochs, lr, trainBatchSize, testBatchSize = (1500, 5 * 10 ** (-4), 15, 2)
        lossFun = nn.MSELoss()

        # Initializing the k-fold cross validation (to get more accurate performance/loss metrics)
        if self.kfState:
            k = 2  # Number of folds to be used in the k-fold cross validation
            training = []
            testing = []
            kf = KFold(n_splits=k, shuffle=True, random_state=randomSeed)
            for fold, (trainIds, testIds) in enumerate(kf.split(X=self.params, y=self.coeffs)):
                print('Fold ', fold)
                trainLoss = []
                testLoss = []

                # Separating training and testing samples by the Ids for the current fold and generating dataloaders
                trainData = NetworkDataset(params=self.params[trainIds, :], coeffs=self.coeffs[trainIds, :], device=device)
                testData = NetworkDataset(params=self.params[testIds, :], coeffs=self.coeffs[testIds, :], device=device)
                trainLoader = DataLoader(dataset=trainData, batch_size=trainBatchSize, shuffle=True)
                testLoader = DataLoader(dataset=testData, batch_size=testBatchSize, shuffle=True)

                network = Network(outSize=self.dimension)
                network.apply(resetWeights)
                optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-3)
                # optimizer = optim.Adam(network.parameters(), lr=lr)

                # Training the network
                for epoch in range(1, epochs + 1):
                    train(network, trainLoader)
                    test(network, testLoader)

                training.append(trainLoss)
                testing.append(testLoss)

            training = np.array(training)
            testing = np.array(testing)
            avgTrainLoss = np.mean(training, axis=0)
            avgTestLoss = np.mean(testing, axis=0)

            if self.plots:
                # Plotting loss metrics
                fig1 = plt.figure()
                plt.plot(np.arange(0, epochs), avgTrainLoss,
                         label=f'Training, final iteration loss: {avgTrainLoss[-1]:.4f}')
                plt.plot(np.arange(0, epochs), avgTestLoss, label=f'Testing, final iteration loss: {avgTestLoss[-1]:.4f}')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('MSE Loss')
                plt.legend()
                fig1.show()

        # Performing final training on the network on the full dataset (no validation)
        self.network = Network(outSize=self.dimension)
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        dataset = NetworkDataset(params=self.params, coeffs=self.coeffs, device=device)
        datasetLoader = DataLoader(dataset, batch_size=trainBatchSize, shuffle=True)
        trainLoss = []
        for epoch in range(1, epochs + 1):
            train(self.network, datasetLoader)

        if self.plots:
            plt.plot(np.arange(0, epochs), trainLoss, label=f'Training, final iteration loss: {trainLoss[-1]:.4f}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Full dataset training losses')
            plt.legend(loc='upper right')
            plt.show()


if __name__ == "__main__":
    snapshotDirectory = r"../../build_release/tests/integration_tests_control_files/reduced_order/"
    # Get latest snapshot iteration
    files = os.listdir(snapshotDirectory)
    itrNumber = 0
    for file in files:
        if "solution_snapshot_iteration" in file or "snapshot_table_iteration" in file:
            num = int("".join(filter(str.isdigit, file)))
            if num > itrNumber:
                itrNumber = num
            else:
                continue
        else:
            continue

    snapshotFile = snapshotDirectory + f"solution_snapshots_iteration_{itrNumber}.txt"
    parametersFile = snapshotDirectory + f"snapshot_table_iteration_{itrNumber}.txt"

    network = NeuralNetROM(snapsFile=snapshotFile, paramsFile=parametersFile, kfstate=True, plots=True)
    network.buildNeuralNetwork()






