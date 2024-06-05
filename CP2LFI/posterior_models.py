import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import normflows as nf
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class normflow_posterior(BaseEstimator):
    """
    A class representing a posterior model using normalizing flows.

    Parameters:
    -----------
    TODO: Add parameters and their descriptions here.

    Attributes:
    -----------
    TODO: Add attributes and their descriptions here.

    Methods:
    --------
    TODO: Add methods and their descriptions here.
    """

    def __init__(
        self,
        latent_size,
        sample_size,
        n_flows=4,
        hidden_units=64,
        hidden_layers=2,
        permute_mask=True,
        enable_cuda=True,
    ):
        self.enable_cuda = enable_cuda
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.enable_cuda else "cpu"
        )
        self.latent_size = latent_size
        self.sample_size = sample_size
        self.n_flows = n_flows
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.permute_mask = permute_mask
        self.enable_cuda = enable_cuda

    def set_up_nflow(
        self,
        context_size,
        **kwargs,
    ):

        flows = []
        for i in range(self.n_flows):
            flows += [
                nf.flows.AutoregressiveRationalQuadraticSpline(
                    self.latent_size,
                    self.hidden_layers,
                    self.hidden_units,
                    num_context_channels=context_size,
                    permute_mask=self.permute_mask,
                    **kwargs,
                )
            ]
            flows += [nf.flows.LULinearPermute(self.latent_size)]

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(self.latent_size, trainable=False)

        # Construct flow model
        model = nf.ConditionalNormalizingFlow(q0, flows)

        # Move model on GPU if available
        model = model.to(self.device)
        return model

    def fit(
        self,
        X,
        theta,
        val_size=0.2,
        n_epochs=1000,
        patience=30,
        batch_size=200,
        learning_rate=3e-4,
        weight_decay=1e-5,
        torch_seed=45,
        split_seed=0,
        **kwargs,
    ):
        """
        Fit the posterior model to the given data.

        Parameters:
        -----------
        X : array-like, shape (n_lambdas, n_samples)
            The simulated data.

        theta : array-like, shape (n_lambdas, n_dim),
            The parameter values.

        Returns:
        --------
        self : object
            Returns self.
        """
        # setting up model
        self.model = self.set_up_nflow(context_size=X.shape[1], **kwargs)

        # splitting data
        x_train, x_val, theta_train, theta_val = train_test_split(
            X, theta, test_size=val_size, random_state=split_seed
        )

        # transforming into tensors
        x_train, x_val = torch.tensor(x_train, dtype=torch.float32).to(
            self.device
        ), torch.tensor(x_val, dtype=torch.float32).to(self.device)
        if theta.ndim == 1:
            theta_train, theta_val = (
                torch.tensor(theta_train, dtype=torch.float32)
                .view(-1, 1)
                .to(self.device),
                torch.tensor(theta_val, dtype=torch.float32)
                .view(-1, 1)
                .to(self.device),
            )
        else:
            theta_train, theta_val = (
                torch.tensor(theta_train, dtype=torch.float32).to(self.device),
                torch.tensor(theta_val, dtype=torch.float32).to(self.device),
            )

        torch.manual_seed(torch_seed)

        # creating tensors datasets
        train_data = TensorDataset(x_train, theta_train)
        val_data = TensorDataset(x_val, theta_val)
        train_loader, val_loader = DataLoader(
            train_data, shuffle=True, batch_size=batch_size
        ), DataLoader(val_data, shuffle=True, batch_size=batch_size)

        # setting up optimizers and lists
        avg_loss_val_list = []
        loss_train = []
        best_loss_history = []

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        best_loss = np.inf

        # starting neural network loop
        for it in tqdm(
            range(n_epochs), desc="Fitting normalizing flows posterior estimator"
        ):
            optimizer.zero_grad()
            loss_val_list = []

            # iterating and fitting model to train data loader
            context, x = iter(train_loader).__next__()
            context.requires_grad_(True)

            loss = self.model.forward_kld(x, context)
            loss_value = loss.to("cpu").data.numpy()
            loss_train.append(loss_value)

            # performing back-propagation and parameters update
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            # validating in validation data loader
            for context, x in val_loader:
                with torch.no_grad():
                    self.model.eval()
                    loss = self.model.forward_kld(x, context)
                    loss_value = loss.to("cpu").data.numpy()
                    loss_val_list.append(loss_value)

            avg_loss_val = np.mean(loss_val_list)
            avg_loss_val_list.append(avg_loss_val)

            if avg_loss_val < best_loss:
                best_loss = avg_loss_val
                best_loss_history.append(best_loss)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {it}")
                    break
            self.model.train()

        # saving all losses
        self.loss_hist_val = np.array(avg_loss_val_list)
        self.loss_hist_train = np.array(loss_train)
        self.loss_hist_best = np.array(best_loss_history)

        print("a")
        return self

    def plot_history(self):
        plt.plot(self.loss_hist_train, color="tab:red", label="training loss")
        plt.plot(self.loss_hist_val, color="tab:blue", label="validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def predict(self, thetas, X):
        """
        Predict posterior probability for each theta and X

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        prob_pred : array-like, shape (n_samples,)
            The predicted target values.
        """
        # converting thetas and X into tensors
        x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if thetas.ndim == 1:
            theta_tensor = (
                torch.tensor(thetas, dtype=torch.float32).view(-1, 1).to(self.device)
            )
        else:
            theta_tensor = torch.tensor(thetas, dtype=torch.float32).to(self.device)

        self.model.eval()
        log_prob = self.model.log_prob(theta_tensor, x_tensor).to("cpu")
        self.model.train()
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0
        return prob.detach().numpy()

    def sample(self, X, num_samples):
        """
        Sample from the posterior probability for theta given X

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        prob_pred : array-like, shape (n_samples,)
            The predicted target values.
        """
        if X.shape[0] == 1:
            X_s = np.tile(X, (num_samples, 1))
            # converting to torch tensor
            X_s = torch.tensor(X_s, dtype=torch.float32).to(self.device)
        else:
            X_s = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            sample, _ = self.model.sample(context=X_s, num_samples=num_samples).to(
                "cpu"
            )
        return sample.numpy()
