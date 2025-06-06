import numpy as np
import torch
import torch.nn as nn
from vehicle_model import VehicleModel as vm
from generate_trajectory import Trajectory, Waypoint
from tools.mpc_tools import State, Inputs


#define seed
torch.manual_seed(42)
class DynamicModel(nn.Module):
    def __init__(self, state_size: int = 5, input_size: int = 2,
                 hidden_size: int = 16, learning_rate: float = 0.001):
        super(DynamicModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_size + input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_size)  # output is \Delta x
        )
        
        self.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, 
                                          weight_decay=1e-6)
        
        self.state = State()
        self.max_steering_angle = 1  # TODO: use comfort criteria to set this value
        self.max_steering_angle = np.deg2rad(75)  
        self.max_acc = 1.2 
        self.max_steering_vel = 1 # TODO: use comfort criteria to set this value
        
    def forward(self, state_tensor: torch.Tensor, input_tensor: torch.Tensor):
        """
        Executa a inferência para prever Δx.
        """
        self.model.to(self.gpu)  
        
        x = torch.cat((state_tensor, input_tensor), dim=-1)
        x = x.to(self.gpu)  
        delta_state = self.model(x)  
        delta_state = delta_state.to(self.cpu)
        return delta_state
    
    def step(self, acc: float, steering_vel: float):
        """
        Atualiza o estado atual aplicando Δx.
        """
        # Tensor do estado atual
        state_tensor = torch.tensor(self.state.to_tensor(), dtype=torch.float32)

        # Tensor da entrada
        input_tensor = torch.tensor([acc, steering_vel], dtype=torch.float32)

        # Predição da variação Δx
        delta_state = self.forward(state_tensor, input_tensor)

        # Atualiza o estado somando Δx
        new_state_tensor = state_tensor + delta_state

        # Atualiza atributos do estado
        self.state.x = new_state_tensor[0].item()
        self.state.y = new_state_tensor[1].item()
        self.state.psi = np.mod(new_state_tensor[2].item() + np.pi, 2 * np.pi) - np.pi
        self.state.v = new_state_tensor[3].item()
        self.state.delta = np.clip(new_state_tensor[4].item(), -self.max_steering_angle, self.max_steering_angle)
        self.state.delta = np.mod(self.state.delta + np.pi, 2 * np.pi) - np.pi

        return self.state
        
    def define_state(self, x, y, psi, v, delta):
        """
        Define o estado inicial do veículo.
        """
        self.state = State(x=x, y=y, psi=psi, v=v, delta=delta)
        
    def train_step(self, delta_pred: torch.Tensor, delta_target: torch.Tensor):
        """
        Realiza um passo de treinamento sobre Δx.
        """
        delta_pred = delta_pred.to(self.gpu)  
        delta_target = delta_target.to(self.gpu)
        loss = self.criterion(delta_pred, delta_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item() 
    
    def save(self, path='dynamic_model.pth'):
        """
        Salva os pesos do modelo.
        """
        torch.save(self.state_dict(), path)

    def load(self, path='dynamic_model.pth'):
        """
        Carrega os pesos do modelo.
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        
    def copy(self):
        """
        Cria uma cópia do modelo.
        """
        new_model = DynamicModel()
        new_model.load_state_dict(self.state_dict())
        return new_model
        
        
        