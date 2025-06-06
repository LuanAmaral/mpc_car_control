import torch
import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import VehicleModel 
from dynamic_model import DynamicModel
from tools.mpc_tools import State

N_SAMPLES = 1000000
BATCH_SIZE = 128
EPOCHS = 500
LR = 0.001


def generate_dataset(model, n_samples=10000, batch_size=1000):
    states = []
    inputs = []
    deltas = []

    # General random samples
    for _ in range(n_samples):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        psi = np.random.uniform(-np.pi, np.pi)
        v = np.random.uniform(0, 10)
        delta = np.random.uniform(-0.5, 0.5)  

        state = np.array([x, y, psi, v, delta])

        acc = np.random.uniform(-1.2, 1.2)
        steering_vel = np.random.uniform(-1, 1)
        input_ = np.array([acc, steering_vel])

        model.define_state(state[0], state[1], state[2], state[3], state[4])

        next_state = model.step(acc, steering_vel)

        next_state = np.array([
            next_state.x,
            next_state.y,
            next_state.psi,
            next_state.v,
            next_state.delta
        ])
        
        delta_state = next_state - state

        states.append(state)
        inputs.append(input_)
        deltas.append(delta_state)

    # Batch where each state variable varies independently
    for i, (low, high) in enumerate([(-10, 10), (-10, 10), (-np.pi, np.pi), (0, 10), (-0.5, 0.5)]):
        for _ in range(batch_size):
            state = np.zeros(5)
            state[i] = np.random.uniform(low, high)  # Vary only one state variable
            state[3] = 1.0  # Set velocity to a constant for meaningful dynamics

            acc = 0.0
            steering_vel = 0.0
            input_ = np.array([acc, steering_vel])

            model.define_state(state[0], state[1], state[2], state[3], state[4])

            next_state = model.step(acc, steering_vel)

            next_state = np.array([
                next_state.x,
                next_state.y,
                next_state.psi,
                next_state.v,
                next_state.delta
            ])
            
            delta_state = next_state - state

            states.append(state)
            inputs.append(input_)
            deltas.append(delta_state)

    # Batch where each input variable varies independently
    for i, (low, high) in enumerate([(-1.5, 1.5), (-0.3, 0.3)]):
        for _ in range(batch_size):
            state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # Fixed state
            input_ = np.zeros(2)
            input_[i] = np.random.uniform(low, high)  # Vary only one input variable

            model.define_state(state[0], state[1], state[2], state[3], state[4])

            next_state = model.step(input_[0], input_[1])

            next_state = np.array([
                next_state.x,
                next_state.y,
                next_state.psi,
                next_state.v,
                next_state.delta
            ])
            
            delta_state = next_state - state

            states.append(state)
            inputs.append(input_)
            deltas.append(delta_state)

    return (np.array(states), np.array(inputs), np.array(deltas))


physical_model = VehicleModel()

states, controls, delta_states = generate_dataset(physical_model, N_SAMPLES, BATCH_SIZE)

idx = np.random.permutation(N_SAMPLES)
states = states[idx]
controls = controls[idx]
delta_states = delta_states[idx]

split = int(0.8 * N_SAMPLES)
train_states, test_states = states[:split], states[split:]
train_controls, test_controls = controls[:split], controls[split:]
train_deltas, test_deltas = delta_states[:split], delta_states[split:]


train_states = torch.tensor(train_states, dtype=torch.float32)
train_controls = torch.tensor(train_controls, dtype=torch.float32)
train_deltas = torch.tensor(train_deltas, dtype=torch.float32)

test_states = torch.tensor(test_states, dtype=torch.float32)
test_controls = torch.tensor(test_controls, dtype=torch.float32)
test_deltas = torch.tensor(test_deltas, dtype=torch.float32)

model = DynamicModel(learning_rate=LR)

train_losses = []
test_losses = []

best_loss = float('inf')
train_sat_count = 0

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(train_states.size()[0])
    epoch_loss = 0.0

    for i in range(0, train_states.size()[0], BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_states = train_states[indices]
        batch_controls = train_controls[indices]
        batch_deltas = train_deltas[indices]

        # Forward
        outputs = model.forward(batch_states, batch_controls)

        # Loss
        loss = model.train_step(outputs, batch_deltas)
        epoch_loss += loss

    epoch_loss /= (train_states.size()[0] / BATCH_SIZE)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        pred_test = model(test_states, test_controls)
        test_loss = model.criterion(pred_test, test_deltas).item()
        test_losses.append(test_loss)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    
    if test_loss < best_loss:
        if np.abs(test_loss - best_loss) < 1e-6:
            train_sat_count += 1
            if train_sat_count >= 10:
                print("Early stopping triggered.")
                break
        
        best_loss = test_loss


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Smooth L1)')
plt.title('Curva de perda durante o treinamento')
plt.legend()
plt.grid()
plt.show()


model.save('dynamic_model.pth')
print("Saved Model: dynamic_model.pth")
