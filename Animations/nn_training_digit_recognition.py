''' This script show a 2D animation of the training of a neural network for digit recognition'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Neural Network Definition
class DigitRecognitionNet(nn.Module):
    def __init__(self):
        super(DigitRecognitionNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer: 28x28 = 784
        self.fc2 = nn.Linear(128, 64)   # Hidden layer 1
        self.fc3 = nn.Linear(64, 32)    # Hidden layer 2
        self.fc4 = nn.Linear(32, 10)    # Output layer: 10 digits
        self.relu = nn.ReLU()
        
        # Storage for activations (for visualization)
        self.activations = {}
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Layer 1
        x = self.fc1(x)
        self.activations['layer1'] = x.detach().cpu().numpy()
        x = self.relu(x)
        
        # Layer 2
        x = self.fc2(x)
        self.activations['layer2'] = x.detach().cpu().numpy()
        x = self.relu(x)
        
        # Layer 3
        x = self.fc3(x)
        self.activations['layer3'] = x.detach().cpu().numpy()
        x = self.relu(x)
        
        # Output layer
        x = self.fc4(x)
        self.activations['output'] = x.detach().cpu().numpy()
        
        return x
    
def get_next_batch():
    """Get next training batch"""
    global data_iter
    try:
        image, label = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        image, label = next(data_iter)
    return image.to(device), label.to(device)

def train_step():
    """Perform one training step with a batch"""
    global training_step, correct_predictions, total_predictions
    global smoothed_loss_history, loss_smoothing
    
    images, labels = get_next_batch()
    batch_size = images.size(0)
    
    # Forward pass
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Calculate batch accuracy
    _, predicted = torch.max(outputs.data, 1)
    batch_correct = (predicted == labels).sum().item()
    correct_predictions += batch_correct
    total_predictions += batch_size
    
    # Store metrics
    loss_value = loss.item()
    loss_history.append(loss_value)
    
    # Compute smoothed loss (exponential moving average)
    if len(smoothed_loss_history) == 0:
        smoothed_loss_history.append(loss_value)
    else:
        smoothed = loss_smoothing * smoothed_loss_history[-1] + (1 - loss_smoothing) * loss_value
        smoothed_loss_history.append(smoothed)
    
    current_accuracy = 100.0 * correct_predictions / total_predictions
    accuracy_history.append(current_accuracy)
    
    training_step += 1
    
    # Return batch data and predictions
    return images, labels, outputs, loss_value, predicted, batch_correct, batch_size

def init():
    """Initialize animation"""
    global input_display, output_bars, loss_line, accuracy_line, prediction_text
    global current_image, current_label, current_output, current_batch_idx
    global smoothed_loss_line
    
    # Get first batch
    current_image, current_label, current_output, _, pred, _, _ = train_step()
    
    # Select first sample from batch for display
    current_batch_idx = 0
    
    # Display input image (first sample from batch)
    img_np = current_image[current_batch_idx].cpu().squeeze().numpy()
    input_display = ax_input.imshow(img_np, cmap='gray', interpolation='nearest')
    ax_input.text(0.5, -0.1, f'Label: {current_label[current_batch_idx].item()}', 
                 transform=ax_input.transAxes, ha='center', fontsize=12, fontweight='bold')
    
    # Initialize output bars (for first sample)
    probs = torch.softmax(current_output[current_batch_idx], dim=0).detach().cpu().numpy()
    output_bars = ax_output.bar(range(10), probs, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Initialize prediction text
    prediction_text = ax_prediction.text(0.5, 0.5, '', transform=ax_prediction.transAxes,
                                        ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Initialize loss lines
    loss_line, = ax_loss.plot([], [], 'r-', linewidth=1, alpha=0.3, label='Loss (raw)')
    smoothed_loss_line, = ax_loss.plot([], [], 'r-', linewidth=2, label='Loss (smoothed)')
    ax_loss.legend(loc='upper right')
    
    # Initialize accuracy line
    accuracy_line, = ax_accuracy.plot([], [], 'g-', linewidth=2, label='Accuracy')
    ax_accuracy.legend(loc='lower right')
    
    return []

def update_neuron_visualization(activations_dict, layer_idx, layer_name):
    """Update neuron colors based on average batch activations"""
    if layer_name not in activations_dict:
        return
    
    # Get activations for entire batch and compute mean across batch
    batch_activations = activations_dict[layer_name]  # Shape: (batch_size, neurons)
    activations = batch_activations.mean(axis=0)  # Average across batch dimension
    
    # Normalize activations for visualization
    if len(activations) > 0:
        act_min = activations.min()
        act_max = activations.max()
        if act_max - act_min > 0:
            normalized_acts = (activations - act_min) / (act_max - act_min)
        else:
            normalized_acts = np.zeros_like(activations)
    else:
        normalized_acts = np.zeros_like(activations)
    
    # Sample activations if layer has more neurons than we're visualizing
    visual_size = len(neuron_circles[layer_idx])
    if len(normalized_acts) > visual_size:
        # Sample evenly distributed neurons
        indices = np.linspace(0, len(normalized_acts) - 1, visual_size, dtype=int)
        sampled_acts = normalized_acts[indices]
    else:
        sampled_acts = normalized_acts[:visual_size]
    
    # Update circle colors
    for i, (circle, act) in enumerate(zip(neuron_circles[layer_idx], sampled_acts)):
        # Color map from purple (low) to orange (high)
        color = plt.cm.plasma(act)
        circle.set_facecolor(color)
        circle.set_alpha(0.7 + 0.3 * act)  # More active = more opaque

def animate(frame):
    """Animation function"""
    global current_image, current_label, current_output, phase
    global training_step, phase_text, current_batch_idx
    
    # Alternate between forward pass visualization and training update
    if frame % 2 == 0:
        phase = 'forward'
        phase_text.set_text('FORWARD PROPAGATION (Batch Training)')
        phase_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        # Perform training step with batch
        current_image, current_label, current_output, loss, pred_batch, batch_correct, batch_size = train_step()
        
        # Cycle through samples in the batch for display
        current_batch_idx = frame % batch_size
        
        # Update input display (show one sample from batch)
        img_np = current_image[current_batch_idx].cpu().squeeze().numpy()
        input_display.set_data(img_np)
        ax_input.texts[0].set_text(f'Label: {current_label[current_batch_idx].item()} (Sample {current_batch_idx+1}/{batch_size})')
        
        # Get activations from model (averaged across batch for network viz)
        activations = model.activations
        
        # Update neurons for each layer (showing batch-averaged activations)
        update_neuron_visualization(activations, 0, 'layer1')  # Input represented by layer1
        update_neuron_visualization(activations, 1, 'layer1')
        update_neuron_visualization(activations, 2, 'layer2')
        update_neuron_visualization(activations, 3, 'layer3')
        update_neuron_visualization(activations, 4, 'output')
        
        # Highlight connections based on activation flow
        for layer_idx, layer_connections in enumerate(connection_lines):
            for line, i, j in layer_connections:
                # Make connections more visible during forward pass
                line.set_alpha(0.3)
                line.set_color('blue')
                line.set_linewidth(0.5)
        
        # Update output probabilities (for displayed sample)
        probs = torch.softmax(current_output[current_batch_idx], dim=0).detach().cpu().numpy()
        pred_sample = pred_batch[current_batch_idx].item()
        label_sample = current_label[current_batch_idx].item()
        is_correct = (pred_sample == label_sample)
        
        for i, (bar, prob) in enumerate(zip(output_bars, probs)):
            bar.set_height(prob)
            if i == label_sample:
                bar.set_color('green' if is_correct else 'red')
                bar.set_alpha(0.9)
            elif i == pred_sample:
                bar.set_color('orange')
                bar.set_alpha(0.9)
            else:
                bar.set_color('steelblue')
                bar.set_alpha(0.5)
        
        # Update prediction text
        batch_acc = 100.0 * batch_correct / batch_size
        if is_correct:
            prediction_text.set_text(f'✓ Predicted: {pred_sample}\n(Batch: {batch_correct}/{batch_size} = {batch_acc:.1f}%)')
            prediction_text.set_color('green')
        else:
            prediction_text.set_text(f'✗ Predicted: {pred_sample}\nActual: {label_sample}\n(Batch: {batch_correct}/{batch_size} = {batch_acc:.1f}%)')
            prediction_text.set_color('red')
        
        # Update stats text
        current_acc = 100.0 * correct_predictions / total_predictions
        stats_text.set_text(f'Step: {training_step} | Loss: {loss:.4f} | Accuracy: {current_acc:.2f}% | Batch Size: {batch_size}')
        
    else:
        phase = 'update'
        phase_text.set_text('WEIGHT UPDATE (Backpropagation)')
        phase_text.set_bbox(dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
        
        # Fade connections during update
        for layer_idx, layer_connections in enumerate(connection_lines):
            for line, i, j in layer_connections:
                line.set_alpha(0.15)
                line.set_color('red')
                line.set_linewidth(0.3)
        
        # Pulse effect on neurons during weight update
        for layer_circles_list in neuron_circles:
            for circle in layer_circles_list:
                current_color = circle.get_facecolor()
                circle.set_alpha(0.5)
    
    # Update loss plot
    if len(loss_history) > 0:
        steps = range(len(loss_history))
        loss_line.set_data(steps, loss_history)
        smoothed_loss_line.set_data(steps, smoothed_loss_history)
        ax_loss.set_xlim(0, max(100, len(loss_history)))
        ax_loss.set_ylim(0, max(3, max(loss_history[:min(50, len(loss_history))]) * 1.1))  # Cap using first 50 steps
    
    # Update accuracy plot
    if len(accuracy_history) > 0:
        steps = range(len(accuracy_history))
        accuracy_line.set_data(steps, accuracy_history)
        ax_accuracy.set_xlim(0, max(100, len(accuracy_history)))
    
    return []

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Use proper batch size for stable training
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss, and optimizer
    model = DigitRecognitionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Get data iterator
    data_iter = iter(train_loader)

    # Visualization setup
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.4)

    # Main network visualization
    ax_network = fig.add_subplot(gs[:, :2])
    ax_network.set_xlim(-0.5, 4.5)
    ax_network.set_ylim(-2, 18)
    ax_network.axis('off')
    ax_network.set_title('Neural Network Activity During Training', fontsize=16, fontweight='bold', pad=20)

    # Input image display
    ax_input = fig.add_subplot(gs[0, 2])
    ax_input.set_title('Input Image (28×28)', fontsize=12, fontweight='bold')
    ax_input.axis('off')

    # Prediction vs Ground Truth
    ax_prediction = fig.add_subplot(gs[0, 3])
    ax_prediction.set_title('Prediction vs Truth', fontsize=12, fontweight='bold')
    ax_prediction.axis('off')

    # Output probabilities
    ax_output = fig.add_subplot(gs[1, 2:])
    ax_output.set_title('Output Probabilities', fontsize=12, fontweight='bold')
    ax_output.set_xlabel('Digit', fontsize=10)
    ax_output.set_ylabel('Probability', fontsize=10)
    ax_output.set_xlim(-0.5, 9.5)
    ax_output.set_ylim(0, 1.1)
    ax_output.set_xticks(range(10))
    ax_output.grid(True, alpha=0.3)

    # Training metrics
    ax_loss = fig.add_subplot(gs[2, 2])
    ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Training Steps', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.set_xlim(0, 100)
    ax_loss.set_ylim(0, 3)
    ax_loss.grid(True, alpha=0.3)

    # Accuracy plot
    ax_accuracy = fig.add_subplot(gs[2, 3])
    ax_accuracy.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax_accuracy.set_xlabel('Training Steps', fontsize=10)
    ax_accuracy.set_ylabel('Accuracy (%)', fontsize=10)
    ax_accuracy.set_xlim(0, 100)
    ax_accuracy.set_ylim(0, 100)
    ax_accuracy.grid(True, alpha=0.3)

    # Network architecture for visualization
    layer_sizes = [784, 128, 64, 32, 10]
    # Sample subset for visualization
    visual_layer_sizes = [49, 20, 15, 10, 10]  # Reduced for cleaner visualization
    n_layers = len(visual_layer_sizes)

    # Calculate neuron positions
    neuron_positions = []
    for layer_idx, size in enumerate(visual_layer_sizes):
        if layer_idx == 0:
            # Input layer in 7x7 grid
            positions = []
            grid_size = 7
            for i in range(grid_size):
                for j in range(grid_size):
                    y = 16 - i * 2.2
                    x = layer_idx - 0.3 + j * 0.1
                    if len(positions) < size:
                        positions.append((layer_idx, y))
        else:
            y_positions = np.linspace(1, 16, size)
            positions = [(layer_idx, y) for y in y_positions]
        
        neuron_positions.append(positions)

    # Draw connections (sample only for performance)
    connection_lines = []
    sample_connections = 5  # Only draw a few connections per neuron

    for layer_idx in range(n_layers - 1):
        layer_connections = []
        for i, (x1, y1) in enumerate(neuron_positions[layer_idx]):
            # Sample connections
            next_layer_size = len(neuron_positions[layer_idx + 1])
            connection_indices = np.random.choice(next_layer_size, 
                                                min(sample_connections, next_layer_size), 
                                                replace=False)
            
            for j in connection_indices:
                x2, y2 = neuron_positions[layer_idx + 1][j]
                line = ax_network.plot([x1, x2], [y1, y2], 'gray', alpha=0.1, 
                                    linewidth=0.2, zorder=1)[0]
                layer_connections.append((line, i, j))
        
        connection_lines.append(layer_connections)

    # Draw neurons
    neuron_circles = []
    for layer_idx, positions in enumerate(neuron_positions):
        layer_circles = []
        for i, (x, y) in enumerate(positions):
            radius = 0.08 if layer_idx == 0 else 0.12
            circle = Circle((x, y), radius, fill=True, color='lightgray', 
                        edgecolor='darkblue', linewidth=0.8, zorder=3, alpha=0.8)
            ax_network.add_patch(circle)
            layer_circles.append(circle)
        
        neuron_circles.append(layer_circles)

    # Add layer labels
    layer_names = ['Input\n(784)', 'Hidden 1\n(128)', 'Hidden 2\n(64)', 
                'Hidden 3\n(32)', 'Output\n(10)']
    for i in range(n_layers):
        ax_network.text(i, -0.5, layer_names[i], ha='center', fontsize=10, 
                    fontweight='bold')

    # Training phase indicator
    phase_text = ax_network.text(2, 17.5, '', ha='center', fontsize=13, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

    # Stats text
    stats_text = ax_network.text(2, -1.5, '', ha='center', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='blue', edgecolor='black', label='Forward Pass', alpha=0.7),
        mpatches.Patch(facecolor='orange', edgecolor='black', label='High Activation', alpha=0.7),
        mpatches.Patch(facecolor='purple', edgecolor='black', label='Low Activation', alpha=0.7)
    ]
    ax_network.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Initialize display elements
    input_display = None
    output_bars = None
    loss_line = None
    accuracy_line = None
    prediction_text = None

    # Training metrics storage
    loss_history = []
    accuracy_history = []
    smoothed_loss_history = []  # Exponential moving average of loss
    training_step = 0
    correct_predictions = 0
    total_predictions = 0
    batch_losses = []  # Store losses within current batch
    current_batch_idx = 0  # Which sample from batch to display
    loss_smoothing = 0.9  # EMA coefficient

    # Animation state
    current_image = None
    current_label = None
    current_output = None
    phase = 'forward'  # 'forward' or 'update'
    smoothed_loss_line = None  # Will be initialized in init()

    # Create animation
    print("Starting training animation...")
    print(f"Training with batch size {BATCH_SIZE} for stable convergence!")
    print("The network will train in real-time on MNIST data!")
    print("Visualization shows:")
    print("  - Network activations: averaged across the batch")
    print("  - Individual sample: cycles through batch samples")
    print("  - Batch accuracy: shown in prediction panel\n")

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=500, interval=200,  # 200ms per frame
                                blit=False, repeat=True)

    plt.tight_layout()
    plt.show()
