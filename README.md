# Google Colab

## Code

```python
# CIFAR-10 Simple Neural Network (< 1000 parameters) with RGB565 Input
# Run this in Google Colab

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import time

def rgb888_to_rgb565(rgb888_tensor):
    """
    Convert RGB888 tensor (3 channels, values 0-1) to RGB565 format
    RGB565: RRRRRGGGGGGBBBBB (5 bits R, 6 bits G, 5 bits B)
    Returns uint16 values
    """
    # Convert from [0,1] to [0,255] and then to appropriate bit ranges
    rgb888_tensor = (rgb888_tensor + 1.0) / 2.0  # Denormalize from [-1,1] to [0,1]
    rgb888_tensor = torch.clamp(rgb888_tensor, 0, 1)
    
    r = (rgb888_tensor[:, 0, :, :] * 255).long()  # [0, 255]
    g = (rgb888_tensor[:, 1, :, :] * 255).long()  # [0, 255]
    b = (rgb888_tensor[:, 2, :, :] * 255).long()  # [0, 255]
    
    # Convert to RGB565 bit ranges
    r5 = (r >> 3)  # 8 bits -> 5 bits (0-31)
    g6 = (g >> 2)  # 8 bits -> 6 bits (0-63)
    b5 = (b >> 3)  # 8 bits -> 5 bits (0-31)
    
    # Pack into RGB565 format: RRRRRGGGGGGBBBBB
    rgb565 = (r5 << 11) | (g6 << 5) | b5
    
    return rgb565.to(torch.uint16)

def rgb565_to_float_tensor(rgb565_tensor):
    """
    Convert RGB565 tensor to normalized float RGB tensor
    """
    # Convert to int32 first since bitwise operations are supported for this type
    rgb565_tensor_int = rgb565_tensor.to(torch.int32)
    
    # Extract RGB components from RGB565
    r5 = ((rgb565_tensor_int >> 11) & 0x1F).float()  # Extract 5 R bits
    g6 = ((rgb565_tensor_int >> 5) & 0x3F).float()   # Extract 6 G bits
    b5 = (rgb565_tensor_int & 0x1F).float()          # Extract 5 B bits
    
    # Normalize to [0, 1] range
    r = r5 / 31.0
    g = g6 / 63.0
    b = b5 / 31.0
    
    # Stack the components
    return torch.stack([r, g, b], dim=1)

class SimpleCIFAR10Net_RGB565(nn.Module):
    def __init__(self):
        super(SimpleCIFAR10Net_RGB565, self).__init__()
        # Input: RGB565 format [batch, 32, 32] uint16
        # We'll convert to float internally and then use same architecture
        
        # Architecture: 32x32 -> 4x4 (via 8x8 avg pool) -> flatten to 48 features -> 16 -> 10
        # Parameters: 48*16 + 16*10 + 16 + 10 = 768 + 160 + 26 = 954 parameters ✓
        
        self.pool = nn.AvgPool2d(8, stride=8)  # 32x32 -> 4x4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48, 16)  # 4*4*3 = 48 -> 16
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)  # 16 -> 10 classes
        
    def forward(self, x):
        # x is RGB565 format [batch, 32, 32] uint16
        
        # Convert RGB565 to float tensor [batch, 3, 32, 32]
        x = rgb565_to_float_tensor(x)
        
        x = self.pool(x)      # [batch, 3, 32, 32] -> [batch, 3, 4, 4]
        x = self.flatten(x)   # [batch, 48]
        x = self.fc1(x)       # [batch, 16]
        x = self.relu(x)
        x = self.fc2(x)       # [batch, 10]
        return x

# Custom Dataset class to handle RGB565 conversion
class CIFAR10_RGB565(torch.utils.data.Dataset):
    def __init__(self, cifar10_dataset):
        self.cifar10_dataset = cifar10_dataset
        
    def __len__(self):
        return len(self.cifar10_dataset)
    
    def __getitem__(self, idx):
        image, label = self.cifar10_dataset[idx]
        # Convert RGB888 tensor to RGB565
        rgb565_image = rgb888_to_rgb565(image.unsqueeze(0)).squeeze(0)  # Remove batch dim
        return rgb565_image, label

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize model
model = SimpleCIFAR10Net_RGB565()
param_count = count_parameters(model)
print(f"Total parameters: {param_count}")

# Data loading with RGB565 conversion
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load original CIFAR-10 datasets
cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Wrap with RGB565 conversion
trainset = CIFAR10_RGB565(cifar10_trainset)
testset = CIFAR10_RGB565(cifar10_testset)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# CIFAR-10 classes
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training with RGB565 input...")
print("Input format: uint16_t pInputBuffer[32*32] // RRRRRGGGGGGBBBBB")

model.train()
for epoch in range(20):  # Quick training
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
            running_loss = 0.0

# Test accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Extract weights for C code
model.eval()
model.to('cpu')

# Get weights and biases
fc1_weight = model.fc1.weight.detach().numpy()  # [16, 48]
fc1_bias = model.fc1.bias.detach().numpy()      # [16]
fc2_weight = model.fc2.weight.detach().numpy()  # [10, 16]
fc2_bias = model.fc2.bias.detach().numpy()      # [10]

print("\n=== WEIGHTS FOR C CODE (RGB565 INPUT) ===")
print("// Input format: uint16_t pInputBuffer[32*32]; // RRRRRGGGGGGBBBBB")
print("// FC1 Weight [16][48]")
print("float fc1_weight[16][48] = {")
for i in range(16):
    print("  {", end="")
    for j in range(48):
        print(f"{fc1_weight[i][j]:.6f}f", end="")
        if j < 47: print(", ", end="")
    print("},")
print("};")

print("\n// FC1 Bias [16]")
print("float fc1_bias[16] = {")
for i in range(16):
    print(f"  {fc1_bias[i]:.6f}f", end="")
    if i < 15: print(",")
print("\n};")

print("\n// FC2 Weight [10][16]")
print("float fc2_weight[10][16] = {")
for i in range(10):
    print("  {", end="")
    for j in range(16):
        print(f"{fc2_weight[i][j]:.6f}f", end="")
        if j < 15: print(", ", end="")
    print("},")
print("};")

print("\n// FC2 Bias [10]")
print("float fc2_bias[10] = {")
for i in range(10):
    print(f"  {fc2_bias[i]:.6f}f", end="")
    if i < 9: print(",")
print("\n};")

# Test single inference time with RGB565 input
print("\n=== TESTING RGB565 INPUT ===")

# Create a test RGB565 input (simulating C array format)
test_rgb565 = torch.randint(0, 65536, (1, 32, 32), dtype=torch.uint16)
print(f"Test RGB565 input shape: {test_rgb565.shape}")
print(f"Test RGB565 input dtype: {test_rgb565.dtype}")
print(f"Sample RGB565 values: {test_rgb565[0, 0, :5]}")

model.eval()
with torch.no_grad():
    start_time = time.time()
    output = model(test_rgb565)
    end_time = time.time()
    print(f"PyTorch inference time with RGB565: {(end_time - start_time) * 1000:.3f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {torch.argmax(output, dim=1).item()} ({classes[torch.argmax(output, dim=1).item()]})")

# Show how to convert from C array format
print("\n=== C CODE INTEGRATION EXAMPLE ===")
print("""
// In your C code, you would have:
uint16_t pInputBuffer[32*32]; // RRRRRGGGGGGBBBBB format

// To use with this neural network:
// 1. The network expects RGB565 values in range [0, 65535]
// 2. Each pixel is stored as: (R5 << 11) | (G6 << 5) | B5
// 3. Array is organized as: [row 0 col 0, row 0 col 1, ..., row 31 col 31]

// Example RGB565 decoding in C:
void decode_rgb565(uint16_t rgb565, uint8_t* r, uint8_t* g, uint8_t* b) {
    *r = (rgb565 >> 11) & 0x1F;        // 5 bits, range 0-31
    *g = (rgb565 >> 5) & 0x3F;         // 6 bits, range 0-63  
    *b = rgb565 & 0x1F;                // 5 bits, range 0-31
}

// The neural network will internally:
// 1. Extract R, G, B components from each RGB565 value
// 2. Normalize: R/31.0, G/63.0, B/31.0 to get [0,1] range
// 3. Convert to [-1,1] range: value * 2.0 - 1.0
// 4. Apply 8x8 average pooling to reduce 32x32 to 4x4
// 5. Flatten to 48 features (4*4*3)
// 6. Pass through FC layers
""")
```

## Results of Code

```
Total parameters: 954
Starting training with RGB565 input...
Input format: uint16_t pInputBuffer[32*32] // RRRRRGGGGGGBBBBB
Epoch 1, Batch 100, Loss: 2.283
Epoch 1, Batch 200, Loss: 2.212
Epoch 1, Batch 300, Loss: 2.126
Epoch 2, Batch 100, Loss: 2.030
Epoch 2, Batch 200, Loss: 2.007
Epoch 2, Batch 300, Loss: 1.988
Epoch 3, Batch 100, Loss: 1.964
Epoch 3, Batch 200, Loss: 1.960
Epoch 3, Batch 300, Loss: 1.947
Epoch 4, Batch 100, Loss: 1.933
Epoch 4, Batch 200, Loss: 1.921
Epoch 4, Batch 300, Loss: 1.903
Epoch 5, Batch 100, Loss: 1.897
Epoch 5, Batch 200, Loss: 1.884
Epoch 5, Batch 300, Loss: 1.892
Epoch 6, Batch 100, Loss: 1.883
Epoch 6, Batch 200, Loss: 1.869
Epoch 6, Batch 300, Loss: 1.874
Epoch 7, Batch 100, Loss: 1.866
Epoch 7, Batch 200, Loss: 1.853
Epoch 7, Batch 300, Loss: 1.860
Epoch 8, Batch 100, Loss: 1.837
Epoch 8, Batch 200, Loss: 1.849
Epoch 8, Batch 300, Loss: 1.840
Epoch 9, Batch 100, Loss: 1.818
Epoch 9, Batch 200, Loss: 1.838
Epoch 9, Batch 300, Loss: 1.841
Epoch 10, Batch 100, Loss: 1.830
Epoch 10, Batch 200, Loss: 1.828
Epoch 10, Batch 300, Loss: 1.810
Epoch 11, Batch 100, Loss: 1.815
Epoch 11, Batch 200, Loss: 1.812
Epoch 11, Batch 300, Loss: 1.811
Epoch 12, Batch 100, Loss: 1.819
Epoch 12, Batch 200, Loss: 1.793
Epoch 12, Batch 300, Loss: 1.797
Epoch 13, Batch 100, Loss: 1.785
Epoch 13, Batch 200, Loss: 1.795
Epoch 13, Batch 300, Loss: 1.799
Epoch 14, Batch 100, Loss: 1.787
Epoch 14, Batch 200, Loss: 1.775
Epoch 14, Batch 300, Loss: 1.792
Epoch 15, Batch 100, Loss: 1.785
Epoch 15, Batch 200, Loss: 1.770
Epoch 15, Batch 300, Loss: 1.782
Epoch 16, Batch 100, Loss: 1.771
Epoch 16, Batch 200, Loss: 1.765
Epoch 16, Batch 300, Loss: 1.767
Epoch 17, Batch 100, Loss: 1.768
Epoch 17, Batch 200, Loss: 1.760
Epoch 17, Batch 300, Loss: 1.761
Epoch 18, Batch 100, Loss: 1.751
Epoch 18, Batch 200, Loss: 1.750
Epoch 18, Batch 300, Loss: 1.761
Epoch 19, Batch 100, Loss: 1.750
Epoch 19, Batch 200, Loss: 1.755
Epoch 19, Batch 300, Loss: 1.734
Epoch 20, Batch 100, Loss: 1.736
Epoch 20, Batch 200, Loss: 1.747
Epoch 20, Batch 300, Loss: 1.729
Test Accuracy: 38.06%

=== WEIGHTS FOR C CODE (RGB565 INPUT) ===
// Input format: uint16_t pInputBuffer[32*32]; // RRRRRGGGGGGBBBBB
// FC1 Weight [16][48]
float fc1_weight[16][48] = {
  {-0.015514f, -0.088211f, -0.039722f, 0.117178f, 0.017408f, 0.138527f, 0.053820f, 0.073875f, -0.005791f, 0.746798f, 0.657041f, 0.070830f, 0.225954f, -0.308882f, -0.235034f, 0.278682f, 0.113728f, 0.251466f, 0.153976f, 0.107867f, 0.006614f, 0.407585f, 0.484569f, 0.116353f, -0.430982f, -0.142089f, 0.037410f, -0.513894f, 0.355240f, -0.478123f, -0.383179f, 0.358733f, -0.174926f, -0.428504f, -0.368023f, -0.308149f, -0.037760f, 0.216601f, 0.298783f, 0.031938f, -0.220893f, 0.189898f, -0.028883f, -0.151448f, 0.704433f, -0.214535f, -0.153832f, 0.629968f},
  {0.211004f, 0.373176f, 0.451919f, 0.249522f, -0.051107f, 0.279318f, 0.238714f, -0.162176f, -0.248161f, -0.182048f, -0.144041f, -0.176025f, 0.223375f, 0.425620f, 0.353234f, 0.321384f, 0.304944f, 0.417434f, 0.178340f, 0.123304f, 0.194598f, 0.408501f, 0.408182f, 0.272740f, 0.342005f, 0.632221f, 0.591183f, 0.218385f, 0.063949f, 0.348312f, 0.176023f, 0.093819f, -0.502171f, -0.340777f, -0.340858f, -0.388133f, -0.367714f, -0.384479f, -0.379754f, -0.250282f, -0.483701f, -0.406614f, -0.254140f, -0.569033f, -0.957596f, -0.679201f, -0.645507f, -0.985158f},
  {-0.222305f, 0.357820f, 0.413368f, -0.328653f, -0.349465f, 0.638772f, 0.619809f, -0.313037f, -0.410935f, 0.206106f, -0.028145f, -0.137495f, -0.081299f, 0.426119f, 0.348921f, -0.085323f, -0.446056f, 0.167524f, -0.072052f, -0.446134f, -0.164663f, -0.084830f, 0.067498f, -0.188040f, -0.132758f, 0.160241f, 0.263640f, -0.043962f, -0.438418f, 0.345989f, 0.187574f, -0.196629f, -0.252832f, 0.396072f, 0.463655f, -0.135488f, -0.090061f, 0.282268f, 0.254534f, -0.035225f, -0.290519f, 0.424309f, 0.357411f, -0.305412f, -0.120640f, 0.177832f, 0.418994f, -0.110711f},
  {0.071822f, 0.018676f, -0.019438f, -0.098413f, -0.085624f, -0.975322f, -0.801716f, -0.008833f, 0.044639f, 0.135227f, -0.161836f, 0.058186f, -0.162301f, -0.256694f, -0.405506f, -0.162358f, -0.130927f, 0.069692f, 0.230814f, -0.052323f, -0.134171f, -0.086304f, -0.174287f, -0.111972f, -0.170096f, -0.090665f, -0.039175f, -0.125515f, -0.217727f, -0.298437f, -0.208359f, -0.094748f, -0.137765f, 0.331672f, 0.314422f, -0.179982f, 0.091008f, 0.855098f, 0.774743f, 0.039188f, 0.354933f, 0.961344f, 1.050579f, 0.360739f, 0.140772f, 0.349634f, 0.418244f, 0.088467f},
  {0.040732f, -0.099356f, 0.024710f, -0.110080f, -0.146270f, 0.057546f, -0.096038f, -0.004496f, -0.051815f, -0.139569f, 0.006397f, -0.040391f, 0.047810f, 0.088892f, -0.087647f, 0.115906f, -0.045255f, -0.033808f, 0.066430f, 0.106420f, 0.034320f, 0.116059f, -0.107708f, 0.024980f, 0.075650f, -0.092586f, 0.052835f, -0.001890f, -0.060490f, -0.147316f, 0.011828f, -0.113090f, -0.109265f, 0.118315f, -0.083997f, 0.007142f, 0.110117f, 0.033484f, -0.150746f, 0.117204f, -0.089864f, 0.062143f, 0.073794f, 0.074988f, -0.125091f, -0.128286f, -0.054639f, 0.033037f},
  {-0.199697f, -0.109135f, -0.176615f, -0.008060f, -0.331077f, 0.169463f, 0.161629f, -0.302794f, -0.114782f, -0.034521f, -0.010572f, -0.049234f, -0.376581f, 0.236895f, 0.349495f, -0.635496f, -0.189707f, -0.265898f, -0.157195f, -0.083742f, -0.052662f, -0.058757f, 0.070783f, 0.066302f, 0.433040f, 0.275497f, 0.263567f, 0.536427f, -0.142085f, 0.691105f, 0.545850f, -0.068363f, 0.283720f, 0.271818f, 0.215584f, 0.188327f, 0.383381f, -0.173238f, -0.082767f, 0.293119f, 0.446850f, -0.228042f, -0.314930f, 0.263006f, -0.077279f, 0.272382f, 0.118748f, -0.169581f},
  {0.214373f, -0.025929f, -0.041318f, 0.147353f, -0.130038f, 0.158919f, 0.100474f, 0.031474f, 0.069397f, 0.528361f, 0.536458f, 0.101750f, -0.118008f, -0.143879f, -0.132736f, -0.062153f, 0.471497f, 0.183725f, 0.158250f, 0.527949f, -0.327323f, -0.384158f, -0.323676f, -0.422274f, -0.358399f, -0.169322f, -0.088330f, -0.497008f, 0.032958f, -0.079783f, 0.004370f, -0.084969f, 0.924646f, 0.738420f, 0.653708f, 0.853857f, -0.163108f, 0.151562f, 0.057630f, -0.454387f, -0.763038f, -0.067637f, 0.097983f, -0.607820f, -0.409349f, -0.194184f, -0.179827f, -0.328531f},
  {-0.072560f, -0.449108f, -0.332057f, -0.077448f, 0.168459f, -0.244939f, -0.276485f, 0.076537f, -0.057319f, -0.567014f, -0.533343f, -0.186537f, 0.396644f, 0.204006f, 0.344079f, 0.471821f, 0.083795f, 0.145007f, 0.043014f, 0.049422f, -0.214537f, -0.010120f, 0.141963f, -0.172904f, -0.621180f, -0.902852f, -0.737517f, -0.642247f, 0.315803f, 0.504864f, 0.473638f, 0.169193f, 0.104541f, 0.419247f, 0.171063f, 0.106957f, 0.017110f, 1.011918f, 0.988502f, 0.006039f, -0.270292f, 0.082646f, -0.064731f, -0.269532f, -0.018728f, 0.403535f, 0.326500f, 0.101861f},
  {0.240925f, -0.203519f, -0.024748f, 0.236937f, -0.234005f, -0.478011f, -0.336528f, -0.224305f, -0.445075f, -0.140883f, -0.040994f, -0.132474f, -0.717484f, -0.153987f, -0.132936f, -0.622635f, 0.310577f, 0.023444f, 0.200146f, 0.409342f, 0.185330f, 0.277158f, 0.310101f, 0.143441f, 0.031285f, -0.091622f, 0.123097f, 0.105456f, -0.363075f, -0.210904f, -0.084533f, -0.349374f, 0.609830f, 0.196873f, 0.214657f, 0.498035f, 0.407198f, 0.124741f, 0.111405f, 0.323547f, 0.238275f, 0.097875f, 0.026391f, 0.296683f, 0.608882f, 0.482985f, 0.613027f, 0.445177f},
  {0.086272f, 0.253242f, 0.261644f, 0.174868f, -0.102999f, -0.177567f, -0.134918f, -0.077480f, -0.203921f, -0.522307f, -0.516808f, -0.165591f, 0.105598f, 0.448577f, 0.512045f, 0.181272f, -0.060691f, 0.120883f, 0.089306f, -0.134721f, 0.260052f, -0.147713f, -0.101962f, 0.116597f, 0.415860f, 0.220966f, 0.239734f, 0.500762f, 0.150335f, 0.354546f, 0.482494f, 0.250278f, -0.127161f, -0.120448f, -0.196991f, -0.102223f, 0.321876f, -0.580724f, -0.519657f, 0.200095f, 0.200625f, -0.202642f, -0.264198f, 0.398171f, -0.057850f, 0.330676f, 0.273455f, 0.031536f},
  {-0.168968f, -0.318655f, -0.346490f, -0.140375f, -0.117563f, -0.268782f, -0.322546f, -0.121755f, 0.168988f, -0.389575f, -0.436307f, 0.220088f, 0.035319f, 0.007174f, 0.274250f, 0.206930f, -0.220839f, -0.068462f, 0.058155f, -0.221650f, 0.178828f, -0.214629f, -0.153809f, 0.152701f, 0.525921f, 0.073650f, 0.006651f, 0.454731f, 0.141539f, 0.520735f, 0.423911f, 0.059093f, 0.037416f, 0.279260f, 0.415714f, 0.078216f, 0.035434f, 0.635053f, 0.639576f, 0.093937f, 0.093024f, 0.195854f, 0.284343f, 0.100367f, -0.613331f, -0.091776f, -0.109716f, -0.745495f},
  {0.404053f, 0.323262f, 0.438100f, 0.180114f, 0.063206f, 0.021977f, 0.228753f, -0.119276f, -0.374753f, 0.191532f, 0.349669f, -0.448280f, -0.214766f, 0.501495f, 0.370706f, -0.053476f, -0.230872f, -0.011565f, -0.123950f, -0.230391f, -0.143775f, 0.330952f, 0.211420f, -0.284699f, -0.411762f, 0.365196f, 0.242866f, -0.385704f, -0.497331f, -0.074382f, 0.041283f, -0.532233f, -0.135579f, -0.338449f, -0.392743f, -0.282020f, 0.174202f, -0.392250f, -0.399623f, -0.040153f, -0.010059f, 0.341575f, 0.276205f, -0.153545f, -0.000429f, 0.607686f, 0.444528f, -0.035515f},
  {0.382071f, 0.092013f, -0.026814f, 0.550570f, 0.536155f, -0.647196f, -0.531928f, 0.597304f, 0.049985f, -0.090576f, -0.115584f, 0.187777f, 0.161971f, 0.130998f, 0.188878f, 0.428405f, 0.179252f, -0.487070f, -0.415779f, 0.117987f, 0.008667f, -0.631478f, -0.531716f, -0.074481f, -0.395232f, -0.332236f, -0.275785f, -0.529268f, -0.180542f, -0.183867f, -0.224673f, -0.227140f, 0.246096f, -0.369772f, -0.075481f, 0.384951f, 0.199491f, -0.370598f, -0.192061f, 0.372276f, 0.024039f, 0.286117f, 0.231013f, 0.025684f, 0.202990f, 0.222482f, 0.280398f, 0.148571f},
  {0.052016f, 0.220755f, 0.290553f, 0.159888f, 0.438428f, 0.129905f, 0.183187f, 0.344451f, 0.462126f, 0.378238f, 0.314736f, 0.372762f, -0.238696f, 0.035853f, 0.021025f, -0.360933f, -0.319990f, -0.165550f, -0.195722f, -0.277554f, -0.428932f, 0.008132f, -0.002322f, -0.309206f, -0.337660f, -0.052944f, -0.088832f, -0.320356f, -0.325065f, -0.262780f, -0.289174f, -0.413987f, 0.003121f, 0.079504f, 0.156227f, -0.107279f, -0.128441f, 0.625755f, 0.574488f, 0.002390f, 0.177433f, 0.432526f, 0.478227f, 0.295062f, 0.180070f, 0.424153f, 0.310068f, 0.348168f},
  {-0.279661f, -0.264050f, -0.099585f, -0.222420f, -0.050651f, 0.424648f, 0.378430f, -0.130464f, -0.106392f, -0.417633f, -0.339635f, 0.024261f, 0.651164f, 0.428855f, 0.450871f, 0.638388f, 0.158199f, -0.123454f, -0.210617f, 0.098639f, 0.094983f, -0.385808f, -0.281911f, -0.045557f, 0.287846f, 0.162195f, 0.123049f, 0.362647f, 0.297611f, 0.287950f, 0.273753f, 0.160296f, 0.613140f, 0.376153f, 0.358526f, 0.569224f, -0.075544f, -0.203410f, -0.155166f, -0.079774f, -0.198296f, 0.101286f, -0.037826f, -0.180792f, -0.460364f, -0.187428f, -0.179483f, -0.591484f},
  {-0.110068f, -0.006522f, -0.051010f, 0.138700f, -0.136649f, 0.047110f, 0.042151f, -0.018639f, 0.026147f, 0.137480f, -0.138089f, -0.090427f, 0.079472f, -0.032448f, -0.005440f, -0.092454f, 0.089632f, -0.126070f, -0.114556f, 0.015970f, -0.032103f, -0.142440f, 0.083370f, 0.030317f, 0.091905f, -0.070675f, 0.077003f, -0.013161f, 0.002736f, -0.109350f, -0.050322f, -0.086388f, -0.078189f, -0.130184f, -0.071932f, -0.040279f, -0.142035f, 0.029959f, 0.073496f, -0.009214f, -0.059962f, -0.056162f, -0.003755f, -0.044888f, -0.050207f, 0.028748f, -0.019386f, 0.044535f},
};

// FC1 Bias [16]
float fc1_bias[16] = {
  0.567779f,
  0.437427f,
  -0.627171f,
  -0.140783f,
  -0.104224f,
  -0.226126f,
  -0.080582f,
  -0.192924f,
  -0.069846f,
  -0.014304f,
  0.013380f,
  0.209370f,
  0.405375f,
  -0.455073f,
  0.101191f,
  -0.103900f
};

// FC2 Weight [10][16]
float fc2_weight[10][16] = {
  {-0.744452f, 0.544372f, -0.980342f, 0.747298f, -0.067108f, 0.289003f, -0.066754f, 0.923619f, 0.347629f, -0.115694f, 0.676485f, -0.864936f, -0.606704f, 0.666978f, -0.091198f, 0.048809f},
  {0.869205f, -0.713689f, -1.736580f, 0.966881f, -0.075235f, -1.396598f, 0.486219f, 1.308736f, -0.537610f, -0.601178f, 0.276293f, -0.207768f, 0.742357f, 0.425722f, -0.883557f, 0.019747f},
  {-0.465126f, 0.991838f, -0.426623f, -0.046168f, -0.004108f, -0.130017f, -1.492390f, -0.893737f, 0.783141f, 0.303176f, 0.370492f, 0.135998f, -0.746880f, -0.759029f, 0.108544f, -0.076005f},
  {-0.186580f, -0.059687f, 0.588216f, -0.659725f, -0.027708f, 0.072135f, -0.575881f, -0.944694f, 0.047933f, 0.294929f, -0.736322f, 1.217845f, 1.008493f, 0.267033f, -0.176006f, -0.150949f},
  {-0.197132f, 0.539833f, -0.733224f, -0.399566f, 0.082942f, 0.553166f, -0.431201f, -0.427524f, -0.332439f, 0.423884f, 0.454473f, -0.019575f, -1.711268f, -1.000521f, 0.218615f, 0.203759f},
  {-0.226325f, 0.205934f, 1.515453f, -0.356280f, -0.208775f, 0.070336f, -0.623883f, -1.439874f, -0.246745f, 0.387001f, -0.485078f, 0.387739f, 0.909592f, 0.407128f, 0.020668f, -0.229357f},
  {0.493377f, 0.523884f, -0.134194f, -1.647887f, -0.006315f, 0.008508f, -1.331403f, -1.337732f, 0.309971f, 0.218799f, -1.261389f, 0.770246f, 0.019041f, -0.849159f, -0.049510f, 0.169802f},
  {-0.461043f, -0.058543f, 0.688392f, -0.752487f, -0.041477f, 0.386340f, 0.704788f, -0.777180f, -1.100999f, -0.054927f, 0.951896f, -0.869828f, 0.256950f, -0.295462f, 0.964235f, -0.027891f},
  {0.138712f, -1.939646f, -0.399666f, 0.182841f, 0.045925f, 0.607822f, 0.755379f, 0.185694f, 0.724303f, -0.673056f, -0.279680f, -0.087336f, -1.913650f, 0.305376f, -0.805153f, 0.100043f},
  {0.364368f, -1.037416f, 0.026690f, 0.310028f, -0.016057f, -1.162530f, 0.854022f, 0.596033f, -0.107574f, -1.040144f, -0.245129f, -0.777353f, 0.482361f, -0.031663f, 0.530030f, 0.241082f},
};

// FC2 Bias [10]
float fc2_bias[10] = {
  -0.892555f,
  0.312242f,
  0.144648f,
  0.116253f,
  0.409479f,
  -0.173313f,
  0.272274f,
  0.189534f,
  -0.202628f,
  0.027251f
};

=== TESTING RGB565 INPUT ===
Test RGB565 input shape: torch.Size([1, 32, 32])
Test RGB565 input dtype: torch.uint16
Sample RGB565 values: tensor([61592, 53516, 14380,  7319, 52306], dtype=torch.uint16)
PyTorch inference time with RGB565: 11.012 ms
Output shape: torch.Size([1, 10])
Predicted class: 2 (bird)

=== C CODE INTEGRATION EXAMPLE ===

// In your C code, you would have:
uint16_t pInputBuffer[32*32]; // RRRRRGGGGGGBBBBB format

// To use with this neural network:
// 1. The network expects RGB565 values in range [0, 65535]
// 2. Each pixel is stored as: (R5 << 11) | (G6 << 5) | B5
// 3. Array is organized as: [row 0 col 0, row 0 col 1, ..., row 31 col 31]

// Example RGB565 decoding in C:
void decode_rgb565(uint16_t rgb565, uint8_t* r, uint8_t* g, uint8_t* b) {
    *r = (rgb565 >> 11) & 0x1F;        // 5 bits, range 0-31
    *g = (rgb565 >> 5) & 0x3F;         // 6 bits, range 0-63  
    *b = rgb565 & 0x1F;                // 5 bits, range 0-31
}

// The neural network will internally:
// 1. Extract R, G, B components from each RGB565 value
// 2. Normalize: R/31.0, G/63.0, B/31.0 to get [0,1] range
// 3. Convert to [-1,1] range: value * 2.0 - 1.0
// 4. Apply 8x8 average pooling to reduce 32x32 to 4x4
// 5. Flatten to 48 features (4*4*3)
// 6. Pass through FC layers
```

## Explain Code

I'll explain the workflow of this Google Colab script that trains a lightweight neural network for CIFAR-10 image classification with RGB565 input.

### Overview

This script builds and trains a compact neural network (<1000 parameters) that classifies CIFAR-10 images while taking RGB565 format inputs instead of the standard RGB888 format. RGB565 is a common format in embedded systems and microcontrollers where memory is limited.

### Main Workflow

1. **Color Format Conversion**
   - The script includes functions to convert between RGB888 (standard 24-bit color) and RGB565 (16-bit color) formats
   - RGB565 uses 5 bits for red, 6 bits for green, and 5 bits for blue (hence the name)
2. **Neural Network Architecture**
   - Creates a very small neural network with only 954 parameters
   - Input: 32x32 RGB565 images
   - Uses 8x8 average pooling to reduce dimensions to 4x4
   - Flattens to 48 features (4×4×3)
   - Uses two fully connected layers: 48→16→10
3. **Dataset Preparation**
   - Loads CIFAR-10 dataset (10 classes: plane, car, bird, cat, etc.)
   - Wraps it in a custom dataset class that converts standard RGB888 images to RGB565 format
4. **Training Process**
   - Trains for 20 epochs using Adam optimizer
   - Uses cross-entropy loss for classification
   - Batch size of 128
   - Uses GPU if available
5. **Evaluation & Export**
   - Tests the model's accuracy on the test set
   - Extracts the trained weights and biases
   - Formats them for easy integration into C code
   - Shows an example of single inference with random RGB565 input
6. **C Integration Example**
   - Provides guidance on how to use the exported weights in embedded C code
   - Explains how to handle RGB565 format in C

This workflow is particularly useful for embedded systems developers who want to deploy image classification on resource-constrained devices like microcontrollers, where using the RGB565 color format can save memory while still maintaining reasonable accuracy.

# Embedded Part

## Neural Network from Scratch

### Code C

```c
// Neural network inference function (CIFAR-10) for RGB565 input
int cifar10_classify(uint16_t pInputBuffer[32*32], float *confidence)
{
    // Step 1: Extract RGB components and perform average pooling to reduce 32x32 to 4x4x3 = 48 features
    float pooled_features[48];
    
    for (int c = 0; c < 3; c++) { // For each channel (R, G, B)
        for (int y = 0; y < 4; y++) { // 4x4 output
            for (int x = 0; x < 4; x++) {
                float sum = 0.0f;
                // Average over 8x8 window
                for (int dy = 0; dy < 8; dy++) {
                    for (int dx = 0; dx < 8; dx++) {
                        int input_y = y * 8 + dy;
                        int input_x = x * 8 + dx;
                        int idx = input_y * 32 + input_x;
                        uint16_t rgb565 = pInputBuffer[idx];
                        
                        float channel_value;
                        if (c == 0) {
                            // Red channel (5 bits)
                            channel_value = ((rgb565 >> 11) & 0x1F) / 31.0f;
                        } else if (c == 1) {
                            // Green channel (6 bits)
                            channel_value = ((rgb565 >> 5) & 0x3F) / 63.0f;
                        } else {
                            // Blue channel (5 bits)
                            channel_value = (rgb565 & 0x1F) / 31.0f;
                        }
                        
                        sum += channel_value;
                    }
                }
                int output_idx = c * 16 + y * 4 + x; // 4x4x3 flattened
                pooled_features[output_idx] = (sum / 64.0f) * 2.0f - 1.0f; // Normalize to [-1,1]
            }
        }
    }

    // Step 2: First fully connected layer (48 -> 16)
    float fc1_output[16];
    for (int i = 0; i < 16; i++) {
        fc1_output[i] = fc1_bias[i];
        for (int j = 0; j < 48; j++) {
            fc1_output[i] += fc1_weight[i][j] * pooled_features[j];
        }
        // ReLU activation
        fc1_output[i] = fc1_output[i] > 0.0f ? fc1_output[i] : 0.0f;
    }

    // Step 3: Second fully connected layer (16 -> 10)
    float fc2_output[10];
    for (int i = 0; i < 10; i++) {
        fc2_output[i] = fc2_bias[i];
        for (int j = 0; j < 16; j++) {
            fc2_output[i] += fc2_weight[i][j] * fc1_output[j];
        }
    }

    // Step 4: Softmax (find max for numerical stability)
    flx
    for (int i = 1; i < 10; i++) {
        if (fc2_output[i] > max_val) {
            max_val = fc2_output[i];
        }
    }

    // Calculate softmax probabilities
    float sum = 0.0f;
    float probabilities[10];
    for (int i = 0; i < 10; i++) {
        probabilities[i] = expf(fc2_output[i] - max_val);
        sum += probabilities[i];
    }

    // Normalize probabilities
    for (int i = 0; i < 10; i++) {
        probabilities[i] /= sum;
    }

    // Find the class with highest probability
    int predicted_class = 0;
    float max_confidence = probabilities[0];
    for (int i = 1; i < 10; i++) {
        if (probabilities[i] > max_confidence) {
            max_confidence = probabilities[i];
            predicted_class = i;
        }
    }

    // Return the confidence score
    *confidence = max_confidence;

    return predicted_class;
}
```

### Explain Workflow

The `cifar10_classify` function takes RGB565 image data and processes it through a simple neural network to classify the image into one of 10 CIFAR-10 categories. Here's the step-by-step workflow:

1. **Input Processing & Average Pooling**

- **Input**: A 32×32 image in RGB565 format (single 16-bit value per pixel) stored in a 1D array
- **Process**:
  - For each 8×8 block in the image, the function calculates the average value for each color channel
  - RGB565 values are unpacked into separate R, G, B components
  - Red and blue values (5 bits) are normalized by dividing by 31.0
  - Green values (6 bits) are normalized by dividing by 63.0
  - These averages reduce the 32×32 image to a 4×4 grid of RGB values
  - Values are normalized from [0,1] to [-1,1] range for neural network processing
- **Output**: 48 floating-point features (4×4×3 channels)

2. **First Fully Connected Layer (FC1)**

- **Input**: 48 features from the pooling step
- **Process**:
  - Matrix multiplication with weights: 48 inputs × 16 outputs
  - Add bias values to each output
  - Apply ReLU activation function (max(0, x))
- **Output**: 16 intermediate features

3. **Second Fully Connected Layer (FC2)**

- **Input**: 16 features from FC1
- **Process**:
  - Matrix multiplication with weights: 16 inputs × 10 outputs
  - Add bias values to each output
- **Output**: 10 raw logits (one per CIFAR-10 class)

4. **Softmax & Classification**

- **Input**: 10 logits from FC2
- **Process**:
  - Find maximum value for numerical stability
  - Calculate exponential values (adjusted by max)
  - Normalize to get probability distribution
  - Identify class with highest probability
- **Output**:
  - Predicted class index (0-9)
  - Confidence score (probability of the predicted class)

Key Optimizations

1. **Memory Efficiency**: Using RGB565 format reduces memory footprint by ~33% compared to RGB888
2. **Dimensionality Reduction**: Average pooling drastically reduces feature space from 32×32×3=3072 to just 48 features
3. **Compact Architecture**: Only 954 parameters total, making it suitable for memory-constrained devices
4. **Numerical Stability**: The softmax implementation includes the max-subtraction trick to prevent overflow

This lightweight neural network is designed for embedded systems while still maintaining reasonable classification accuracy on the CIFAR-10 dataset.

# Results

## Prepare for Testing

```python
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Create directories
!mkdir -p cifar10_images
for i in range(10):  # CIFAR-10 has 10 classes
    !mkdir -p cifar10_images/class_{i}

# Load CIFAR-10 dataset
transform = transforms.ToTensor()
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Counters for each class
class_counts = [0] * 10

# Iterate through the dataset and save images
for idx, (image, label) in enumerate(dataset):
    if class_counts[label] < 100:
        # Convert tensor to numpy array and change shape from CxHxW to HxWxC
        img_np = image.permute(1, 2, 0).numpy()
        
        # Convert to PIL Image and save
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save(f'cifar10_images/class_{label}/image_{class_counts[label]}.png')
        
        class_counts[label] += 1
    
    # Stop if we've collected 100 images for each class
    if all(count >= 100 for count in class_counts):
        break

# Verify the counts
print("Number of images saved per class:")
for i, name in enumerate(class_names):
    print(f"{name}: {class_counts[i]}")

# Zip the folder for download
!zip -r cifar10_images.zip cifar10_images/

# Display sample images
plt.figure(figsize=(10, 10))
for i in range(10):
    img = Image.open(f'cifar10_images/class_{i}/image_0.png')
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(class_names[i])
    plt.axis('off')
plt.show()
```

Result:
![image](https://github.com/user-attachments/assets/72dbcc30-7d2a-4331-9cb0-1c2e3742ed25)
