import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
class Graph():
    def __init__(self, layout='openpose', strategy='spatial'):
        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18  # or whatever number matches your data
            print(f"Graph: num_node = {self.num_node}")
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError("Do not support this layout.")

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.num_node - 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do not support this strategy.")

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), 
                              padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A
class STGCN(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        # Replace BatchNorm with GroupNorm
        self.data_bn = nn.GroupNorm(num_groups=4, num_channels=in_channels * A.size(1))
        
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, stride=1, residual=False),
            STGCNBlock(64, 64, kernel_size, stride=1),
            STGCNBlock(64, 64, kernel_size, stride=1),
            STGCNBlock(64, 64, kernel_size, stride=1),
            STGCNBlock(64, 128, kernel_size, stride=2),
            STGCNBlock(128, 128, kernel_size, stride=1),
            STGCNBlock(128, 128, kernel_size, stride=1),
            STGCNBlock(128, 256, kernel_size, stride=2),
            STGCNBlock(256, 256, kernel_size, stride=1),
            STGCNBlock(256, 256, kernel_size, stride=1)
        ))
        
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
    
    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)
        
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        
        return x


class ParkinsonsDetectionModel(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()
        self.st_gcn = STGCN(in_channels, graph_args, edge_importance_weighting, **kwargs)
        self.angle_embedding = nn.Embedding(3, 64)  # 3 angles (0, 90, 180), embedding size 64
        self.fc1 = nn.Linear(256 + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, angle):
        x = self.st_gcn(x)
        angle = torch.div(angle, 90, rounding_mode='floor').long()  # Convert angles to 0, 1, 2
        angle_embed = self.angle_embedding(angle)
        combined = torch.cat([x, angle_embed], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)  # Remove the last dimension if it's 1
# Example usage
graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
model = ParkinsonsDetectionModel(in_channels=2, graph_args=graph_args, edge_importance_weighting=True)

# Example input
batch_size = 2
num_channels = 2  # X and Y coordinates
num_frames = 300
num_joints = 18
x = torch.randn(batch_size, num_channels, num_frames, num_joints)
angle = torch.LongTensor([0, 1])  # Example angles for each sample in the batch

# Forward pass
output = model(x, angle)
print(output.shape)  # Should be [batch_size, 1]
angle
print(x)
x[0][0][0]
# Load the CSV file
df = pd.read_csv("../output_data/output_2D.csv")

# Convert the vector column from string to numpy array
df['vector'] = df['vector'].apply(eval).apply(np.array)

df
#get max length of the vectors
max_length = df['vector'].apply(len).max()

# Pad the vectors to the max length
df['vector'] = df['vector'].apply(lambda x: np.pad(x, ((0, max_length - len(x)), (0, 0))))
df
def parse_vector(vector):
    vector_array = np.array(vector)
    # Ensure we have exactly 34 joints with 2 coordinates each
    if vector_array.shape[0] != 18 or vector_array.shape[1] != 2:
        padded = np.zeros((18, 2))
        padded[:min(vector_array.shape[0], 18), :min(vector_array.shape[1], 2)] = vector_array[:18, :2]
        return padded
    return vector_array


# Parse and stack the pose data
pose_data = np.stack(df['vector'].apply(parse_vector).values)

# Reshape to [N, C, T, V]
N, V, C = pose_data.shape
T = 1  # Assuming each sample is a single frame
pose_data = pose_data.transpose(0, 2, 1).reshape(N, C, T, V)

print("Pose data shape after reshaping:", pose_data.shape)

# Convert to torch tensors
pose_data_tensor = torch.FloatTensor(pose_data)
angles_tensor = torch.LongTensor(df['angle'].values)
labels_tensor = torch.FloatTensor(df['parkinson'].values)

print("Final pose data tensor shape:", pose_data_tensor.shape)
print("Pose data shape:", pose_data_tensor.shape)
pose_data_tensor[0][0][0]
graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
in_channels = pose_data_tensor.shape[1]
model = ParkinsonsDetectionModel(in_channels=in_channels, graph_args=graph_args, edge_importance_weighting=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
pose_data_tensor = pose_data_tensor.to(device)
angles_tensor = angles_tensor.to(device)
labels_tensor = labels_tensor.to(device)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss instead of BCELoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3
batch_size = 10  # or whatever size works for your memory constraints


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(0, pose_data_tensor.size(0), batch_size):
        # Get batch
        batch_pose = pose_data_tensor[i:i+batch_size]
        batch_angles = angles_tensor[i:i+batch_size]
        batch_labels = labels_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(batch_pose, batch_angles)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (pose_data_tensor.size(0) // batch_size)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
# For inference:
model.eval()
with torch.no_grad():
    outputs = model(pose_data_tensor, angles_tensor)
    predicted = (outputs.squeeze() > 0.5).float()  # Threshold at 0.5 for binary classification
    accuracy = (predicted == labels_tensor).float().mean()
    print("Accuracy:", accuracy.item())