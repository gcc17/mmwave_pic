import matplotlib.pyplot as plt
import torch
import numpy as np
from mmWaveDataset import MMBody, MiliPointPose, mmPoseNLP
import os
import plotly.graph_objs as go

def plot_3d_graph(tensor1, tensor2, elev=-45, azim=-135, roll=45, save_path=None):
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.numpy()
    if torch.is_tensor(tensor2):
        tensor2 = tensor2.numpy()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x1, z1, y1 = tensor1[:, 0], tensor1[:, 2], -tensor1[:, 1]
    x2, z2, y2 = tensor2[:, 0], tensor2[:, 2], -tensor2[:, 1]
    
    ax.scatter(x1, z1, y1, c='b', marker='o', label='joints')
    for i in range(len(x1)):
        ax.text(x1[i], z1[i], y1[i], str(i), size=10, zorder=1, color='k')
    
    ax.scatter(x2, z2, y2, c='green', marker='o', label='mmwave points')
    
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], 
                          [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]], dtype=torch.long)
    # Plotting lines based on the edges
    for edge in edges:
        start_node = tensor1[edge[0]]
        end_node = tensor1[edge[1]]
        ax.plot([start_node[0], end_node[0]],
                [start_node[2], end_node[2]],
                [-start_node[1], -end_node[1]], c='b', linestyle='-', linewidth=2)
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Z-axis')
    ax.set_zlabel('Y-axis')
    # Invert the y-axis
    ax.invert_yaxis()
    
    # Adjusting the angle of view
    ax.view_init(elev=elev, azim=azim, roll=roll)

    # Save the plot to a file if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()



# 25 keypoints
def plot_3d_nlp(tensor1):
    if torch.is_tensor(tensor1):
        tensor1 = tensor1.numpy()
        
    x, y, z = tensor1[:, 0], tensor1[:, 1], tensor1[:, 2]
    print(x, y, z)
    text_labels = [f'{i}' for i in range(len(x))]
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+text',  # Add text to the markers
        text=text_labels,  # Text labels for each point
        textposition='top center',  # Position of the text labels
        marker=dict(
            size=4,
            color=z,  # set color to z values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])

    # Update layout for better visualization
    fig.update_layout(
        title='Interactive 3D Scatter Plot',
        scene=dict(
            xaxis_title='X axis',
            yaxis_title='Y axis',
            zaxis_title='Z axis'
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # Show the plot
    fig.show()




if __name__ == '__main__':
    # root = '../../mmBody'
    # mmbody_dataset = MMBody(root, split='train', test_scenario=['lab1', 'lab2'], normalized=True, device='cuda')
    # image_dir = 'images'
    # os.makedirs(image_dir, exist_ok=True)
    # for i in range(len(mmbody_dataset)):
    #     input, label = mmbody_dataset.__getitem__(i)
    #     label = label - label[:1, :]
    #     plot_3d_graph(label, input, save_path=os.path.join(image_dir, f'{i}.png'))

    # raw_data_dir = '../../MiliPoint/data/raw'
    # mili_dataset = MiliPointPose(raw_data_dir, split='train', device='cuda')
    # image_dir = 'images_mili'
    # os.makedirs(image_dir, exist_ok=True)
    # for i in range(len(mili_dataset)):
    #     input, label = mili_dataset.__getitem__(i)
    #     label = label - label[:1, :]
    #     plot_3d_graph(label, input, save_path=os.path.join(image_dir, f'{i}.png'))

    # train_data1 = np.load(os.path.join('merged_data', 'frames1_test.npy'))
    # print(train_data1.shape)
    # plot_3d_nlp(train_data1[0])
    
    train_data_path = os.path.join('merged_data', 'frames4_test.npy')
    pose_dataset = mmPoseNLP(train_data_path)
    image_dir = 'images_poseNLP4'
    os.makedirs(image_dir, exist_ok=True)
    for i in range(10):
        input, label = pose_dataset.__getitem__(i)
        plot_3d_graph(label, input, save_path=os.path.join(image_dir, f'{i}.png'))