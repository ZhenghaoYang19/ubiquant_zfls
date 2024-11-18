import socketio
import time
import numpy as np
import os
from pprint import pprint
import socketio.exceptions
import torch
from models.resnet import resnet18
from torchvision import transforms
from utils.data_stats import load_dataset_stats, calculate_dataset_stats
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class PatchDataset(Dataset):
    def __init__(self, patches, transform=None):
        """
        Args:
            patches: shape (N, 50, 50, 3) 的numpy数组
            transform: 图像变换
        """
        self.patches = patches
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch

class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images_path = os.path.join('jk_zfls', 'round0_train')
        # 尝试加载已保存的数据集统计信息，如果不存在则重新计算
        try:
            self.mean, self.std = load_dataset_stats()
            print("Loaded pre-calculated dataset statistics")
        except FileNotFoundError:
            print("FileNotFound, Calculating dataset statistics...")
            self.mean, self.std = calculate_dataset_stats(images_path)
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # 加载模型
        self.model = resnet18(num_classes=20)
        checkpoint = torch.load('models/best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载OpenMax模型
        self.openmax = torch.load('models/best_openmax.pth')
        self.threshold = 0.08  # 未知类别判断阈值
        
    def resnet_predict(self, patches):
        """
        仅使用ResNet模型预测图像块的类别（不使用MetaMax）
        Args:
            patches: shape (N, 50, 50, 3) 的torch tensor
        Returns:
            predictions: shape (N,) 的类别预测结果
        """
        # 创建数据集和数据加载器
        dataset = PatchDataset(patches, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                # 移动到设备
                batch = batch.to(self.device)
                
                # 获取logits并直接预测
                logits, _ = self.model(batch, return_features=True)
                _, batch_predictions = logits.max(1)
                predictions.append(batch_predictions.cpu())
        
        return torch.cat(predictions)
    
    def predict(self, patches):
        """
        使用OpenMax预测图像块的类别
        Args:
            patches: shape (N, 50, 50, 3) 的torch tensor
        Returns:
            predictions: shape (N,) 的类别预测结果
        """
        # 创建数据集和数据加载器
        dataset = PatchDataset(patches, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                # 移动到设备
                batch = batch.to(self.device)
                
                # 获取特征和logits
                logits, features = self.model(batch, return_features=True)
                
                # 使用OpenMax进行预测
                openmax_probs = self.openmax.predict(features, logits)
                
                # 使用阈值判断未知类别
                max_probs, batch_predictions = torch.max(openmax_probs[:, :-1], dim=1)
                batch_predictions[max_probs < self.threshold] = 20
                predictions.append(batch_predictions.cpu())
        
        return torch.cat(predictions)


def action_policy(action_shape):
    # 0: down, loc+=[1,0]
    # 1: right, loc+=[0,1]
    # 2: up, loc+=[-1,0]
    # 3: left, loc+=[0,-1]
    # 4: collect
    return np.random.randint(action_shape)

def recognition(img):
    """
    Args:
        img: shape [600,600,3] 的list，RGB格式
    Returns:
        grid: (12,12) 的numpy数组
    """
    if not hasattr(recognition, 'classifier'):
        recognition.classifier = ImageClassifier()
    
    # 先转换为numpy数组
    img = np.array(img, dtype=np.uint8)
    # 将图像分割成网格
    patches = []
    tile_size = 50
    
    for i in range(12):
        for j in range(12):
            patch = img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            patches.append(patch)
    
    patches = np.array(patches)
    # 获取预测结果
    # predictions = recognition.classifier.resnet_predict(patches)
    predictions = recognition.classifier.predict(patches)
    # 重塑为12x12网格
    grid = predictions.reshape(12, 12)
    
    return grid.numpy()


def team_play_game(team_id, game_type, game_data_id, ip, port):
    sio = socketio.Client(request_timeout=60)
    grid = None
    begin = game_type + game_data_id
    @sio.event
    def connect():
        # print(f"Connected to server, game_type: {game_type}, game data id: {begin}")
        pass
    @sio.event
    def disconnect():
        # print(f"End game {begin}, disconnected from server")
        pass
    @sio.event
    def connect_error(data):
        print('Connect error', data)
    @sio.event
    def disconnect_error(data):
        print('Disconnect error', data)
    @sio.event
    def response(data):
        nonlocal grid
        if 'error' in data:
            print(data['error'])
            sio.disconnect()
        else:
            try:
                if data['rounds'] == 0:
                    print(f"Team {data['team_id']} begin game {data['game_id']}")
                is_end = data.get('is_end', False)
                score = data['score']
                bag = data['bag']
                loc = data['loc']
                game_id = data['game_id']
                os.makedirs(f'./{data["team_id"]}/', exist_ok=True)
                send_data = {'team_id': data['team_id'], 'game_id': game_id}
                if data['rounds']==0:
                    if (game_type == 'a'):
                        grid = np.array(data['grid'], dtype=int)
                    if (game_type == '2'):
                        grid = recognition(data['img'])
                        send_data['grid_pred'] = grid.tolist()
                score_npy = f'./{data["team_id"]}/{data["game_id"]}_score.npy'
                if os.path.exists(score_npy):
                    prev_score = np.load(score_npy)
                else:
                    prev_score = np.array(0.0)
                np.save(score_npy, prev_score + score)
                if is_end:
                    print(f"Team {data['team_id']} end game {data['game_id']}, cum_score: {prev_score + score:.2f}")
                    if game_type == '2':
                        print(f'Recognition acc on this game fig: {data["acc"]}')
                    sio.disconnect()
                else:
                    action = action_policy(5)
                    if action == 4:
                        grid[loc[0], loc[1]] = -1
                    send_data['action'] = action
                    if sio.connected:
                        sio.emit('continue', send_data)
                    else:
                        print('sio not connected')
            except Exception as e:
                print(f'{e}')
                sio.disconnect()
    try:
        # 连接到服务器
        sio.connect(f'http://{ip}:{port}/', wait_timeout=30)
        # 发送消息到服务器
        message = {'team_id': team_id, 'begin': begin}
        sio.emit('begin', message)
        sio.wait()
    except socketio.exceptions.ConnectionError as e:
        print('Connection Error')
        sio.disconnect()
    except Exception as e:
        print(f'Exception: {e}')
        sio.disconnect()
    finally:
        #print('end team play game')
        pass


if __name__ == '__main__':
    team_id = f'ewv9ssdcuvg6'
    ip = '69.230.243.237'
    port = '8086'
    # game_type must be in ['2', 'a'], '2' for full game and recognition only, 'a' for action_only
    game_type = '2'
    
    # 初赛的第1阶段，game_data_id  must be in ['00000', '00001', ..., '00099']
    # 初赛的终榜阶段，game_data_id  must be in ['00000', '00001', ..., '00199']
    game_data_id = [f'{i:05}' for i in range(0, 40)]
    st = time.time()
    for gdi in game_data_id:
        team_play_game(team_id, game_type, gdi, ip, port)
    print(f'Total time: {(time.time()-st):.1f}s')
