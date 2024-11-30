import socketio
import time
import numpy as np
import os
import socketio.exceptions
from models.resnet import ImageClassifier
from search import search, adjust_grid

actions = []
def action_policy(grid=None, loc=None, rounds=0):
    """
    Args:
        grid: 当前网格状态
        loc: 当前位置
        rounds: 当前回合数
    Returns:
        action: 下一步动作
    """
    # 第0回合时计算整个行动序列
    
    return None

def recognition(img):
    """
    Args:
        img: shape [600,600,3] 的list，RGB格式
    Returns:
        grid: (12,12) 的numpy数组
    """
    if not hasattr(recognition, 'classifier'):
        recognition.classifier = ImageClassifier(model_type='resnet18', model_path='models/best_model_99.92_02.pth', openmax_path='models/best_openmax_95.62_02.pth', multiplier=0.6)
    
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
    predictions, openmax_probs = recognition.classifier.predict(patches)
    # 重塑为12x12网格
    grid = adjust_grid(predictions.cpu().numpy(), openmax_probs.cpu().numpy())
    # rows_without_high_confidence = np.all(openmax_probs < 0.9, axis=1)
    # indices = np.where(rows_without_high_confidence)[0]
    # class_counts = np.bincount(grid.flatten(), minlength=21)

    return grid, openmax_probs


def team_play_game(team_id, game_type, game_data_id, ip, port):
    sio = socketio.Client(request_timeout=60)
    grid = None
    begin = game_type + game_data_id
    @sio.event
    def connect():
        print(f"Connected to server, game_type: {game_type}, game data id: {begin}")
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
                        grid, openmax_probs = recognition(data['img'])
                        print('Recognition Finished!')
                        send_data['grid_pred'] = grid.tolist()
                        
                    # 使用 search 函数替代直接使用 Algorithm_Agent
                    actions[:] = search(grid=grid, loc=loc, pred_grid=grid, pred_loc=loc)[:]
                
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
                    print(f'time penalty:{data["time penalty"]}')
                    sio.disconnect()
                else:
                    action = actions.pop(0)
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
        # print('end team play game')
        pass


if __name__ == '__main__':
    team_id = f'ewv9ssdcuvg6'
    ip = '69.230.243.237'
    port = '8086'
    # game_type must be in ['2', 'a'], '2' for full game and recognition only, 'a' for action_only
    game_type = '2'
    
    # 初赛的第1阶段，game_data_id  must be in ['00000', '00001', ..., '00099']
    # 初赛的终榜阶段，game_data_id  must be in ['00000', '00001', ..., '00199']
    game_data_id = [f'{i:05}' for i in range(50, 100)]
    st = time.time()
    for gdi in game_data_id:
        team_play_game(team_id, game_type, gdi, ip, port)
    print(f'Total time: {(time.time()-st):.1f}s')
