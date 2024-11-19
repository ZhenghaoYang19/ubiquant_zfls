import os
import shutil
from pathlib import Path

def move_images():
    # 基础路径
    eval_dir = Path('jk_zfls/round0_eval')
    train_dir = Path('jk_zfls/round0_train')
    
    
    # 处理前20个类别
    for class_idx in range(20):
        # 使用zfill(2)来生成带前导零的类别名称（00, 01, 02...）
        class_name = str(class_idx).zfill(2)
        eval_class_dir = eval_dir / class_name
        train_class_dir = train_dir / class_name
        
        # 获取评估集中该类别的所有图片
        if not eval_class_dir.exists():
            print(f"Warning: Directory {eval_class_dir} does not exist")
            continue
            
        # 修改这里：获取label编号在200-239之间的图片
        images = []
        for i in range(200, 240):  # 200到239
            img_name = f"label{class_name}_{str(i).zfill(3)}.jpg"
            img_path = eval_class_dir / img_name
            if img_path.exists():
                images.append(img_path)
        
        # 移动图片
        for img_path in images:
            target_path = train_class_dir
            try:
                shutil.move(str(img_path), str(target_path))
                print(f"Moved {img_path.name} to {target_path}")
            except Exception as e:
                print(f"Error moving {img_path}: {e}")

if __name__ == '__main__':
    move_images()
    print("Image moving completed!")
