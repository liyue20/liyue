import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_se import mobile_vit_xx_small as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载图像
    img_path = "/successFUL/MobileViT/10.jpg"
    assert os.path.exists(img_path), "文件: '{}' 不存在.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    # [N, C, H, W]
    img = data_transform(img)

    # 在批次维度上扩展
    img = torch.unsqueeze(img, dim=0)

    # 加载头部姿态数据（假设它是pitch、roll、yaw）
    # 0.jpg	3.972348999	2.243744197	5.919841496
    # 1.jpg	4.379605077	0.998174681	5.652957738
    # 10.jpg	4.032361652	1.481964145	5.521542748


    head_pose = torch.tensor([[4.032361652, 1.481964145, 5.521542748]])  # 用实际的头部姿态值替换这里

    # 结合图像和头部姿态数据
    input_data = {"image": img, "head_pose": head_pose}

    # 读取类别字典
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "文件: '{}' 不存在.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = create_model(num_classes=6).to(device)

    # 加载模型权重
    model_weight_path = "runs/weight/best_model.pth"
    # strict=False 解决键不匹配的问题
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # 预测类别
        output = torch.squeeze(model(input_data["image"].to(device), input_data["head_pose"].to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "类别: {}   概率: {:.3}".format(class_indict[str(predict_cla)],
                                                predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("类别: {:10}   概率: {:.3}".format(class_indict[str(i)],
                                                 predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
