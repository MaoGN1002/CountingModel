import torch
import os
import argparse
from datasets.crowd import Crowd
from models.fusion import FusionModel
from utils.evaluation import eval_game, eval_relative
import numpy as np

parser = argparse.ArgumentParser(description='Test')
# parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
#                         help='training data directory')
# parser.add_argument('--save-dir', default='/home/teddy/vgg',
#                         help='model directory')
# parser.add_argument('--model', default='best_model_17.pth'
#                     , help='model name')

parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datas', 'bayes-RGBT-CC-V2'),
                        help='training data directory')
parser.add_argument('--save-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'datas','0202-013003'),
                        help='model directory')

# parser.add_argument('--save-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datas', 'bayes-RGBT-CC-V2','res'),
#                         help='model directory')
                        


# 这里看一下名字对不对
parser.add_argument('--model', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datas', '0202-013003','best_model.pth')
                    , help='model name')

parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':
    datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = FusionModel()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0


    # 创建 outputs 文件夹
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datas', 'bayes-RGBT-CC-V2','outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error


        save_outputs = False
        if save_outputs:
            # 保存人群计数结果为 .npy 文件
            output_numpy = outputs.cpu().numpy()
            # 假设 name 是类似 1234_RGB 的形式，将其改为 1234_GT
            base_name = name[0].split('_RGB')[0]
            npy_save_path = os.path.join(outputs_dir, f'{base_name}_GT.npy')
            np.save(npy_save_path, output_numpy)


    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f} Re {relative:.4f}, '.\
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

    print(log_str)

