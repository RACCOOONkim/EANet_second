import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, augmentation
from utils.vis import vis_keypoints_with_skeleton
from utils.human_models import MANO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=str, default='29', dest='test_epoch')
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--input', type=str, default='example_image1.png', dest='input')
    args = parser.parse_args()

    # GPU 설정 검증
    if not args.gpu_ids:
        assert 0, print("적절한 GPU ID를 설정해주세요.")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

# 인자 파싱
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
model = get_model('test')
model = DataParallel(model).cuda()
model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % int(args.test_epoch))
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# 입력 이미지 준비
transform = transforms.ToTensor()
img = load_img(args.input)
height, width = img.shape[:2]
bbox = [0, 0, width, height]
bbox = process_bbox(bbox, width, height)
img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, 'test')
img = transform(img.astype(np.float32))/255.
img = img.cuda()[None,:,:,:]
inputs = {'img': img}
targets = {}
meta_info = {}

# 모델 실행 및 결과 처리
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')

# 이미지에 스켈레톤 그리기
img = (img[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
mano = MANO()
hand_joints = out['hand_joints'][0].cpu().numpy()  # 손 관절 좌표 추정
hand_lines = mano.sh_skeleton  # 손 관절 연결 정보
skeleton_image = vis_keypoints_with_skeleton(img, hand_joints, hand_lines)

# 결과 이미지 저장
cv2.imwrite('output_image.png', skeleton_image)
