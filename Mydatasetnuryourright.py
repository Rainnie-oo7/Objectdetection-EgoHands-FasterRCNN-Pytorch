import os.path as osp
import os
import glob
import torch
import cv2
import scipy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops import masks_to_boxes
import torchvision
from triton.language import tensor


def extract_poly(file_path):
    mat_data = scipy.io.loadmat(file_path)
    data = mat_data['polygons']
    poly = [[yr] for yr in zip(data['yourright'][0])]

    return poly

def polygons_to_binary_mask(polygon, width, height):
    blk_img = Image.new("L", (width, height), 0)
    polygon_list = polygon.flatten().tolist()

    if len(polygon_list) == 0:
        return blk_img
    ImageDraw.Draw(blk_img).polygon(polygon_list, outline=1, fill=1)
    mask = np.array(blk_img)

    return mask

def beatboxing(mask):
    x_values = [x for x, y in mask]
    y_values = [y for x, y in mask]

    # Berechne die Minimal- und Maximalwerte
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    # Ausgabe
    # print("min_x:", min_x)
    # print("max_x:", max_x)
    # print("min_y:", min_y)
    # print("max_y:", max_y)
    tupel1 = (min_x, min_y)
    tupel2 = (max_x, max_y)
    bbmatrix = np.array([tupel1, tupel2])
    # print("das ist bbmatrix:", bbmatrix)
    return bbmatrix

# def get_box(masks):
#     for m in masks:
#     # for i in range(num_objs):
#     #     pos = np.nonzero(masks[i])
#         ''' Get the bounding box of a given mask '''
#         pos = np.where(m)   # find out the position where a_mask=1
#         # print("postype", type(pos)) #Tupel
#         xmin = np.min(pos[1])  # min pos will give min co-ordinate
#         xmax = np.max(pos[1])   # max-position give max co-ordinate
#         ymin = np.min(pos[0])
#         ymax = np.max(pos[0])
#         return (xmin, ymin, xmax, ymax)

# def getboxes(masks):
#     boxes_appended = []
#     # maske = torch.tensor([])
#     #Um die Boxen-Tensors zusammenzuführen, weil nur das Durchlaufen einzelner Masken (=einzelner Hände) ging. Um jeder Hand's Box in einer Variable abzuspeichern
#     for i, m in enumerate(masks):
#         boxes_appended.append(get_box(m))   #[[118, 661, 290, 716], [645, 574, 870, 719], [593, 360, 774, 498], [411, 315, 532, 411]]
#
#     return boxes_appended


def extractboxes_from_dict(boxes_appended):
    tensors = [torch.tensor(value) for value in boxes_appended.values()]
    boxes_tensor = torch.stack(tensors)
    # print("Shape:", boxes_tensor.shape) #4, 1, 4
    #boxes_tensor = boxes_tensor.reshape(4, 4)
    boxes_tensor = boxes_tensor.squeeze(1)
    # print("Shape:", boxes_tensor.shape) #4, 4
    return boxes_tensor

class Mydataset(torch.utils.data.Dataset):

    #path = osp.normpath(osp.join(osp.dirname(__file__), "egohands_dir"))
    # files = glob.glob(os.path.join('path', 'to', 'dir', '*.jpg'))
    #print(path)
    def __init__(self, _, transforms=None):
        self.imgs = []
        self.polys = []
        self.transforms = transforms
        self.path = osp.normpath(osp.join(osp.dirname(__file__), "data"))

        for dir in os.listdir(self.path):
            full_dir_path = os.path.join(self.path, dir)
            if os.path.isdir(full_dir_path):
                files = sorted(os.listdir(full_dir_path))
                for file in files:
                    if file.startswith('frame'):
                        img_path = os.path.join(full_dir_path, file)
                        self.imgs.append(img_path)  # Append image to the list
                    elif file.startswith('polygons'):
                        file_path = os.path.join(full_dir_path, file)
                        self.polys  = self.polys  + extract_poly(file_path)



    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])  # Load image
        img = tv_tensors.Image(img)
        img = img.permute(2, 0, 1)
        #ValueError: images is expected to be a list of 3d tensors of shape [C, H, W], got torch.Size([720, 1280])
        your_right = self.polys[idx] # yourright 1 Maske, zu dem bild(idx)
        your_right = your_right[0][0]

        your_right_mask = polygons_to_binary_mask(your_right, 1280, 720)
        masks = tv_tensors.Mask(your_right_mask)
        ##

        bbmatrixyourright = beatboxing(your_right)

        tensor_labels = torch.ones((1,), dtype=torch.int64)
        tensor_labels = tensor_labels.squeeze()
        ##

        image_id = idx
        ##

        area = [(int(bbmatrixyourright[0,0]) - int(bbmatrixyourright[0,1])) * (int(bbmatrixyourright[1,0]) - int(bbmatrixyourright[1,1]))]
        tensor_area = torch.tensor(area)
        tensor_area = tensor_area.squeeze()
        ##

        bbmatrixyourright = bbmatrixyourright.flatten()
        bbmatrixyourright = list(bbmatrixyourright)
        # boxes_tensor = torch.tensor(bbmatrixyourright)
        # Expected target boxes to be a tensor of shape [N, 4], got torch.Size([1, 1, 4]).   BOXES!!!!
        tensor_boxes = tv_tensors.BoundingBoxes(bbmatrixyourright, format="XYXY", canvas_size=[720, 1280])
        tensor_boxes = tensor_boxes.squeeze()
        ##

        iscrowd = torch.zeros((1,), dtype=torch.int64)
        tensor_iscrowd = iscrowd
        ##

        target =  {'boxes': tensor_boxes,  # Tensor der Form [N, 4] # Nr. 3
            'labels': tensor_labels,  # Tensor der Form [N]     #
            'masks': masks,  # Tensor der Form [N, H, W]
            'image_id': image_id,  # Tensor der Form [N]
            'area': tensor_area,  # Tensor der Form [N]
            'iscrowd': tensor_iscrowd}  # Tensor der Form [N]
        #Disable when cheking with clean_test.if _name_ = '_main_':
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



