import cv2
import numpy as np

# TODO match the color in color_default.py
BG_COLOR = (255, 255, 255, 255)
START_COLOR = (100, 149, 237, 255)
DEST_COLOR = (69, 139, 0, 255)
OBSTACLE_COLOR = (150, 150, 150, 255)
TRAJ_COLOR = (10, 10, 150, 255)

class Obs_Processor():
    def __init__(self) -> None:
        self.downsample_rate = 4
        self.morph_kernel = np.array([
            [0,0,0,1,0,0,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [1,1,1,1,1,1,1],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,0,0,1,0,0,0]],dtype=np.uint8)
        self.n_channels = 2
        pass

    def process_img(self, img):
        
        obstacle_ege = self.get_obstacle_edge(img)
        img_traj = self.rgb2binary(img, TRAJ_COLOR[:3])

        return np.array([obstacle_ege, img_traj]).transpose(-1,0,1)

    def rgb2binary(self, img, color):
        img_b = (img==color).astype(np.uint8)
        img_b = (np.sum(img_b,axis=-1) == 3).astype(np.uint8)
        img_b = self.max_pooling2d(img_b)
        return img_b
    
    def max_pooling2d(self, img:np.ndarray):
        # do not find available function to do maxPooling from cv2,
        # so here implement a simple version.
        w,h = img.shape
        k = self.downsample_rate
        if w%self.downsample_rate or h%self.downsample_rate:
            raise Warning('image shape can not be divided !')
        w_, h_ = w//k, h//k
        img = img.reshape(w_, k, h_, k).transpose(1,3,0,2).reshape(k**2, w_, h_)
        img = np.max(img, axis=0)
        return img


    def get_obstacle_edge(self, img):
        img_obstacle = self.rgb2binary(img, OBSTACLE_COLOR[:3])*255
        img_obstacle = cv2.morphologyEx(img_obstacle, cv2.MORPH_CLOSE, self.morph_kernel)
        edge_obstacle = cv2.Canny(img_obstacle, 100, 200)
        return edge_obstacle
    