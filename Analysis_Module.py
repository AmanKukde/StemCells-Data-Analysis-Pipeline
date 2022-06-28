
from multiprocessing import Process
from sympy import interpolate
import tifffile as tff
import cv2
from matplotlib import pyplot as plt
import os
import seaborn as sns
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from mpl_point_clicker import clicker
import pdb
import tkinter as tk
from tkinter import messagebox
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from tqdm import tqdm 
import math
from scipy import ndimage
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
import gc
sns.set_style("white")



class IndexTracker(object):
    def __init__(self,ax, X):
        self.points = {}
        self.coords = []
        self.ax = ax
        plt.subplots_adjust(left = 0.15,bottom = 0.15,right = 0.95,top = 0.9)
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        # self.ind = self.slices//2
        self.ind = 0
        self.im = ax.imshow(self.X[:, :, self.ind],cmap = 'gray')
        # self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        # self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.update()
        

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        elif event.button == 'down':
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        
   
 
def find_xy_pair(img,m,x,c):
    d = math.floor(c)
    pair_i= []
    for i in tqdm(range(img.shape[-1])):
        im = img[:,:,i]
        pair_c =[]
        for c in (range(d-50,d+50)):
            pair_x = []
            for xi in range(x-50,x+50):
                if xi>=1200 or xi<0:
                    pair_x.append(0)
                else:
                    y = math.floor(m*xi+c)
                    if y>=1200 or y<0:
                        pair_x.append(0)
                    else:
                        pair_x.append(im[xi,y])
            pair_c.append(pair_x)
        pair_i.append(np.array(pair_c))
    return np.array(pair_i)

def unpack_pos(axes_points):
    
    x_coords = [math.floor(x) for x in [axes_points[i][0] for i in range(axes_points.shape[0])]]
    y_coords = [math.floor(y) for y in [axes_points[i][1] for i in range(axes_points.shape[0])]]
    return x_coords,y_coords

def slope(positions):
    axes_points = positions['axis']
    x_coords,y_coords = unpack_pos(axes_points)
    theta = (y_coords[1] - y_coords[0])/(x_coords[1] - x_coords[0])
    theta = math.degrees(math.atan(theta))
   
    return theta


def scroll_through(img):
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    tracker = IndexTracker(ax, img)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show(block = True)

def clean_up_kinograph(kinograph):
    new = np.array([])
    for i in range(kinograph.shape[0]-1):
        if kinograph[i].all()!=kinograph[i+1].all():
            new.append(kinograph[i])
    return new

def Processing(img,title,positions):
    m = slope(positions)
    ro = ndimage.rotate(img,m,mode= "nearest")
    del img
    gc.collect()
    klicker_2 = initialise_clicker(ro,title = f"Rotated {title}",classes = ["cut point"])
    points = klicker_2.get_positions()['cut point'][0]
    cut_x,cut_y = list(map(math.floor,points))
    img_cut = ro[cut_y-5:cut_y+5,cut_x - 150:cut_x +150,:]
    r_i = np.array([cv2.resize(img_cut[:,:,i],(1,300), interpolation= cv2.INTER_LANCZOS4) for i in range(img_cut.shape[-1])])
    ri = np.hstack(r_i)
    print(ri.shape)
    plt.figure();
    plt.imshow(ri.T,cmap = 'gray');plt.plot()
    # kinograph = clean_up_kinograph(kinograph)
    # plt.imshow(kinograph,cmap = 'gray',filternorm= True)
    return 0


def new_point_after_rot(m,positions):
    cut_points = positions['cut'][0] 
    x,y = cut_points
    x1 = (x - y*m)*(math.cos(m))
    y1 = (y + x*m)*(math.cos(m))
    return np.array([x1,y1])


def get_desired_cut_positions(file,path):
    print(file)
    img = tff.imread(path + file)
    img = img.T
    #SLICES VIEWER
    klicker  = initialise_clicker(img,title = str(file),classes=["axis"],markers = ["x"], colors = ['r'])
    Processing(img,str(file),klicker.get_positions())

def initialise_clicker(img,title,classes = ["Dummy"],markers = ["."],colors = ['Yellow']):
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    fig.canvas.set_window_title(title)
    klicker = clicker(ax = ax,classes=classes,markers = markers ,colors = colors,X = img)
    return klicker

def input_taker():
    root = tk.Tk()
    root.withdraw()
    user_input = messagebox.askyesnocancel("Continue Analysis","Do you want to move onto next")
    if user_input == None:
            if messagebox.askyesno("Stop Analysis","Exit?"):
                exit()
    return not user_input

path = "/Users/aman/Desktop/pasteur/data/"
# path = "/Users/aman/Desktop/pasteur/test/"
def start_processing():
    files = os.listdir(path)
    files.sort()
    # for file in files:
    for file in files[4:]:
        user_input = True
        while user_input:
            get_desired_cut_positions(file,path)
            user_input = input_taker()
            print(user_input)

start_processing()