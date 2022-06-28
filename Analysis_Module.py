
from multiprocessing import Process
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
sns.set_style("white")



class IndexTracker(object):
    def __init__(self,ax, X):
        self.points = {}
        self.coords = []
        self.ax = ax
        plt.subplots_adjust(left = 0,bottom = 0.15,right = 0.845,top = 0.9)
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
        else:
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
        for c in (range(d-500,d+500)):
            pair_x = []
            for xi in range(x-50,x+50):
                if xi>=1200 or xi<0:
                    pass
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
    cut_points = positions['cut']
    x_coords,y_coords = unpack_pos(axes_points)
    slope  = find_slope(x_coords,y_coords)
    theta_deg = math.degrees(math.atan(slope))
    intercept = cut_points[0][1] - int(slope*cut_points[0][0])
    return theta_deg,intercept

def find_slope(x_coords,y_coords):
    if x_coords[0]>x_coords[1]:
        slope = (y_coords[0] - y_coords[1])/(x_coords[0] - x_coords[1]) 
    else:
        slope =  (y_coords[1] - y_coords[0])/(x_coords[1] - x_coords[0]) 
    return slope
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

def Processing(img,positions):
    m,c = slope(positions)
    img_cut = find_xy_pair(img,m,int(m*int(positions['cut'][0][0])),c)

    kinograph = np.array([abs(np.min_scalar_type(i,axis = 0)) for i in img_cut])
    kinograph = clean_up_kinograph(kinograph)
    plt.imshow(kinograph,cmap = 'gray',filternorm= True)
    return 0


def get_desired_cut_positions(file,path):
    print(file)
    img = tff.imread(path + file)
    img = img.T
    #SLICES VIEWER
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    fig.canvas.set_window_title(file)

    klicker = clicker(ax = ax,classes=["cut","axis"],markers = ["x","."],colors = ["red",'yellow'],X = img)
    
    Processing(img,klicker.get_positions())



def input_taker():
    root = tk.Tk()
    root.withdraw()
    user_input = messagebox.askyesnocancel("Continue Analysis","Do you want to move onto next")
    if user_input == None:
            if messagebox.askyesno("Stop Analysis","Exit?"):
                exit()
    return not user_input

path = "/Users/aman/Desktop/pasteur/data/"
def start_processing():
    # path = "/Users/akukde/Desktop/pasteur/edge_data/"
    path = "/Users/akukde/Desktop/pasteur/data/"
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