#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import filedialog
import numpy as np
from imageio import imread
from skimage.filters import difference_of_gaussians
from tkinter import *

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
       
        
         # Load an image using OpenCV
        self.image_path = filedialog.askopenfilename()
        self.cv_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.height, self.width, no_channels = self.cv_img.shape
       
 
         # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        
       # Create Filter label
        label1 = tkinter.Label(window,text="Filters")
        label1.grid(row=0,column=13, columnspan=5)
 
         # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(window, width = self.width, height = self.height)


        self.canvas.grid(row=0, column=1, rowspan=15, columnspan=5)

       # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))

         # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        
        #Linear type
        label2 = tkinter.Label(window,text="Linear type")
        label2.grid(row=1,column=13, columnspan=1)
        
        # Color Image Button
        self.btn_load=tkinter.Button(window, text="Color", width=10, command=self.color_image)
        self.btn_load.grid(row=2, column=13)
       
        # Gray Scale Image Button
        self.btn_gray=tkinter.Button(window, text="Gray", width=10, command=self.gray_image)
        self.btn_gray.grid(row=3, column=13)
        
        # Lowpass button
        self.past = 1 
        def fix_lp(n):
            n = int(n)
            if not n % 2:
                self.kernel_lp.set(n+1 if n > self.past else n-1)
                self.past = self.kernel_lp.get()
               
        self.kernel_lp = Scale(from_=1, to_=21, command=fix_lp, orient=HORIZONTAL, label='Kernel Size')
        self.kernel_lp.grid(row=4, column=14)
        
        self.btn_lpf=tkinter.Button(window, text="LPF", width=10, command=self.LPF)
        self.btn_lpf.grid(row=4, column=13)
        
        
        # High pass button
        def fix_hp(n):
            n = int(n)
            if not n % 2:
                self.kernel_hp.set(n+1 if n > self.past else n-1)
                self.past = self.kernel_hp.get()
               
        self.kernel_hp = Scale(from_=1, to_=21, command=fix_hp, orient=HORIZONTAL, label='Kernel Size')
        self.kernel_hp.grid(row=5, column=14)
        
        self.sigma_hp=Scale(from_=1, to_=21, orient=HORIZONTAL, label='Sigma')
        self.sigma_hp.grid(row=5, column=15)
        
        self.btn_hp=tkinter.Button(window, text="HP", width=10, command=self.high_pass)
        self.btn_hp.grid(row=5, column=13)
       
        # Band pass button
        self.kernel_bp1=Scale(from_=1, to_=8, orient=HORIZONTAL, label='higher_cut_off_kernel')
        self.kernel_bp1.grid(row=6, column=14)
        self.kernel_bp2=Scale(from_=9, to_=20, orient=HORIZONTAL, label='lower_cut_off_kernel')
        self.kernel_bp2.grid(row=6, column=15)
               
        self.btn_bp=tkinter.Button(window, text="BandPass", width=10, command=self.bandpass)
        self.btn_bp.grid(row=6, column=13)
        
        # band Reject button
        
        self.kernel_br1=Scale(from_=1, to_=8, orient=HORIZONTAL, label='higher_cut_off_kernel')
        self.kernel_br1.grid(row=7, column=14)
        self.kernel_br2=Scale(from_=9, to_=20, orient=HORIZONTAL, label='lower_cut_off_kernel')
        self.kernel_br2.grid(row=7, column=15)
        
        self.btn_br=tkinter.Button(window, text="BandReject", width=10, command=self.bandreject)
        self.btn_br.grid(row=7, column=13)
        
        #Statistical type
        label3 = tkinter.Label(window,text="Statistical type")
        label3.grid(row=8,column=13, columnspan=1)
        
        
        # Median filter button with kernal scaling
 
        def fix(n):
            n = int(n)
            if not n % 2:
                self.kernel.set(n+1 if n > self.past else n-1)
                self.past = self.kernel.get()
               
        self.kernel = Scale(from_=1, to_=15, command=fix, orient=HORIZONTAL, label='Kernel Size')
        self.kernel.grid(row=9, column=14)
        
        self.btn_median=tkinter.Button(window, text="Median", width=10, command=self.median)
        self.btn_median.grid(row=9, column=13)
        
        #Derivative type
        label4 = tkinter.Label(window,text="Derivative type")
        label4.grid(row=10,column=13, columnspan=1)
        
        # Laplacian filter button with kernal scaling
        def fix_l(n):
            n = int(n)
            if not n % 2:
                self.kernel_lap.set(n+1 if n > self.past else n-1)
                self.past = self.kernel_lap.get()
               
        self.kernel_lap = Scale(from_=1, to_=15, command=fix_l, orient=HORIZONTAL, label='Kernel Size')
        self.kernel_lap.grid(row=11, column=14)
        
        self.btn_laplacian =tkinter.Button(window, text="Laplacian", width=10, command=self.laplacian)
        self.btn_laplacian.grid(row=11, column=13)
                            
        
        # Laplacian of Gaussian filter button with kernal scaling
        def fix_lg(n):
            n = int(n)
            if not n % 2:
                self.kernel_lg.set(n+1 if n > self.past else n-1)
                self.past = self.kernel_lg.get()                   
         
        
        self.kernel_lg = Scale(from_=1, to_=23, command=fix_lg, orient=HORIZONTAL, label='Kernel Size')
        self.kernel_lg.grid(row=12, column=14)  
        
        self.kernel_X = Scale(from_=0, to_=1, orient=HORIZONTAL, label='SigmaX')
        self.kernel_X.grid(row=12, column=15)
       
        self.kernel_Y = Scale(from_=0, to_=1, orient=HORIZONTAL, label='SigmaY')
        self.kernel_Y.grid(row=12, column=16)
                             
        self.btn_lap_gaauss=tkinter.Button(window, text="Laplace_gauss", width=10, command=self.lap_gauss)
        self.btn_lap_gaauss.grid(row=12, column=13)
        
         # Prewitt filter button 
                     
        self.dir_a = Scale(from_=0, to_=1, orient=HORIZONTAL, label='dx')
        self.dir_a.grid(row=13, column=14)
       
        self.dir_b = Scale(from_=0, to_=1, orient=HORIZONTAL, label='dy')
        self.dir_b.grid(row=13, column=15)
        
        self.btn_prewitt=tkinter.Button(window, text="Prewitt", width=10, command=self.prewitt)
        self.btn_prewitt.grid(row=13, column=13) 
                             
        # Sobel filter button 
        def fix_sob(n):
            n = int(n)
            if not n % 2:
                self.kernel_sob.set(n+1 if n > self.past else n-1)
                self.past = self.kernel_sob.get()
               
        self.kernel_sob = Scale(from_=1, to_=23, command=fix_sob, orient=HORIZONTAL, label='Kernel Size')
        self.kernel_sob.grid(row=14, column=14)
       
        self.dir_X = Scale(from_=0, to_=1, orient=HORIZONTAL, label='dx')
        self.dir_X.grid(row=14, column=15)
       
        self.dir_Y = Scale(from_=0, to_=1, orient=HORIZONTAL, label='dy')
        self.dir_Y.grid(row=14, column=16)
        
        self.btn_sobel=tkinter.Button(window, text="Sobel", width=10, command=self.sobel)
        self.btn_sobel.grid(row=14, column=13)                  
                             
        # Histogram Equalization
        label5 = tkinter.Label(window,text="Histogram Equalization")
        label5.grid(row=15,column=13, columnspan=1)
        
        # Global Equalization
        self.btn_hist=tkinter.Button(window, text="Global_hist", width=10, command=self.global_hist)
        self.btn_hist.grid(row=16, column=13)                    
                             
                             
        # Adaptive Equalization
        self.btn_ad_hist=tkinter.Button(window, text="Adaptive_hist", width=10, command=self.adaptive_hist)
        self.btn_ad_hist.grid(row=17, column=13)
            

        self.window.mainloop()

                             
    def color_image(self):
        self.cv_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
       
    def gray_image(self):
       
        img_sh = imread(self.image_path)
        img = cv2.imread(self.image_path)
        if np.ndim(img_sh) == 3:
            self.cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
           
        elif np.ndim(img_sh) == 2:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)                           
                                                         
                             
    def LPF(self):
        kn = round(self.kernel_lp.get())   
        self.cv_img = cv2.blur(self.cv_img, (kn, kn))
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
       
                             
    def high_pass(self):

        sigma = round(self.sigma_hp.get())
        kh = round(self.kernel_hp.get())
        image_hp = self.cv_img - cv2.GaussianBlur(self.cv_img, (kh,kh), sigma) + 127
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image_hp))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
       
    def bandpass(self):       
        img_a = imread(self.image_path)
        img = cv2.imread(self.image_path)
        h_c=round(self.kernel_bp1.get()) 
        l_c=round(self.kernel_bp2.get())
        if np.ndim(img_a) == 2:
            filtered_image = difference_of_gaussians(self.cv_img, h_c, l_c)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(filtered_image*255))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

           
        elif np.ndim(img_a) == 3:
            self.cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            filtered_image = difference_of_gaussians(self.cv_img, h_c, l_c)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(filtered_image*255))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)          
 
        
    def bandreject(self):
        img_r = imread(self.image_path)
        img = cv2.imread(self.image_path)
        h_ct=round(self.kernel_br1.get())
        l_ct=round(self.kernel_br2.get())   
        if np.ndim(img_r) == 2:
            image_br= self.cv_img-255*difference_of_gaussians(self.cv_img, h_ct, l_ct)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image_br))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW) 

           
        elif np.ndim(img_r) == 3:
            self.cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_br= self.cv_img-255*difference_of_gaussians(self.cv_img, h_ct, l_ct)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image_br))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW) 
        
                                 
      
    def median(self):
        p = round(self.kernel.get())
        self.cv_img = cv2.medianBlur(self.cv_img,p)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
       
    def laplacian(self):
        p = round(self.kernel_lap.get())
        img_gray = cv2.imread(self.image_path,0)
        laplacian = cv2.Laplacian(img_gray,cv2.CV_64F, ksize = p)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(laplacian))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                             
                             
    def lap_gauss(self):
        img_lg = imread(self.image_path)
        img = cv2.imread(self.image_path)
        p = round(self.kernel_lg.get())
        q = round(self.kernel_X.get())
        r = round(self.kernel_Y.get())  

        
        if np.ndim(img_lg) == 2:
            image2= cv2.GaussianBlur(self.cv_img,(p,p), sigmaX = q, sigmaY = r, borderType = cv2.BORDER_DEFAULT)
            image1=cv2.Laplacian(image2,cv2.CV_64F,ksize=p)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image1))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        
                   
        elif np.ndim(img_lg) == 3:
            src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image2= cv2.GaussianBlur(src_gray,(p,p), sigmaX = q, sigmaY = r, borderType = cv2.BORDER_DEFAULT)                 
            image1=cv2.Laplacian(image2,cv2.CV_64F,ksize=p)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image1))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                             
                             
    def prewitt(self):
        a = round(self.dir_a.get())
        b = round(self.dir_b.get())
        
        kernelx = a*np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = b*np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        gray = cv2.imread(self.image_path,0)
        img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
        img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
        img_prewit=img_prewittx+img_prewitty
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img_prewit))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
                             
       
    def sobel(self):
        p = round(self.kernel_sob.get())
        q = round(self.dir_X.get())
        r = round(self.dir_Y.get())
        img_gray = cv2.imread(self.image_path,0)
        sobel = cv2.Sobel(img_gray,-1,  dx=q, dy=r, ksize=p, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(sobel))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
            

   
    def global_hist(self):
        
        img_glo = imread(self.image_path)
        img = cv2.imread(self.image_path)
        if np.ndim(img_glo) == 2:
            equ = cv2.equalizeHist(self.cv_img)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(equ))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        
                   
        elif np.ndim(img_glo) == 3:
            src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(src_gray)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(equ))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

       

    def adaptive_hist(self):
        img_raw = imread(self.image_path)                     
        img = cv2.imread(self.image_path)
       
        if np.ndim(img_raw) == 3:
            blue, green, red = cv2.split(img)
            clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize=(8,8))
            clahe_img_b =clahe.apply(blue)
            clahe_img_g =clahe.apply(green)
            clahe_img_r =clahe.apply(red)
            rgb_clahe = cv2.merge([clahe_img_r,clahe_img_g,clahe_img_b])
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(rgb_clahe))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
           
        elif np.ndim(img_sh) == 2:
            img_g = cv2.imread(self.image_path,0)
            clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(img_g)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(clahe_img))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        
       
       
       
 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Image Filter")


# In[ ]:





# In[ ]:




