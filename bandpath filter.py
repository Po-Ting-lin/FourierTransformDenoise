import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import skimage as sk

class bandpath():
    def __init__(self,path):
        self.path = path
        self.img = None
        self.mask = None
        self.row = 0
        self.col = 0

    def openimg(self):
        img = cv2.imread(self.path,0)
        if img is None:
            raise FileNotFoundError("Cannot open this image!")
        else :
            self.img = img

    def masks(self):
        self.row = int(self.img.shape[0]/2)
        self.col = int(self.img.shape[1]/2)
        mask1 = np.ones(self.img.shape,np.uint8)
        mask1=mask1*255
        for r in range(1,51):   #range = max pixel
            magni1 = (0+(r/50))*255
            cv2.circle(mask1,(self.row,self.col),r,(magni1,magni1,magni1),r-1) # central remain one
        mask2=np.zeros(self.img.shape,np.uint8)
        for r in range(500,0,-1):   #range = min pixel
            magni2 = (2/(1+math.exp(r/100)))*255   #sigmoid decay
            cv2.circle(mask2,(self.row,self.col),r,(magni2,magni2,magni2),r)
        self.mask = (mask2/255)*(mask1/255)

    #
    def fft(self):
        self.openimg()
        self.masks()
        f=np.fft.fft2(self.img)            #2d fourier transform
        fshift=np.fft.fftshift(f)             # move dc component transformed by fft to central
        p = 1
        n = 1                   # the range to suppress all except for the DC component
        fshift = fshift*self.mask                         #combine with first filter
        fshift[self.row-p:self.row+p,0:self.col-n ] = 0.001          # suppress upper part
        fshift[self.row-p:self.row+p,self.col+n:self.col+self.col] = 0.001  # suppress lower part
        # calculate new amplitude spectrum
        mag_spec2 = 20*np.log(np.abs(fshift))          # owing to fft result is way bigger than moniter , using log to make detail easier to be observed
        inv_fshift = np.fft.ifftshift(fshift)          #reverse fourier
        img_recon = np.real(np.fft.ifft2(inv_fshift))  # reconstruct image
        plt.figure()
        plt.imshow(img_recon)
        plt.show()

