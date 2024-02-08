""" 

Main script to calculate the areas of ineterest for different regions of the brain in segmented Real vs fake 2D image

Created by Hadya Yassin
Date: 22.Aug.2022
Time: 14:00:00

"""

import numpy as np
from PIL import Image
import nibabel as nib    
from glob import glob
import csv
import os

import torch
import torchcomplex.nn.functional as cF

class preprocessing:

    def cropping(self, data_squeeze, IMAGE_HEIGHT, IMAGE_WIDTH):
        """ Crops the image to the desired size """
        # return tf.image.resize_with_crop_or_pad(data_squeeze, IMAGE_HEIGHT, IMAGE_WIDTH)  
      

    def padding(self, data_squeeze, pad_width, mode):
        """ Zero pads the image to the desired size """
        return np.pad(data_squeeze, pad_width, mode)                                                             

   
    def reorienting(self, data_squeeze, axes):
        """ Reorient image to (Sagittal, Coronal, Axial) """
        return np.transpose(data_squeeze, axes)                       


    def flipping(self, data_squeeze, axis):
        """ Flips the image  """
        return np.flip(data_squeeze, axis)                                          


    def rotating(self, data_squeeze, k, axes):
        """ Rotates the image 90 degrees to the left or right """
        return np.rot90(data_squeeze, k, axes)       
        

    def delete(self, folderPath):
        """ Deletes unwanted files in the folder """
        files = glob.glob(os.path.join(folderPath,'**/**/*.txt')) + glob.glob(os.path.join(folderPath,'**/**/reconCorrected.nii')) + glob.glob(os.path.join(folderPath,'**/**/reconCorrectedNorm.nii'))

        for file in files:
            os.remove(file)


    def interpWithTorchComplex(self, data, size, mode="sinc"):
        data = torch.from_numpy(data.copy()).unsqueeze(0).unsqueeze(0)
        if mode == "sinc":
                data = cF.interpolate(data, size=size, mode=mode)
        else:
                data = cF.interpolate(data+1j, size=size, mode=mode).real
        return data.numpy().squeeze()

    
def _load_nib(pth, pth2):
    img = nib.load(pth)
    affine1 = img.affine

    fake = nib.load(pth2)
    affine2 = img.affine
    
    data_squeeze = np.squeeze(img.get_fdata(), axis=None)
    data_squeeze2 = np.squeeze(fake.get_fdata(), axis=None)

    img = Image.fromarray(data_squeeze)
    fake = Image.fromarray(data_squeeze2)
    
    return img, fake


files = glob("/dhc/groups/fglippert/Oasis3/mri/*/*/NIFTI/synthseg/img.nii.gz")

IDs = []
BKGs = []
LVs = []
ILVs = []
V3s = []
V4s = []
TVs = []
HCs = []
TPs = []

ID2s = []
BKG2s = []
LV2s = []
ILV2s = []
V32s = []
V42s = []
TV2s = []
HC2s = []
TP2s = []


for file in files:
    file2 = file.replace('img.nii.gz', 'fake.nii.gz')
    real, fake = _load_nib(file, file2)
    real= real.convert("L")    ##grey scale
    fake = fake.convert("L")    ##grey scale

    ####Method 1
    im1 = Image.Image.getcolors(real)    #https://www.geeksforgeeks.org/python-pil-getcolors-method/
    im2 = Image.Image.getcolors(fake)    #https://www.geeksforgeeks.org/python-pil-getcolors-method/

    # TODO: Save output in csv file for each scan with scan ID and anat

    print(f"\nGet no of pixels for each label in real image {file}: \n {im1} \n") 
    print(f"\nGet no of pixels for each label in fake image {file}: \n {im2} \n") 

    Total_Pixels_Count = 0
    Total_Pixels_Count2 = 0

    for i in range(len(im1)):
        Total_Pixels_Count += im1[i][0] 
    TPs.append(Total_Pixels_Count)

    for i in range(len(im2)):
        Total_Pixels_Count2 += im2[i][0] 
    TP2s.append(Total_Pixels_Count2)


    print("Percentage of certain region pixel to whole image:")

    for i in range(len(im1)):
        if im1[i][1] == 0:
            BKG = im1[i][0]     # no of Back ground pixels * area(1mm^2) = no of pixels mm^2

        elif im1[i][1] == 4:
            LLV = im1[i][0]     # no of left Lateral ventricle pixels * area(1mm^2) = no of pixels mm^2

        elif im1[i][1] == 5:
            LILV = im1[i][0]    # no of left Inferior lateral ventricle pixels * area(1mm^2) = no of pixels mm^2

        elif im1[i][1] == 14:
            V3 = im1[i][0]      # no of 3rd ventricle pixels * area(1mm^2) = no of pixels mm^2

        elif im1[i][1] == 15:
            V4 = im1[i][0]      # No label 15 of the 4th ventricle info is visible in the mid slice, brain stem with label 16 at idx 9 shows instead at that position in the slice
        
        elif im1[i][1] == 17:
            LHC = im1[i][0]    # no of left hippocampus pixels * area(1mm^2) = no of pixels mm^2
        
        elif im1[i][1] == 43:
            RLV = im1[i][0]    # no of right Lateral ventricle pixels * area(1mm^2) = no of pixels mm^2
        
        elif im1[i][1] == 44:
            RILV = im1[i][0]   # no of right Inferior lateral ventricle pixels * area(1mm^2) = no of pixels mm^2
        
        elif im1[i][1] == 53:
            RHC = im1[i][0]    # no of right hippocampus pixels * area(1mm^2) = no of pixels mm^2

    for i in range(len(im2)):
            if im2[i][1] == 0:
                BKG2 = im2[i][0]     # no of Back ground pixels * area(1mm^2) = no of pixels mm^2

            elif im2[i][1] == 4:
                LLV2 = im2[i][0]     # no of left Lateral ventricle pixels * area(1mm^2) = no of pixels mm^2

            elif im2[i][1] == 5:
                LILV2 = im2[i][0]    # no of left Inferior lateral ventricle pixels * area(1mm^2) = no of pixels mm^2

            elif im2[i][1] == 14:
                V32 = im2[i][0]      # no of 3rd ventricle pixels * area(1mm^2) = no of pixels mm^2

            elif im2[i][1] == 15:
                V42 = im2[i][0]      # No label 15 of the 4th ventricle info is visible in the mid slice, brain stem with label 16 at idx 9 shows instead at that position in the slice
            
            elif im2[i][1] == 17:
                LHC2 = im2[i][0]    # no of left hippocampus pixels * area(1mm^2) = no of pixels mm^2
            
            elif im2[i][1] == 43:
                RLV2 = im2[i][0]    # no of right Lateral ventricle pixels * area(1mm^2) = no of pixels mm^2
            
            elif im2[i][1] == 44:
                RILV2 = im2[i][0]   # no of right Inferior lateral ventricle pixels * area(1mm^2) = no of pixels mm^2
            
            elif im2[i][1] == 53:
                RHC2 = im2[i][0]    # no of right hippocampus pixels * area(1mm^2) = no of pixels mm^2

    #Background
    BKGs.append(BKG)
    print(f"BKG: {round(BKG/ Total_Pixels_Count *100, 2)}%")

    BKG2s.append(BKG2)
    print(f"BKG2: {round(BKG2/ Total_Pixels_Count2 *100, 2)}%")


    #Ventricles
    LV = LLV + RLV
    LVs.append(LV)
    print(f"LLV: {round(LLV/ Total_Pixels_Count *100, 2)}%, RLV: {round(RLV/ Total_Pixels_Count *100, 2)}%, LV: {round(LV/ Total_Pixels_Count *100, 2)}%")

    LV2 = LLV2 + RLV2
    LV2s.append(LV2)
    print(f"LLV2: {round(LLV2/ Total_Pixels_Count2 *100, 2)}%, RLV2: {round(RLV2/ Total_Pixels_Count2 *100, 2)}%, LV2: {round(LV2/ Total_Pixels_Count2 *100, 2)}%")


    ILV = LILV + RILV
    ILVs.append(ILV)
    print(f"LILV: {round(LILV/ Total_Pixels_Count *100, 2)}%, RILV: {round(RILV/ Total_Pixels_Count *100, 2)}%, ILV: {round(ILV/ Total_Pixels_Count *100, 2)}%")
    
    ILV2 = LILV2 + RILV2
    ILV2s.append(ILV2)
    print(f"LILV2: {round(LILV2/ Total_Pixels_Count2*100, 2)}%, RILV2: {round(RILV2/ Total_Pixels_Count2 *100, 2)}%, ILV: {round(ILV2/ Total_Pixels_Count2 *100, 2)}%")
    

    V3s.append(V3)
    print(f"V3: {round(V3/ Total_Pixels_Count *100, 2)}%")       
    
    V32s.append(V32)
    print(f"V32: {round(V32/ Total_Pixels_Count2 *100, 2)}%")      


    try:
        V4s.append(V4)
        print(f"V4: {round(V4/ Total_Pixels_Count *100, 2)}%")      
    except:
        V4 = 0
        V4s.append(V4)
        print("V4 is missing")
        pass

    try:
        V42s.append(V42)
        print(f"V42: {round(V42/ Total_Pixels_Count2 *100, 2)}%")      
    except:
        V42 = 0
        V42s.append(V42)
        print("V42 is missing")
        pass


    Total_Ventricle = LV + ILV + V3 + V4 
    TVs.append(Total_Ventricle)
    print(f"Total_Ventricle: {round(Total_Ventricle/ Total_Pixels_Count *100, 2)}%")

    Total_Ventricle2 = LV2 + ILV2 + V32 + V42 
    TV2s.append(Total_Ventricle2)
    print(f"Total_Ventricle2: {round(Total_Ventricle2/ Total_Pixels_Count2 *100, 2)}%")


    # Hippocasmpus
    HC = LHC + RHC
    HCs.append(HC)
    print(f"LHC: {round(LHC/ Total_Pixels_Count *100, 2)}%, RHC: {round(RHC/ Total_Pixels_Count *100, 2)}%, HC: {round(HC/ Total_Pixels_Count *100, 2)}%")

    HC2 = LHC2 + RHC2
    HC2s.append(HC2)
    print(f"LHC2: {round(LHC2/ Total_Pixels_Count2 *100, 2)}%, RHC2: {round(RHC2/ Total_Pixels_Count2 *100, 2)}%, HC2: {round(HC2/ Total_Pixels_Count2 *100, 2)}%")


    #Areas of the regions
    print("\nAreas of certain region in mm^2:")

    print(f"areas:- BackGround: {BKG} mm^2, Lateral ventricle: {LV} mm^2, Inferior lateral ventricle: {ILV} mm^2, 3rd ventricle: {V3} mm^2, Total_Ventricle: {Total_Ventricle} mm^2, Hippocampus: {HC} mm^2")

    print(f"areas:- BackGround2: {BKG2} mm^2, Lateral ventricle2: {LV2} mm^2, Inferior lateral ventricle2: {ILV2} mm^2, 3rd ventricle2: {V32} mm^2, Total_Ventricle2: {Total_Ventricle2} mm^2, Hippocampus2: {HC2} mm^2")

    print("\n")
    

