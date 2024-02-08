import argparse
import numpy as np
import nibabel as nib    
from glob import glob
import os

import torch
import torchcomplex.nn.functional as cF
import pandas as pd
import random
import h5py


__author__ = "Hadya Yassin"
__maintainer__ = "Hadya Yassin"
__email__ = "hadya.yassin@hpi.de"
__status__ = "Production"
__copyright__ = "Copyright (c) 2023 Yassin, Hadya Sa'ad Abdalwadoud"
__license__ = "MIT"

""" 

Main script to preprocess the OASIS, ADNI and Ukbiobank images and synthseg segmentation masks to be used as an input to the Segmentation Repo: 

1-Conform to 1mm voxel size in coronal slice direction with 256^3 or more
2-Remap the Synthseg masks to the specified region/s
3-Additional feature is the estimation of class weights (commented out)
4-Create h5 type files for the stacked MidSl Images, Masks, ClassWeights
4-Save each MidSl image/mask/weightclass in a folder in nifti form if nifti=True

"""

def _write_h5(data, label, f):   #  class_weights, weights,
    with h5py.File(f['data'], "w") as data_handle:
        data_handle.create_dataset("data", data=data)
    with h5py.File(f['label'], "w") as label_handle:
        label_handle.create_dataset("label", data=label)
    # with h5py.File(f['class_weights'], "w") as class_weights_handle:
    #     class_weights_handle.create_dataset("class_weights", data=class_weights)


def estimate_weights_mfb(labels):
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(33)
    for i, label in enumerate(unique):
        class_weights += (median_freq // counts[i]) * np.array(labels == label)
        weights[int(label)] = median_freq // counts[i]

    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    # return class_weights, weights
    return class_weights



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
        

    def interpWithTorchComplex(self, data, size, mode="sinc"):
        data = torch.from_numpy(data.copy()).unsqueeze(0).unsqueeze(0)
        if mode == "sinc":
                data = cF.interpolate(data, size=size, mode=mode)
        else:
                data = cF.interpolate(data+1j, size=size, mode=mode).real
        return data.numpy().squeeze()

    
def _load_nib(pth, pth2):
    try:
        img = nib.load(pth)
        affine = img.affine
        img = np.squeeze(img.get_fdata(), axis=None)

        mask = nib.load(pth2)
        affine2 = mask.affine
        mask = np.squeeze(mask.get_fdata(), axis=None)

    except Exception:
        s = np.zeros((2, 2))
        img, affine, mask, affine2 = s, s, s, s 
        pass
    
    
    return img, affine, mask, affine2



def remap(labels, remap_config, re_m):
    """
    Function to remap the label values into the desired range of algorithm
    """
    if remap_config == 'FS':
        label_list = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50,
                      51, 52, 53, 54, 58, 60]           # 24 is present in new version of synthseg label and it corresponds to csf (cranial) region

    elif remap_config == 'Neo':
        labels[(labels >= 100) & (labels % 2 == 0)] = 210
        labels[(labels >= 100) & (labels % 2 == 1)] = 211
        label_list = [45, 211, 52, 50, 41, 39, 60, 37, 58, 56, 4, 11, 35, 48, 32, 46, 30, 62, 44, 210, 51, 49, 40, 38,
                      59, 36, 57, 55, 47, 31, 23, 61]
    else:
        raise ValueError("Invalid argument value for remap config, only valid options are FS and Neo")

    new_labels = np.zeros_like(labels)

    k = 0

    for i, label in enumerate(label_list):
        
        if re_m == "VV" and (label == 4 or label == 43): ### ventricles only not > or label == 7 or label == 8 or label == 41 or label == 42 or label == 46 or label == 47 or label == 16 
            label_present = np.zeros_like(labels)
            label_present[labels == label] = 1
            new_labels = new_labels + (k + 1) * label_present
            k+=1

        elif re_m == "HV" and (label == 17 or label == 53): ### Hippocampus only not > or label == 7 or label == 8 or label == 41 or label == 42 or label == 46 or label == 47 or label == 16 
            label_present = np.zeros_like(labels)
            label_present[labels == label] = 1
            new_labels = new_labels + (k + 1) * label_present
            k+=1

        elif re_m == "CV" and (label == 3 or label == 42): ### cerebral cortex only not > or label == 7 or label == 8 or label == 41 or label == 42 or label == 46 or label == 47 or label == 16 
            label_present = np.zeros_like(labels)
            label_present[labels == label] = 1
            new_labels = new_labels + (k + 1) * label_present
            k+=1

        elif re_m == "BV" and (label != 16): ### everything but brain stem label = 16, without csf label 24, which is not available in the old synthseg labels
            label_present = np.zeros_like(labels)
            label_present[labels == label] = 1
            new_labels = new_labels + (k + 1) * label_present
            k+=1
        
        elif re_m == "BV_stem": ### everything but csf label 24, which is not available in the old synthseg labels
            label_present = np.zeros_like(labels)
            label_present[labels == label] = 1
            new_labels = new_labels + (k + 1) * label_present
            k+=1               

        elif re_m == "IV":                  #####everything including brainstem and csf from new synthseg version labels
            label_present = np.zeros_like(labels)
            label_present[labels == label] = 1
            new_labels = new_labels + (k + 1) * label_present
            k+=1

    ##merging the regions to one class to represent BV: Brain Volume, IV:Intracranial Volume only available in the new synthseg folder
    if "BV" in re_m or "IV" in re_m:
        new_labels[new_labels>0] = 1


    return new_labels

def fetch_files(Dataset, mask_name, re_m): 
    files = []
    #chose only 3Tesla MagStrength from Oasis3:
    if Dataset == "Oasis3": 
        CSV = '/dhc/groups/fglippert/Oasis3/mr_sessions_3T_HY.csv'
        df = pd.read_csv(CSV)
        df =  df[df.Scanner == "3.0T"] ##Value is str
        # df.column_name == whole string from the cell
        # now, all the rows with the column: Scanner and Value other than: "3.0T" will be deleted
        df.to_csv(CSV, index=False)
        Exfiles = (df.MR_ID.values.tolist())
        for Exfile in Exfiles:
            inters= glob(f"/dhc/groups/fglippert/{Dataset}/mri/{Exfile}/*/NIFTI/synthseg/{mask_name}")
            for inter in inters:
                files.append(inter)

    #chose only 3Tesla MagStrength from ADNI:
    elif Dataset == "adni_t1_mprage":
        CSV = '/dhc/groups/fglippert/adni_t1_mprage/df_mprage_3T_HY.csv'
        df = pd.read_csv(CSV)
        df =  df[df.MAGSTRENGTH == 3.0] ##Value is int
        df.to_csv(CSV, index=False)
        Exfiles = (df.path.values.tolist())
        for Exfile in Exfiles:
            files.append(Exfile.replace("/T1_unbiased_brain.nii.gz", f"/synthseg/{mask_name}"))

    #Uk biobank only contains 3Tesla MagStrength:
    elif Dataset == "ukbiobank":
        V = "_7_3_2" if re_m == "IV" else ""        #check if it is the new(_7_3_2) or older folder from synthseg masks, 
        files = glob(f"/dhc/projects/{Dataset}/*/*/*/*/*/*/*/synthseg{V}/{mask_name}")  

    return files

def sub_id(Dataset, words):
    ##Get Sub ID from path
    if  Dataset == "Oasis3":
        Sub_ID = f"{words[5]}_{words[6]}" 
    elif  Dataset == "adni_t1_mprage": 
        Sub_ID = f"{words[4]}_{words[5]}_{words[6]}_{words[7]}"  
    elif  Dataset == "ukbiobank":
        Sub_ID = f"{words[8]}" 
    return Sub_ID


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset', '-Data', required=True, help='Dataset, valid values are ukbiobank or adni_t1_mprage or Oasis3')
    parser.add_argument('--region_mask', '-re_m', required=True, help='mask region, valid values are VV for Ventricle volujme or BV for Brain Volume or IV for Intracranial Volume or HV for Hippocampus Volume or CV for Cerebral Cortex Volume')
    parser.add_argument('--SliceType', '-SlTyp', required=False, help='slice type, valid values is midSl for middle slices ')
    parser.add_argument('--destination_path', '-destin_path', required=False, help='destination_path of the prprocessing output folder should be specified or it will be the default path of Datasets/{Dataset}/{target_orient}/{Typ}')
    parser.add_argument('--mri_convert', type=bool, required=False, help='mri_convert, valid values are empty string "" for False or any string for True as a default to preprocess MRI images (1- reorient to Coronal, 2-Rotate, 3-resize to 256^3)')
    parser.add_argument('--target_orient', '-orient', required=False, help='target orientation default is Coronal, valid options are Axial and Sagittal')
    parser.add_argument('--stack', type=bool, required=False, help='stack, valid values are empty string "" for False or any string for True as a default to stack the images and masks into two seperate h5 files')
    parser.add_argument('--nifti', '-nii', type=bool, required=False, help='nifti, valid values are empty string "" for False as default or any string for True, it would save single slices whether from images or masks in nii form')
    parser.add_argument('--return_weights', '-r_Ws', type=bool, required=False, help='return_weights, valid values are empty string "" for False as default or any string for True, it would calculate the class weights from the mask but it is memory heavy')
    parser.add_argument('--trial', type=bool, required=False, help='trial, valid values are empty string "" for False as default or any string for True, It will run the script for 20 samples for test purposes')
    
    args = parser.parse_args()

    ############################### Specify Parameters #################################### 
    Dataset = args.Dataset          #"ukbiobank" "adni_t1_mprage" or "Oasis3"
    re_m = args.region_mask         #"VV" Ventricle volujme or "BV" Brain Volume or "IV" Intracranial Volume or "HV" Hippocampus Volume or "CV" Cerebral Cortex Volume
    Typ = args.SliceType            #midSl: middle slices 
    print(f"\n This is a Preprocessing script for mri volumes of the nifti format, the current specified slice type set for training is {Typ} and Dataset {Dataset}, brain region of interest to be remapped in masks is: {re_m}")
    mask_name = "T1toMNInonlin_synthseg.nii.gz" if Dataset == "adni_t1_mprage" or Dataset == "Oasis3" else "T1_brain_to_MNI_synthseg.nii.gz" 
    image_name = "T1toMNInonlin.nii.gz" if Dataset == "adni_t1_mprage" or Dataset == "Oasis3" else "T1_brain_to_MNI.nii.gz"     
    base_name = mask_name.replace('.nii.gz', '')
    mri_convert = args.mri_convert if args.mri_convert is not None else True                                 # Conform to 1mm voxel size in coronal slice direction with 256^2 or more (style gan output: 256x256 Coronal MidSl)
    target_orient = "Coronal"
    destin_path = args.destination_path if args.destination_path else f'Datasets/{Dataset}/{target_orient}/{Typ}'
    
    missing_pth = 0
    missing_pths = []

    files = fetch_files(Dataset, mask_name, re_m)

    trial = args.trial if args.trial is not None else False                                       # for testing purposes
    sample = 20                                         # random trial sample
    amount = "_All" if not trial else f"_{sample}Trial-sample"
    fno = 0                                             # files counter                                   
    m = 0                                               # images, masks, affines counter
    nifti = args.nifti if args.nifti is not None else False
    stack = args.stack if args.stack is not None else True       
    remap_config = "FS"
    return_weights = args.return_weights if args.return_weights is not None else False

    Sub_ID = []
    n = 0                                               #folder name counter
    IDs = {}                                            #folder content Dict
    ms = []
    ws = []
    Xs = []

    print(f"Dataset: {Dataset}, target orient: {target_orient}")
    print(f"Slice type: {Typ}, region are: {re_m}")

    ###################################################################################### 

    #Generate 1340 half of the size of oasis random numbers between 0 and len(files)
    randomlist = random.sample(range(0, len(files)), sample)    
                  
    for i in range(len(files) if Typ == "midSl" and not trial else len(randomlist)):              
           
        fno += 1    
        print(f"fno:{fno}")
        
        ## get image file from corresponding mask path
        file = files[i] if Typ == "midSl" else files[randomlist[i]]
        file1 = file.replace(f'/synthseg/{mask_name}', f'/{image_name}')        ####image file
        words = (file1.replace("/", " ")).split()   #get name of folders of the image path        

        Sub_ID = sub_id(Dataset, words)

        img, affine, mask, affine2 = _load_nib(file1, file)

        ###if any loaded path doesn't exsist, happend when loading 3T adni from csv
        if img.shape == affine.shape == mask.shape == affine2.shape:
            missing_pth += 1
            missing_pths.append(file1)

        else:
            if mri_convert:

                p = preprocessing()
                
                if target_orient == "Coronal":    #coronal direction
                    img = p.reorienting(img, axes=(2, 0, 1))   
                    img = p.rotating(img, k=1, axes=(1, 0))      # rotate img 90 to left to get std 

                    mask = p.reorienting(mask, axes=(2, 0, 1)) 
                    mask = p.rotating(mask, k=1, axes=(1, 0))    # rotate mask 90 to left to get std 


                if img.shape[0] != 256 or img.shape[1] != 256 or img.shape[2] != 256:
                    img = p.interpWithTorchComplex(img, size=(256,256,256), mode="sinc")
                    mask = p.interpWithTorchComplex(mask, size=(256,256,256), mode="nearest")

            if nifti: #saves the preprossed volume per subject 
                destin_path3 = file1.replace(f'/dhc/groups/fglippert/{Dataset}/{words[4]}/{words[5]}/{words[6]}/{words[7]}/', (destin_path + f'/{Sub_ID}/'))
                destin_path4 = destin_path3.replace(f'{image_name}', f'remap_{mask_name}')
                os.makedirs(destin_path3.replace(os.path.basename(destin_path3), '')[:-1],exist_ok=True)

                nib.save(nib.Nifti1Image(mid_img, affine), destin_path3) 
                nib.save(nib.Nifti1Image(mid_mask, affine2), destin_path4) 
                

            mid = img.shape[1] // 2
            if Typ == "midSl":   #mid slice only
                indices = [mid]

            #Select mid slices from each image and and its corrresponding mask from volumes
            mid_img = np.take(
                img,
                indices,
                2                   ##0 is sagittal slice, 1 is axial slice, 2 is coronal slice 
            )
            mid_mask = np.take(
                mask,
                indices,
                2
            )

            if remap_config == "FS":
                mid_mask = remap(mid_mask, remap_config, re_m)
                
            if return_weights:
                class_weights = estimate_weights_mfb(mid_mask)


            if "midSl" in Typ:
                Xs.append(mid_img)
                ms.append(mid_mask)
                if return_weights:
                    ws.append(class_weights)
                IDs.update({f'folder{n}' : Sub_ID})         #update the content of the current folder
                Sub_ID = []                                 #reset sub ID list for new volume
                n += 1    


    if stack: ##stack images/masks/weightclasees in seperate h5 files
        H, W, b = Xs[0].shape        ###X, m, w all have same shape

        Xs=np.concatenate(Xs, axis=2)#.reshape((-1, H, W))
        ms=np.concatenate(ms, axis=2)#.reshape((-1, H, W))
        

        destin_folder = destin_path + f"_remap{amount}_{re_m}_h5"
        print(f"h5 of Typ:{Typ} will be saved here: {destin_folder}")
        os.makedirs(destin_folder,exist_ok=True)

        f = {
            "data": os.path.join(destin_folder, f"Data_train.h5"),
            "label": os.path.join(destin_folder, f"Label_train.h5"),
            "class_weights": os.path.join(destin_folder, f"Class_Weight_train.h5"),
            }
    
        _write_h5(Xs, ms, f)

        if return_weights:
            ws=np.concatenate(ws, axis=2)#.reshape((-1, H, W))

            with h5py.File(f['class_weights'], "w") as class_weights_handle:
                class_weights_handle.create_dataset("class_weights", data=ws)
        
        
    os.makedirs(f'Datasets/{Dataset}/StackFoldersContent/',exist_ok=True)

    ##write to an output file the Subject ID content of each stack
    if stack:        
        with open(f'Datasets/{Dataset}/StackFoldersContent/{Dataset}_StackContent_{Typ}{amount}.txt', 'w') as f:
            for h in range(len(IDs)):
                print(f"folde{h}: {IDs[f'folder{h}']}", file=f)
    

    #### if there is any missing paths from directory available in csv
    if missing_pth != 0: 
        with open(f'Datasets/{Dataset}/StackFoldersContent/{Dataset}_missing_paths_{Typ}{amount}.txt', 'w') as f:
            print(f"missing_paths_counter: {missing_pth}", file=f)
            for s in range(len(missing_pths)):
                print(f"Path: {missing_pths[s]}", file=f)
    ###############
 
    print("Done")



if __name__ == "__main__":
    main()