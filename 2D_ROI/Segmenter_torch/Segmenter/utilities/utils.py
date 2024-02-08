import os
from statistics import mean, stdev
import numpy as np
import nibabel as nib
import torch


def getValStat(data: list, percentile: float=99):
    vals = []
    for datum in data:
        vals += list(datum.flatten())
    return mean(vals), stdev(vals), np.percentile(np.array(vals), percentile)


def result_analyser(mask_pred, mask_true, classes, path, image, files, dataset, runMode, test_typ, re_m, test_path):

    rotate = False
    flip = False

    if mask_pred is not None and len(mask_pred) == len(image) ==len(files):

        for i in range(len(mask_pred)):

            if runMode == 'onlyTest' and "-Wes" in test_typ:
                TestType = "External" if "External" in test_typ else "Internal"
                modelID = "baseline_model" if "Base" in  test_typ else "ms_model"

                #adjust path for coldstore directory
                PATH = files[i].replace(f"/dhc/groups/fglippert", test_path)

                if dataset == 'adni_t1_mprage':
                    #adjust real path for test folder
                    PATH= PATH.replace("/imaging/brain_mri/t1_coronal_mni_new",  f"/{TestType}/MidSl/{dataset}{re_m}_Model/t1_coronal_mni_new")
                    #adjust Synth path for test folder
                    PATH= PATH.replace(f"/imaging/brain_mri/eval_segments/{modelID}/gen_test_re256_mni", f"/{TestType}/MidSl/{dataset}{re_m}_Model/{modelID}/gen_test_re256_mni/pred_mask")


                elif dataset == 'ukbiobank':
                    #adjust real path for test folder
                    PATH= PATH.replace(f"/T1_3T_coronal_mni_nonlinear", f"/{TestType}/MidSl/{dataset}{re_m}_Model/T1_3T_coronal_mni_nonlinear")
                    #adjust Synth path for test folder
                    PATH= PATH.replace(f"/dhc/cold/groups/syreal/ADNI/eval_segments", test_path + f"/adni_t1_mprage")
                    PATH= PATH.replace(f"/{modelID}/gen_test_re256_mni", f"/{TestType}/MidSl/{dataset}{re_m}_Model/{modelID}/gen_test_re256_mni/pred_mask")

                print(PATH)

                os.makedirs(PATH.replace(os.path.basename(PATH), '')[:-1],exist_ok=True)
                PATH = PATH.replace(".png","_rot.nii.gz")
                MASK_PATH = PATH.replace("img","mask")
                

            else:   # #Internal Training and or Testing
                os.makedirs(path, exist_ok=True)
                PATH = path + f"/{i}_img_rot.nii.gz"
                MASK_PATH = path + f"/{i}_mask_rot.nii.gz"

            
            if rotate and flip:

                mask_pred[i] = rotating(mask_pred[i], k=1, axes=(1, 0))               #rotate 90 deg to the right one time
                image[i] = rotating(image[i], k=1, axes=(1, 0))                       #rotate 90 deg to the right one time
                mask_true[i] = rotating(mask_true[i], k=1, axes=(1, 0))               #rotate 90 deg to the right one time

                mask_pred[i] = flipping(mask_pred[i], axis=1)                         #axis = 1 flips horizontal
                image[i] = flipping(image[i], axis=1)                                 #axis = 1 flips horizontal
                mask_true[i] = flipping(mask_true[i], axis=1)                         #axis = 1 flips horizontal


            nib.save((nib.Nifti1Image(mask_pred[i].astype('int16'), None)), MASK_PATH)
            img = image[i]
            if img.shape[0] == 4:
                img = np.transpose(img,(1, 2, 0))
            nib.save((nib.Nifti1Image(img, None)), PATH) 
            
        #############


def flipping(data_squeeze, axis):
    """ Flips the image  """
    return np.flip(data_squeeze, axis)                                          
    
def rotating(data_squeeze, k, axes):
    """ Rotates the image 90 degrees to the left or right """
    return np.rot90(data_squeeze, k, axes)       
        
class Dice(torch.nn.Module):
    """
    Class used to get dice_loss and dice_score
    """

    def __init__(self, smooth=1):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = (y_pred - torch.min(y_pred)) / (torch.max(y_pred) - torch.min(y_pred))  
        y_true = (y_true - torch.min(y_true)) / (torch.max(y_true) - torch.min(y_true))  
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss, dice_score
