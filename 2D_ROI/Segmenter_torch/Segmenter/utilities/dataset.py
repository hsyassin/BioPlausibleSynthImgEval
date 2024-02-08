from torch.utils.data import Dataset
import numpy as np


class BrainDataset(Dataset): 
    def __init__(self, contrast, images, masks, maxinorm=False, transforms=None, trans=None):        
        self.X = images # images        
        self.m = masks # masks
        self.maxinorm = maxinorm
        self.transforms = transforms
        self.contrast = contrast
        self.trans = trans

    def __len__(self):
        return len(self.X) # return length of image samples

    def __getitem__(self, idx):
        image = self.X[idx].astype(np.float32)
        mask = np.expand_dims(self.m[idx].astype(np.int64), 0)

        if self.transforms:

            image_trans = self.transforms(image)
            
            ##save some examples after transformation to check correctness of the operation
            # image_numpy = np.transpose(image_trans,(1, 2, 0))
            # image_numpy = np.array(image_numpy)
            # images2 = nib.Nifti1Image(image_numpy, None)      #np.eye(4)
            # # nib.save((nib.Nifti1Image(image_numpy, None)), main_path + f'TestImg.nii.gz') 


        if self.maxinorm:

            if image_trans.max()==0.0:
                print(f"max of this array with indx {idx} is zero after transform, therefore this image won't be transformed")
                image = self.trans(image)   #only converts np to Tensor 
                image /= image.max()
                if image.max()==0.0:
                    raise RuntimeError(f"max of this array with indx {idx} is zero after ToTensor Trans, this is not acceptable and will result in loss is nan")

            else:
                image_trans /= image_trans.max()
                image = image_trans
        
        return image, mask