import os
from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Any, List

import numpy as np
import torch
import torchvision.models as models
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

from utilities.dataset import BrainDataset
from utilities.utils import (Dice, result_analyser)  
from utilities import dsc 

####for QuickNat
from model.QuickNat.quicknat import QuickNat
from nn_common_modules import losses as additional_losses
import h5py

##########Torch vision
from model.LRASPP_mobilenet_v3 import LRASPP_mobilenet_v3
from model.Fcn_resnet50 import Fcn_resnet50
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR
from glob import glob
from PIL import Image
import nibabel as nib

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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


seed_everything(1701, workers=True)

def criterion(inputs, target):
    losses = {}

    for name, x in inputs.items():
        losses[name] =  nn.functional.cross_entropy(x, target)  

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


class Segmenter(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.prepare_dataset_splits(**kwargs)


        if len(self.classIDs) == 2 and self.hparams.re_m == "_BV":    ###chosen remapped classes
            self.classes = ["Background", "Brain Volume"]
        
        elif len(self.classIDs) == 2 and self.hparams.re_m == "_IV":    ###chosen remapped classes
            self.classes = ["Background", "Intracranial Volume"]

        elif len(self.classIDs) == 3 and self.hparams.re_m == "_VV":    ###chosen remapped classes
            self.classes = ["Background", "Left Lateral ventricle", "Right Lateral Ventricle"]

        elif len(self.classIDs) == 3 and self.hparams.re_m == "_HV":    ###chosen remapped classes
            self.classes = ["Background", "Left Hippocampus", "Right Hippocampus"]

        elif len(self.classIDs) == 3 and self.hparams.re_m == "_CV":    ###chosen remapped classes
            self.classes = ["Background", "Left Cortex", "Right Cortex"]

        elif len(self.classIDs) == 7:    ###chosen remapped classes
            self.classes = ["Background", "Left Cortex", "Left Lateral ventricle", "Left Hippocampus", "Right Cortex", "Right Lateral Ventricle", "Right Hippocampus"]

        elif len(self.classIDs) == 9:    ###chosen remapped classes
            self.classes = ["Background", "Left WM", "Left Cortex", "Left Cerebellum WM", "Left Cerebellum Cortex", "Right WM", "Right Cortex", "Right Cerebellum WM", "Right Cerebellum Cortex"]

        elif len(self.classIDs) == 14:    ###chosen remapped classes
            self.classes = ["Background", "Left WM", "Left Cortex", "Left Lateral ventricle", "Left Cerebellum WM", "Left Cerebellum Cortex", "Brain Stem", "Left Hippocampus", "Right WM", "Right Cortex", "Right Lateral Ventricle", "Right Cerebellum WM", "Right Cerebellum Cortex", "Right Hippocampus"]

        elif len(self.classIDs) == 33:      ##all synthseg labels
            self.classes = ["Background", "Left WM", "Left Cortex", "Left Lateral ventricle", "Left Inf LatVentricle", "Left Cerebellum WM", "Left Cerebellum Cortex", "Left Thalamus", "Left Caudate", "Left Putamen", "Left Pallidum", "3rd Ventricle", "4th Ventricle", "Brain Stem", "Left Hippocampus", "Left Amygdala", "CSF (Cranial)", "Left Accumbens", "Left Ventral DC", "Right WM", "Right Cortex", "Right Lateral Ventricle", "Right Inf LatVentricle", "Right Cerebellum WM", "Right Cerebellum Cortex", "Right Thalamus", "Right Caudate", "Right Putamen", "Right Pallidum", "Right Hippocampus", "Right Amygdala", "Right Accumbens", "Right Ventral DC"]


        self.trans = [ transforms.ToTensor(), ]

    
        self.trans_aug = [
                            transforms.ToTensor(),
                            transforms.RandomRotation((-330, +330)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                        ]
        self.transform1 = self.trans_aug


        if kwargs['normmode'] == 0: #Z-Score normalisation
            self.trans.append(transforms.Normalize([self.DSmean], [self.DSstd]))
        elif kwargs['normmode'] == 1: #Divide by nth percentile (default: 99th)
            self.trans.append(transforms.Normalize([0], [self.DSperval]))
        elif kwargs['normmode'] == 2: #Divide by Max
            self.trans.append(transforms.Normalize([0], [self.DSmax]))


        ###Seg Network####

        self.hparams.net_params['num_class'] = len(self.classIDs)

        if kwargs['network'] == "quicknat":
            self.net = QuickNat(self.hparams.net_params)

        elif kwargs['network'] == "LRASPP_mobilenet_v3":
            self.net = LRASPP_mobilenet_v3(model=models.segmentation.lraspp_mobilenet_v3_large, num_classes=len(self.classIDs))

        elif kwargs['network'] == "Fcn_resnet50":
            self.net = Fcn_resnet50(model=models.segmentation.fcn_resnet50, num_classes=len(self.classIDs), aux_loss=True)


        # ## loss
        if kwargs['network'] == "quicknat":
            self.loss = additional_losses.CombinedLoss()         


        #Accuracy Metrics
        self.mask_accuracy = Dice()


    
    def configure_optimizers(self):

        if self.hparams.network == "quicknat":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
                weight_decay=self.hparams.w_d
            )

            return {
            'optimizer': optimizer,
            'monitor': 'val_loss',
            }

        else:

            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.w_d)

            iters_per_epoch = 7480
            scheduler = PolynomialLR(
            optimizer, total_iters=iters_per_epoch * (self.hparams.max_epochs - self.hparams.lr_warmup_epochs), power=0.9
            )

            return {
                'optimizer': optimizer,
                'scheduler' : scheduler,
                'monitor': 'val_loss',
            }



    def forward(self, x):
        return self.net(x)


    def training_step(self, batch, _):
        # image, weight, mask = batch
        image, mask = batch
        # weight, mask = torch.squeeze(weight), torch.squeeze(mask)
        mask = torch.squeeze(mask, dim=1)
        out = self(image)
        
        if self.hparams.network == "quicknat":
            loss = self.loss(out, mask, weight=None)
            _, batch_output = torch.max(out, dim=1)

        else:
            for name, x in out.items():
                _, batch_output = torch.max(x, dim=1)
            loss = criterion(out, mask)

        _, dice = self.mask_accuracy(batch_output, mask)
        dice_per_class = dsc.dice_score_perclass(batch_output, mask, num_classes=len(self.classIDs))


        if loss != loss:  #when loss = nan, loss is not a number so it is not equal itself
            print("loss is nan")
        self.log("loss", loss)


        del image, out, _
        torch.cuda.empty_cache()


        return {'loss': loss, 'batch_output': batch_output.cpu(), 'mask': mask.cpu(), 'dice': dice.cpu(), 'dice_per_class': dice_per_class.cpu()}



    def validation_step(self, batch, _):
        # image, weight, mask = batch
        image, mask = batch
        
        mask = torch.squeeze(mask, dim=1)

        out = self(image)

        if self.hparams.network == "quicknat":
            loss = self.loss(out, mask, weight=None)
            _, batch_output = torch.max(out, dim=1)

        else:
            for name, x in out.items():
                _, batch_output = torch.max(x, dim=1)
            loss = criterion(out, mask)


        _, dice = self.mask_accuracy(batch_output, mask)
        dice_per_class = dsc.dice_score_perclass(batch_output, mask, num_classes=len(self.classIDs))



        del image, out, _
        torch.cuda.empty_cache()
        return {'val_loss': loss, 'batch_output': batch_output.cpu(), 'mask': mask.cpu(), 'dice': dice.cpu(), 'dice_per_class': dice_per_class.cpu()}



    def test_step(self, batch, _):
        # image, weight, mask = batch
        image, mask = batch
        # weight, mask = torch.squeeze(weight), torch.squeeze(mask)
        mask = torch.squeeze(mask, dim=1)

        pred_mask = self(image)

        if self.hparams.network == "quicknat":
            loss = self.loss(pred_mask, mask, weight=None)
            _, batch_output = torch.max(pred_mask, dim=1)

        else:
            for name, x in pred_mask.items():
                _, batch_output = torch.max(x, dim=1)
            loss = criterion(pred_mask, mask)


        _, dice = self.mask_accuracy(batch_output, mask)
        dice_per_class = dsc.dice_score_perclass(batch_output, mask, num_classes=len(self.classIDs))

        del pred_mask, _
        torch.cuda.empty_cache()

        return {'test_loss': loss.cpu(), 'batch_output': batch_output.cpu(), 'mask': mask.cpu(), 'dice': dice.cpu(), 'image': image.cpu(), 'dice_per_class': dice_per_class.cpu()}


    def predict_step(self, batch, _):
        # image, weight, mask = batch
        image, mask = batch
        # weight, mask = torch.squeeze(weight), torch.squeeze(mask)
        mask = torch.squeeze(mask, dim=1)

        pred_mask = self(image)

        if self.hparams.network == "quicknat":
            loss = self.loss(pred_mask, mask, weight=None)
            _, batch_output = torch.max(pred_mask, dim=1)

        else:
            for name, x in pred_mask.items():
                _, batch_output = torch.max(x, dim=1)
            loss = criterion(pred_mask, mask)


        del image, pred_mask, _
        torch.cuda.empty_cache()

        return {'pred_loss': loss.cpu(), 'batch_output': batch_output.cpu(), 'mask':mask.cpu()}


    def rotating(data_squeeze, k, axes):
        """ Rotates the image 90 degrees to the left or right """
        return np.rot90(data_squeeze, k, axes)       
            
    def prepare_dataset_splits(self, **kwargs):

        if kwargs['resume_from_checkpoint']:
            self.test_typ = kwargs['test_typ']
            self.runMode = kwargs['runMode']
            self.test_path = kwargs['test_path']
            print(f"\n run Mode: {self.runMode}, Training_Data: {self.hparams.Dataset}, Test type: {self.test_typ} \n")

        if kwargs['runMode'] == 'onlyTest' and "-Wes" in kwargs['test_typ']:

            if "External" in kwargs['test_typ']:
                real_path = "Ukbiobank/imaging/brain_mri/t1_coronal_mni_new"  if "adni" in self.hparams.Dataset else "adni_t1_mprage/T1_3T_coronal_mni_nonlinear"
                fake_path = "/dhc/groups/fglippert/Ukbiobank/imaging/brain_mri" if "adni" in self.hparams.Dataset else "/dhc/cold/groups/syreal/ADNI" 

            elif "Internal" in kwargs['test_typ']:
                real_path = "adni_t1_mprage/T1_3T_coronal_mni_nonlinear" if "adni" in self.hparams.Dataset else "Ukbiobank/imaging/brain_mri/t1_coronal_mni_new" 
                fake_path = "/dhc/cold/groups/syreal/ADNI" if "adni" in self.hparams.Dataset else "/dhc/groups/fglippert/Ukbiobank/imaging/brain_mri"

            else:
                raise RuntimeError('sth wrong line 411')
            
            ####Internal and external only Test on Wes        
            if "Real" in  kwargs['test_typ']:   
                pth = f'/dhc/groups/fglippert/{real_path}/testset' 
                files = glob(pth + "/*/*.png")

            elif "Fake_Base" in  kwargs['test_typ']: 
                pth = fake_path +"/eval_segments/baseline_model/gen_test_re256_mni"  
                files = glob(pth + "/*.png")

            elif "Fake_Ms" in  kwargs['test_typ']: 
                pth = fake_path + "/eval_segments/ms_model/gen_test_re256_mni"
                files = glob(pth + "/*.png")

            print(f"\n Data Path: {pth} \n")

            self.X_test = []
            for file in files:
                img =Image.open(file)
                img = img.convert("L")    ##grey scale
                img = np.asarray(img)
                ###Test if rotation of images is correct
                # nib.save((nib.Nifti1Image(img, None)), "Test1.nii.gz") 
                if "adni" in self.hparams.Dataset:
                    if "Fake" in kwargs['test_typ']:
                        img = np.rot90(img, k=2, axes=(1,0))    ##rotate to the left to have adni testset rotated as the original ukbio trained images
                    else:
                        img = np.rot90(img, k=2, axes=(0,1)) 

                else:
                    if "Fake" in kwargs['test_typ']:
                        img = np.rot90(img, k=2, axes=(1,0))    ##rotate to the left to have adni testset rotated as the original ukbio trained images
                    else:
                        img = np.rot90(img, k=1, axes=(0,1))    ##rotate to the left to have adni testset rotated as the original ukbio trained images
                ###Test if rotation of images is correct
                # nib.save((nib.Nifti1Image(img, None)), "Test2.nii.gz") 
                img = np.expand_dims(img, axis=0)
                self.X_test.append(img)
            self.X_test=np.concatenate(self.X_test)
            # ##Test if rotation of images is correct
            # ll=self.X_test.transpose(1,2,0)
            # nib.save((nib.Nifti1Image(ll, None)), "Vol.nii.gz") 
            self.m_test=np.zeros((self.X_test.shape))
            self.files = files

            if kwargs['re_m'] == '_BV' or kwargs['re_m'] == '_IV':
                self.classIDs=[0,1]
            elif kwargs['re_m'] == '_VV' or kwargs['re_m'] == '_HV' or kwargs['re_m'] == '_CV':
                self.classIDs=[0,1,2]
            else:
                self.classIDs=[0,1,2,3,4,5,6]       ##old CV,VV,HV adni model

            print(f"\n region_mask: {kwargs['re_m']}, Classes ID: {self.classIDs} \n")


        else:   #Internal Training and or Testing
            print("Internal Training and or Testing")
            
            pth = f"/dhc/home/hadya.yassin/SyReal/2D_ROI/PreProcessing/{self.hparams.Dataset}/{self.hparams.extra}"

            data_params = {
                "data": pth + "/Data_train.h5",
                "label": pth + "/Label_train.h5",
                # "class_weights": pth + "/Class_Weight_train.h5",
            }

            print(f"Loading dataset ... {self.hparams.Dataset}")

            X = h5py.File(data_params['data'], 'r')
            X = X['data'][()]

            X = np.transpose(X, (2, 0, 1))

            ind_trainval, ind_test  = list(ShuffleSplit(n_splits=1, test_size=self.hparams.test_percent, random_state=13).split(X))[0]

            if kwargs['runMode'] == 'onlyTest':    
                self.X_test = X[ind_test] 
                del X
                m = h5py.File(data_params['label'], 'r')
                m = m['label'][()]
                m = np.transpose(m, (2, 0, 1))
                self.m_test = m[ind_test]
                self.classIDs = np.unique(self.m_test)
                del m
                # w = h5py.File(data_params['class_weights'], 'r')
                # w = w['class_weights'][()]
                # self.w_test = w[ind_test]
                # del w

            elif kwargs['runMode'] == 'onlyTrain':
                X = X[ind_trainval]

                self.DSmean, self.DSstd, self.DSperval, self.DSmax, self.DSmin = X.mean(), X.std(), np.percentile(X, self.hparams.percentile), X.max(), X.min()

                ind_train, ind_val  = list(ShuffleSplit(n_splits=self.hparams.n_folds, test_size=self.hparams.val_percent, random_state=42).split(X))[self.hparams.foldID]
                self.X_train = X[ind_train]
                self.X_val = X[ind_val]
                del X


                m = h5py.File(data_params['label'], 'r')
                m = m['label'][()]

                m = np.transpose(m, (2, 0, 1))
            
                m = m[ind_trainval]
                self.m_train = m[ind_train]
                self.m_val = m[ind_val]

                self.classIDs = np.unique(self.m_val)
                
                del m

                # w = h5py.File(data_params['class_weights'], 'r')
                # w = w['class_weights'][()]
                # w = w[ind_trainval]
                # self.w_train = w[ind_train]
                # self.w_val = w[ind_val]
                # del w

            elif kwargs['runMode'] == 'TrainTest':       ##
                self.X_test = X[ind_test] 
                X = X[ind_trainval]

                self.DSmean, self.DSstd, self.DSperval, self.DSmax, self.DSmin = X.mean(), X.std(), np.percentile(X, self.hparams.percentile), X.max(), X.min()

                ind_train, ind_val  = list(ShuffleSplit(n_splits=self.hparams.n_folds, test_size=self.hparams.val_percent, random_state=42).split(X))[self.hparams.foldID]
                self.X_train = X[ind_train]
                self.X_val = X[ind_val]
                del X


                m = h5py.File(data_params['label'], 'r')
                m = m['label'][()]

                m = np.transpose(m, (2, 0, 1))

                self.m_test = m[ind_test]
                m = m[ind_trainval]
                self.m_train = m[ind_train]
                self.m_val = m[ind_val]
                del m

                self.classIDs = np.unique(self.m_val)
        

                # w = h5py.File(data_params['class_weights'], 'r')
                # w = w['class_weights'][()]
                # self.w_test = w[ind_test]
                # w = w[ind_trainval]
                # self.w_train = w[ind_train]
                # self.w_val = w[ind_val]
                # del w

            else:
                raise RuntimeError("sth wrong in runMode")



    def train_dataloader(self) -> DataLoader:
        return DataLoader(BrainDataset(self.hparams.contrast ,self.X_train, self.m_train, maxinorm=True if self.hparams.normmode == 3 else False,
                          transforms=transforms.Compose(self.transform2 if self.hparams.MRIAug else self.transform1), trans=transforms.Compose(self.trans)),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)   

    def val_dataloader(self) -> DataLoader:
        return DataLoader(BrainDataset(self.hparams.contrast if not "Initial" in self.hparams.trainID else "t1", self.X_val, self.m_val, maxinorm=True if self.hparams.normmode == 3 else False, transforms=transforms.Compose(self.trans), trans=transforms.Compose(self.trans)),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers) 

    def test_dataloader(self) -> DataLoader:
        return DataLoader(BrainDataset(self.hparams.contrast if not "Initial" in self.hparams.trainID else "t1", self.X_test, self.m_test, maxinorm=True if self.hparams.normmode == 3 else False, transforms=transforms.Compose(self.trans), trans=transforms.Compose(self.trans)),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers) 

    def training_epoch_end(self, outputs: List[Any]) -> None:

        avg_loss = torch.stack([x['loss'] for x in outputs]).median()
        self.log('training_loss', avg_loss)

        self.log('training_dice_per_class', len(self.classIDs))

        avg_dice_per_class_0 = torch.stack([x['dice_per_class'][:][0] for x in outputs]).median()
        avg_dice_per_class_1 = torch.stack([x['dice_per_class'][:][1] for x in outputs]).median()

        if len(self.classIDs) == 3:
            avg_dice_per_class_2 = torch.stack([x['dice_per_class'][:][2] for x in outputs]).median()
            self.log('training_dice_per_class_2', avg_dice_per_class_2)
        elif len(self.classIDs) == 2:
            self.log('training_dice_per_class_0', avg_dice_per_class_0)
            self.log('training_dice_per_class_1', avg_dice_per_class_1)  
        else:
            raise RuntimeError(f"{len(self.classIDs)} > 3 or <2")



    def validation_epoch_end(self, outputs: List[Any]) -> None:

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).median()
        self.log('val_loss', avg_loss)

        self.log('validation_dice_per_class', len(self.classIDs))

        avg_dice_per_class_0 = torch.stack([x['dice_per_class'][:][0] for x in outputs]).median()
        avg_dice_per_class_1 = torch.stack([x['dice_per_class'][:][1] for x in outputs]).median()

        if len(self.classIDs) == 3:
            avg_dice_per_class_2 = torch.stack([x['dice_per_class'][:][2] for x in outputs]).median()
            self.log('validation_dice_per_class_2', avg_dice_per_class_2)
        elif len(self.classIDs) == 2:
            self.log('validation_dice_per_class_0', avg_dice_per_class_0)
            self.log('validation_dice_per_class_1', avg_dice_per_class_1)  
        else:
            raise RuntimeError(f"{len(self.classIDs)} > 3 or <2")


 
    def test_epoch_end(self, outputs: List[Any]) -> None:

        test_loss = torch.stack([x['test_loss'] for x in outputs]).median()
        self.log('test_loss', test_loss)

        self.log('test_dice_per_class', len(self.classIDs))

        avg_dice_per_class_0 = torch.stack([x['dice_per_class'][:][0] for x in outputs]).median()
        avg_dice_per_class_1 = torch.stack([x['dice_per_class'][:][1] for x in outputs]).median()

        if len(self.classIDs) == 3:
            avg_dice_per_class_2 = torch.stack([x['dice_per_class'][:][2] for x in outputs]).median()
            self.log('test_dice_per_class_2', avg_dice_per_class_2)
        elif len(self.classIDs) == 2:
            self.log('test_dice_per_class_0', avg_dice_per_class_0)
            self.log('test_dice_per_class_1', avg_dice_per_class_1)  
        else:
            raise RuntimeError(f"{len(self.classIDs)} > 3 or <2")


        mask = torch.cat([x['mask'] for x in outputs]).squeeze().numpy()
        pred_mask = torch.cat([x['batch_output'] for x in outputs]).squeeze().numpy()

        image = torch.cat([x['image'] for x in outputs]).squeeze().numpy()

        output_path = os.path.join(self.hparams.out_path, "Results", self.hparams.trainID)

        result_analyser(pred_mask, mask, self.classes, output_path, image, self.files, self.hparams.Dataset, self.runMode, self.test_typ, self.hparams.re_m, self.test_path) 


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False) ## add_help default = True
        parser.add_argument('--Dataset', type=str, required=True, help='Dataset, valid values are ukbiobank or adni_t1_mprage or Oasis3')
        parser.add_argument('--network', '-net', type=str, required=True, help='network, valid values are onlyTest with default best epoch or onlyTrain or TrainTest')
        parser.add_argument('--runMode', '-mode', type=str, required=True, help='run mode, valid values are Fcn_resnet50 (default Pretrained and rest of settings are optimized for it) or LRASPP_mobilenet_v3 or quicknat')
        parser.add_argument('--resume_from_chkp', '-resume', type=bool, required=True, help='resume_from_checkpoint, valid values are empty string "" for False or any string for True as default for pretrained models or False ')
        parser.add_argument('--mask_region', '-re_m', type=str, required=True, help='mask region, valid values are VV for Ventricle volujme or BV for Brain Volume or IV for Intracranial Volume or HV for Hippocampus Volume or CV for Cerebral Cortex Volume')
        parser.add_argument('--test_typ', type=str, required=True, help='test typ, valid values are Internal or External-Wes_Real or External-Wes_Fake_Base or External-Wes_Fake_Ms or Internal-Wes_Real or Internal-Wes_Fake_Base or Internal-Wes_Fake_Ms')
        parser.add_argument('--main_path', type=str, required=True, help='main_path, valid value is the path to your Preprocessing folder if PreProceesing.py was used other trace the code and specify your own')
        parser.add_argument('--out_path', type=str, required=True, help='out_path, valid value is the general path to the original output of the model whether checkpoints or test or prediction results')
        parser.add_argument('--test_path', type=str, required=True, help='test_path, valid value is the path to the specified test_typ output of the model')        
        parser.add_argument('--chkp_type', type=str, required=False, help='checkpoint type: resume from Best (default) or Last')
        parser.add_argument('--wnbactive', type=bool, required=False, help='wnbactive , valid values are empty string "" for False or any string for True')
        parser.add_argument('--num_epochs', type=int, required=False, help='num_epochs , valid values are 300 as default for pretrained models or any int number')
        parser.add_argument('--SliceType', '-SlTyp', type=str, required=False, help='slice type, valid value is midSl for middle slices ')
        parser.add_argument('--run_prefix', type=str)
        parser.add_argument('--contrast', type=str)
        parser.add_argument('--orient', type=str)
        parser.add_argument('--normmode', type=int)
        parser.add_argument('--n_folds', type=int)
        parser.add_argument('--foldID', type=int)
        parser.add_argument('--useClassWeight', type=bool, help='useClassWeight , valid values are empty string "" for False or any string for True ')
        parser.add_argument('--MRIAug', type=bool, help='MRIAug , valid values are empty string "" for False or any string for True ')
        parser.add_argument('--test_percent', type=float)
        parser.add_argument('--val_percent', type=float)
        parser.add_argument('--trainID', type=str)

        return parser


def main():

    torch.set_num_threads(1)

    parser = ArgumentParser()
    parser = Segmenter.add_model_specific_args(parser)
    args = parser.parse_args()
    parser = Trainer.add_argparse_args(parser)
    
    ###Shared hparams
    runMode: str = args.runMode if args.runMode else "onlyTest"                     #"onlyTest" default is best epoch# "onlyTrain"  # "TrainTest"
    resume_from_checkpoint = args.resume_from_chkp if args.resume_from_chkp is not None else (True if runMode == "onlyTest" else False)
    chkp_type = args.chkp_type if args.chkp_type else "Best"                        #checkpoint type: resume from "Best" or #"Last".
    wnbactive = args.wnbactive if args.wnbactive is not None else False
    network: str = args.network if args.network else "Fcn_resnet50"                 #"Fcn_resnet50": current defaul setttings are optimzed acording to it, other networks "LRASPP_mobilenet_v3", "quicknat"
    aux_loss: bool = True if network ==  "Fcn_resnet50" else False
    Dataset: str =  args.Dataset if args.Dataset else "adni_t1_mprage"              #ukbiobank adni_t1_mprage BRATS or Oasis3 
    test_typ: str = args.test_typ if args.test_typ else "External-Wes_Real"         # "Internal" or "External-Wes_Real" or "External-Wes_Fake_Base" or "External-Wes_Fake_Ms" or "Internal-Wes_Real" or "Internal-Wes_Fake_Base" or "Internal-Wes_Fake_Ms"
    num_epochs: int = args.num_epochs if args.num_epochs else 300                   #150
    Typ: str = args.SliceType if args.SliceType else 'midSl'                        #"midSl" = middle slices
    re_m: str = f"_{args.mask_region}" if args.mask_region else "_VV"               #"_BV" Brain volume without brain stem or "_VV" Ventricle Volume, "_IV" Intracranial Volume, "_HV" Hippocampus Volume or "_CV" Cerebral Cortex Volume
    
    if runMode == "onlyTest":
        run_prefix = f"{network}_150_e_b8" if (Dataset =="ukbiobank" and (re_m == "_BV" or re_m == "_VV")) else f"{network}_{num_epochs}_e_b8"
    else:
        run_prefix = f"{network}_{num_epochs}_e_b8"

    use_amp: bool = False                   #Automatic Mixed Precission
    batch_size:int = 4                    
    accumulate_grad_batches: int = 2       
    workers: int = 4                        
    normmode: int = 3                       #0: ZNorm, 1: Divide by nth percentile, 2: Divide by Max, 3 Max Norm
    percentile: int = 99                    #nth percentile to be used only for normmode 1
    lr:float = 0.01 if network ==  "Fcn_resnet50" else "specify it?"  #Default: 0.01 in torch vision models, 1e-4 in Segmentor QuickNat
    lr_warmup_decay: float =0.01
    lr_warmup_epochs: int =0
    lr_warmup_method: str ='linear'
    MRIAug: bool = args.MRIAug if args.MRIAug is not None else False                    # True only when addidtional Augumentation needed other than the basic available rotation, flip, ...
    useClassWeight: bool = args.useClassWeight if args.useClassWeight is not None else False            # True when class weights is calculated but it was proven to be memory heavy and casuses crashaes  
    main_path: str = args.main_path if args.main_path else '/dhc/home/hadya.yassin/SyReal/2D_ROI/PreProcessing'
    out_path: str = args.out_path if args.out_path else '/dhc/home/hadya.yassin/SyReal/2D_ROI/Segmenter/output'
    test_path: str = args.test_path if args.test_path else "/dhc/cold/groups/syreal/Segmentation_Tests"
    amount : str = "All" if Typ == "midSl" else "1340" ##or 656
    extra: str = f'Coronal/midSl_remap_{amount}{re_m}_h5'       
    contrast: str = "t1"
    orient: str = "Coronal"                 #"All", "Sag", "Coronal" for quicknat, "Axi"
    test_percent: float = 0.25
    val_percent: float = 0.20
    n_folds: int = 5
    foldID: int = 0                         #0, 1, 2, 3, 4
    which_Beste: int = -1                   #-1 -2 -3 the last best Epoch, -2 before last and ...
    gpu = 1                                 # number of used gpu

    print(f"\n This script's current run mode is: {runMode}, resume from chkp is: {resume_from_checkpoint}, network is: {network}, slice type set for training is {Typ} and Dataset {Dataset}, brain region of interest to be segmented is: {re_m}, performed Test Type: {test_typ}, \n \nplease note that the training settings have been optimized only for the pre-trained Fcn_resnet50 models. If one wishes to experiment with the other implemented networks must adjust the code accordingly")

    ####QuickNat
    net_params : dict = {'num_class': 7 if Dataset == "adni_t1_mprage" else 9,
                         'num_channels': 1,
                         'num_filters': 64,
                         'kernel_h': 5,
                         'kernel_w': 5,
                         'kernel_c': 1,
                         'stride_conv': 1,
                         'pool': 2,
                         'stride_pool': 2,
                         'se_block': 'CSSE',
                         'drop_out': 0.2}

    betas: tuple = (0.9, 0.999) #=Default Adam
    eps: int = 1e-8 #=Default Adam
    momentum: float =0.9

    ###in case of overfitting
    w_d:float = 1e-06
    g_c_val: int = 0
    g_c_algo: str = 'norm'                  #'norm' by default, or set to "value"
    #######

    ##Defaults##
    autoScaleBatchSize = False          #Default: False
    findLR = False                      #Default: False

    
    if (Dataset =="ukbiobank" and (re_m == "_BV" or re_m == "_VV")): 
        trainID = run_prefix + "_" + Dataset + "_" + Typ + "_" + re_m + contrast + "_" + orient + "_" + str(g_c_val) + "-" + str(g_c_algo) + "_w_d-" + str(w_d) + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
            "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)

    else: ###Standard 
        trainID = run_prefix + "_" + Dataset + "_" + Typ + re_m + "_" + contrast + "_" + orient + "_" + str(g_c_val) + "-" + str(g_c_algo) + "_w_d-" + str(w_d) + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
            "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)

    print(f"trainID: {trainID}")

    if Typ == "midSl" and "midSl" in trainID:
        print("...")
    else:
        raise RuntimeError("trianTD doesn't match Typ")


    hparams = parser.parse_args()
    hparams.run_prefix = run_prefix ####
    hparams.lr = lr
    hparams.w_d = w_d
    hparams.g_c_val = g_c_val
    hparams.g_c_algo = g_c_algo
    hparams.max_epochs = num_epochs
    hparams.batch_size = batch_size
    hparams.trainID = trainID
    hparams.normmode = normmode
    hparams.percentile = percentile
    hparams.network = network
    hparams.accumulate_grad_batches = accumulate_grad_batches
    hparams.use_amp = use_amp
    hparams.workers = workers
    hparams.main_path = main_path
    hparams.out_path = out_path
    hparams.Dataset = Dataset
    hparams.contrast = contrast
    hparams.orient = orient
    hparams.test_percent = test_percent
    hparams.val_percent = val_percent
    hparams.n_folds = n_folds
    hparams.foldID = foldID
    hparams.MRIAug = MRIAug
    hparams.autoScaleBatchSize = autoScaleBatchSize
    hparams.findLR = findLR
    hparams.runMode = runMode
    hparams.net_params = net_params
    hparams.betas = betas
    hparams.eps = eps
    hparams.extra = extra
    hparams.momentum = momentum
    hparams.lr_warmup_decay = lr_warmup_decay
    hparams.lr_warmup_epochs = lr_warmup_epochs
    hparams.lr_warmup_method = lr_warmup_method
    hparams.aux_loss = aux_loss
    hparams.Typ = Typ
    hparams.re_m = re_m


    #specify which checkpoint
    if resume_from_checkpoint:
        if chkp_type == "Best":
            checkpoint_dir = pjoin(hparams.out_path, "Checkpoints", hparams.trainID)
            chkpoint = pjoin(checkpoint_dir, sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1])
        elif chkp_type == "Last":
            #method 1
            # checkpoint_dir = pjoin(hparams.out_path, "Checkpoints", hparams.trainID)
            # base = sorted([x for x in os.listdir(checkpoint_dir) if "last" in x])[0]
            # chkpoint = pjoin(hparams.out_path, "Checkpoints", hparams.trainID, "last-v1.ckpt")
            ## safer method 2
            checks = glob(pjoin(hparams.out_path, "Checkpoints", hparams.trainID, "*.ckpt"))
            for check in checks:
                check.replace("."," ").replace("-", " ")
                v1 = True if "v1" in check else False
            chkpoint = pjoin(hparams.out_path, "Checkpoints", hparams.trainID, "last-v1.ckpt" if v1 else "last.ckpt")
    else:
        chkpoint = None


    if not wnbactive:
        os.environ["WANDB_MODE"] = "dryrun"

    # Only test using the best model!
    if runMode == "onlyTest":

        #or load the best chckpoint as a default
        checkpoint_dir = pjoin(hparams.out_path, "Checkpoints", hparams.trainID)
        folder_name = sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[which_Beste]
        chkpoint = pjoin(checkpoint_dir, folder_name)
        print(f"\n Only Test chkp: {chkpoint} \n")
        folder_name = folder_name.replace("=", " ")
        folder_name = folder_name.replace("-", " ")
        epochs = [int(s) for s in folder_name.split() if s.isdigit()]

        if chkpoint == None:
            raise RuntimeError("the next line will produce an error, resume_from_Checkpoint is set to False")

        model = Segmenter.load_from_checkpoint(chkpoint, runMode=runMode, resume_from_checkpoint=resume_from_checkpoint, test_typ=test_typ, test_path=test_path)

        model.hparams.max_epochs = epochs[0]
    
    else:
        model = Segmenter(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(hparams.out_path, "Checkpoints", hparams.trainID),
        monitor='val_loss',
        save_last=True,
        # save_top_k = -1,  #added to save every epoch as a checkpoint for experimental purposes
    )

    logger = WandbLogger(name=trainID, id=trainID, project='2DROI_Hadya',
                            group='Baseline', entity='hyassin_hpi', config=hparams)
    logger.watch(model, log='all', log_freq=100)


    # train
    trainer = Trainer(
        logger=logger,
        precision=16 if use_amp else 32,
        gpus= gpu,     #gpus=1 or specific [9] for example
        # checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        max_epochs=hparams.max_epochs,
        # terminate_on_nan=True,
        deterministic=False,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_from_checkpoint=chkpoint,
        auto_scale_batch_size='binsearch' if autoScaleBatchSize else None,
        auto_lr_find=findLR,
        gradient_clip_val=g_c_val,
        gradient_clip_algorithm=g_c_algo
    )

    # Only train
    if runMode == "onlyTrain":
        if autoScaleBatchSize or findLR:
            trainer.tune(model)
            
        trainer.fit(model)


    # Only test using the best model!
    elif runMode == "onlyTest":
        #onlyTest
        trainer.test(model, dataloaders=model.test_dataloader())


    #else training and testing (resuming training or from scratch)
    elif runMode == "TrainTest":
        if autoScaleBatchSize or findLR:
            trainer.tune(model)

        trainer.fit(model)
        trainer.test(dataloaders=model.test_dataloader())


if __name__ == '__main__':
    main()
