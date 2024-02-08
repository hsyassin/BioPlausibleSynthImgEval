import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

class Fcn_resnet50(nn.Module):
    def __init__(self, model=torchvision.models.segmentation.fcn_resnet50, num_classes=21, aux_loss=True):
        super(Fcn_resnet50, self).__init__()
        self.net = model(num_classes=num_classes, aux_loss=True)     #https://programmerah.com/inceptionoutputs-object-has-no-attribute-log_softmax-27522/

    def forward(self, x):
        return self.net(x)


#to run it here from this script, uncomment the following
if __name__ == "__main__":                #to run it
    image = torch.rand(2, 1, 240, 240)    #specify your image: batch size, Channel, height, width
    # model = Fcn_resnet50(model=torchvision.models.segmentation.fcn_resnet50, num_classes=14, ch=1)
    model = Fcn_resnet50(model=torchvision.models.segmentation.fcn_resnet50, num_classes=3, aux_loss=True)
    model.train()
    out = model(image)
    print(model(image))
