import torch
import torch.nn as nn
import torchvision.models

class LRASPP_mobilenet_v3(nn.Module):
    def __init__(self, model=torchvision.models.segmentation.lraspp_mobilenet_v3_large, num_classes=21):
        super(LRASPP_mobilenet_v3, self).__init__()
        self.net = model(num_classes=num_classes)     #https://programmerah.com/inceptionoutputs-object-has-no-attribute-log_softmax-27522/

    def forward(self, x):
        return self.net(x)


#to run it here from this script, uncomment the following
if __name__ == "__main__":                #to run it
    image = torch.rand(2, 1, 240, 240)    #specify your image: batch size, Channel, height, width
    model = LRASPP_mobilenet_v3(model=torchvision.models.segmentation.lraspp_mobilenet_v3_large, num_classes=14)
    model.eval()
    out = model(image)
    print(model(image))
