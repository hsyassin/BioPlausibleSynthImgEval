"""
Deeper Unet than the originally proposed one for pytorch


MIT License

Copyright (c) 2019 mateuszbuda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



https://github.com/mateuszbuda/brain-segmentation-pytorch
"""



from collections import OrderedDict

import torch
import torch.nn as nn


class DD_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=96):
        super(DD_UNet, self).__init__()

        self.out_channels = out_channels

        features = init_features
        self.encoder1 = DD_UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DD_UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DD_UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DD_UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = DD_UNet._block(features * 8, features * 16, name="enc5")  # New layer 1
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # New layer 1
        self.encoder6 = DD_UNet._block(features * 16, features * 32, name="enc6")  # New layer
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)  # New layer


        self.bottleneck = DD_UNet._block(features * 32, features * 64, name="bottleneck")

        self.upconv6 = nn.ConvTranspose2d(
            features * 64, features * 32, kernel_size=2, stride=2)  # New layer
        
        self.decoder6 = DD_UNet._block((features * 32) * 2, features * 32, name="dec6")  # New layer
        self.upconv5 = nn.ConvTranspose2d(
            features * 32, features * 16, kernel_size=2, stride=2  # New layer 2
        )
        self.decoder5 = DD_UNet._block((features * 16) * 2, features * 16, name="dec5")  # New layer 2
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = DD_UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = DD_UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = DD_UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = DD_UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))  # New layer 1
        enc6 = self.encoder6(self.pool5(enc5))  # New layer

        bottleneck = self.bottleneck(self.pool6(enc6))  # Updated for new layer

        dec6 = self.upconv6(bottleneck)  # New layer
        dec6 = torch.cat((dec6, enc6), dim=1)  # New layer
        dec6 = self.decoder6(dec6)  # New layer
        dec5 = self.upconv5(dec6)  # New layer 2
        dec5 = torch.cat((dec5, enc5), dim=1)  # New layer 2
        dec5 = self.decoder5(dec5)  # New layer 2
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.softmax(self.conv(dec1), dim=1) if self.out_channels > 2 else torch.sigmoid(self.conv(dec1)) # Use softmax for multi-class segmentation

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    


#to run it here from this script, uncomment the following
if __name__ == "__main__":                #to run it
    image = torch.rand(2, 1, 256, 256)    #specify your image: batch size, Channel, height, width
    model = DD_UNet(in_channels=image.shape[1], out_channels=1)
    model.train()
    out = model(image)
    print(model(image))