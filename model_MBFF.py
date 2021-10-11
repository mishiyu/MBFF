import torch
import torch.nn.functional as F
from torch import nn

from resnext import ResNeXt101


class MBFF(nn.Module):
    def __init__(self):
        super(MBFF, self).__init__()
        self.resnext = ResNeXt101()

        self.line = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.line_predict = nn.Conv2d(64, 1, kernel_size=1)

        self.line_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.attention = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        )

        self.dilation1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.dilation3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.preconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.SFT = SFTLayer(64)

        self.refine = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.predict = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x,is_train):
        x, bw = x
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)
        print(x.size())
        print(layer0.size())
        print(layer1.size())
        print(layer2.size())
        print(layer3.size())
        print(layer4.size())

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        predict4 = self.predict(down4)
        predict3 = self.predict(down3)
        predict2 = self.predict(down2)
        predict1 = self.predict(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        branch11 = self.fuse2(torch.cat((fuse1, down1), 1))
        branch12 = self.fuse2(torch.cat((fuse1, down2), 1))
        branch13 = self.fuse2(torch.cat((fuse1, down3), 1))
        branch14 = self.fuse2(torch.cat((fuse1, down4), 1))

        attention4 = self.attention(branch14)
        attention3 = self.attention(branch13)
        attention2 = self.attention(branch12)
        attention1 = self.attention(branch11)

        branch243 = self.dilation3(down4)
        branch242 = self.dilation2(down4)
        branch241 = self.dilation1(down4)
        branch243 = F.upsample(branch243, size=down1.size()[2:], mode='bilinear')
        branch242 = F.upsample(branch242, size=down1.size()[2:], mode='bilinear')
        branch241 = F.upsample(branch241, size=down1.size()[2:], mode='bilinear')
        branch24 = self.fuse3(torch.cat((branch243, branch242, branch241), 1))

        branch233 = self.dilation3(down3)
        branch232 = self.dilation2(down3)
        branch231 = self.dilation1(down3)
        branch233 = F.upsample(branch233, size=down1.size()[2:], mode='bilinear')
        branch232 = F.upsample(branch232, size=down1.size()[2:], mode='bilinear')
        branch231 = F.upsample(branch231, size=down1.size()[2:], mode='bilinear')
        branch23 = self.fuse3(torch.cat((branch233, branch232, branch231), 1))

        branch223 = self.dilation3(down2)
        branch222 = self.dilation2(down2)
        branch221 = self.dilation1(down2)
        branch223 = F.upsample(branch223, size=down1.size()[2:], mode='bilinear')
        branch222 = F.upsample(branch222, size=down1.size()[2:], mode='bilinear')
        branch221 = F.upsample(branch221, size=down1.size()[2:], mode='bilinear')
        branch22 = self.fuse3(torch.cat((branch223, branch222, branch221), 1))

        branch213 = self.dilation3(down1)
        branch212 = self.dilation2(down1)
        branch211 = self.dilation1(down1)
        branch213 = F.upsample(branch213, size=down1.size()[2:], mode='bilinear')
        branch212 = F.upsample(branch212, size=down1.size()[2:], mode='bilinear')
        branch211 = F.upsample(branch211, size=down1.size()[2:], mode='bilinear')
        branch21 = self.fuse3(torch.cat((branch213, branch212, branch211), 1))

        bw = self.preconv(bw)
        branch34 = self.SFT([down4,bw])
        branch33 = self.SFT([down3,bw])
        branch32 = self.SFT([down2,bw])
        branch31 = self.SFT([down1,bw])

        refine4 = self.refine2(self.fuse3(torch.cat((attention4 + down4, branch24, branch34), 1)))
        refine3 = self.refine2(self.fuse3(torch.cat((attention3 + down3, branch23, branch33), 1)))
        refine2 = self.refine2(self.fuse3(torch.cat((attention2 + down2, branch22, branch32), 1)))
        refine1 = self.refine2(self.fuse3(torch.cat((attention1 + down1, branch21, branch31), 1)))
        
        predict4_1 = self.predict(refine4)
        predict3_1 = self.predict(refine3)
        predict2_1 = self.predict(refine2)
        predict1_1 = self.predict(refine1)

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        predict1_1 = F.upsample(predict1_1, size=x.size()[2:], mode='bilinear')
        predict2_1 = F.upsample(predict2_1, size=x.size()[2:], mode='bilinear')
        predict3_1 = F.upsample(predict3_1, size=x.size()[2:], mode='bilinear')
        predict4_1 = F.upsample(predict4_1, size=x.size()[2:], mode='bilinear')

        fuse1 = F.upsample(fuse1, size=layer0.size()[2:], mode='bilinear')
        line_feature = self.line(torch.cat((layer0, fuse1), 1))
        #line_feature = layer0
        refine1 = F.upsample(refine1, size=layer0.size()[2:], mode='bilinear')
        refine2 = F.upsample(refine2, size=layer0.size()[2:], mode='bilinear')
        refine3 = F.upsample(refine3, size=layer0.size()[2:], mode='bilinear')
        refine4 = F.upsample(refine4, size=layer0.size()[2:], mode='bilinear')
              
        refine4_2 = self.line_conv(torch.cat((refine4, line_feature), 1))
        refine3_2 = self.line_conv(torch.cat((refine3, line_feature), 1))
        refine2_2 = self.line_conv(torch.cat((refine2, line_feature), 1))
        refine1_2 = self.line_conv(torch.cat((refine1, line_feature), 1))
        
        predict4_2 = self.predict(refine4_2)
        predict3_2 = self.predict(refine3_2)
        predict2_2 = self.predict(refine2_2)
        predict1_2 = self.predict(refine1_2)

        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')

        line_predict = self.line_predict(line_feature)
        line_predict = F.upsample(line_predict, size=x.size()[2:], mode='bilinear')

        if is_train:
            return line_predict, predict1, predict2, predict3, predict4, predict1_1, predict2_1, predict3_1, predict4_1, predict1_2, predict2_2, predict3_2, predict4_2
        else:
            return F.sigmoid((predict1_1 + predict2_1 + predict3_1 + predict4_1) / 4)

class SFTLayer(nn.Module):
    def __init__(self, nfeat):
        super(SFTLayer,self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(nfeat,nfeat,1)
        self.SFT_scale_conv1 = nn.Conv2d(nfeat,nfeat,1)
        self.SFT_shift_conv0 = nn.Conv2d(nfeat,nfeat,1)
        self.SFT_shift_conv1 = nn.Conv2d(nfeat,nfeat,1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0]*(scale+1)+shift