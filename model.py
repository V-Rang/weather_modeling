import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):  
    def __init__(self, in_len, out_len):
        super(SegNet, self).__init__()
        encoder = nn.Sequential(
            nn.Conv2d(in_len,5, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            nn.BatchNorm2d(5, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(5,10, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            nn.BatchNorm2d(10, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=0, ceil_mode=False),
            
            nn.Conv2d(10,15, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            nn.BatchNorm2d(15, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(15,15, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            nn.BatchNorm2d(15, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=0, ceil_mode=False),

            nn.Conv2d(15,20, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            nn.BatchNorm2d(20, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(20,20, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            nn.BatchNorm2d(20, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(20, 20, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            nn.BatchNorm2d(20, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=0, ceil_mode=False)
        )

        decoder =  encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        decoder[-1] = nn.Conv2d(out_len, 5, kernel_size=3, stride=1, padding=1)
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        for i, module in enumerate(decoder):
                    if isinstance(module, nn.Conv2d):
                        if module.in_channels != module.out_channels:
                            decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                            decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.stage1_decoder = nn.Sequential(*decoder[:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:15])
        self.stage3_decoder = nn.Sequential(*decoder[15:])
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        #encoder part
        x = self.stage1_encoder(x)
        #need to preserve size ouput for unpooling in decoder stage
        x1_size = x.size()
        x, indices1 = self.pool(x)
        
        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        # decoder part
        x = self.unpool(x, indices = indices3, output_size = x3_size)
        x = self.stage1_decoder(x)
        
        x = self.unpool(x, indices = indices2, output_size = x2_size)
        x = self.stage2_decoder(x)
        
        x = self.unpool(x, indices = indices1, output_size = x1_size)
        x = self.stage3_decoder(x)
        
        return x
