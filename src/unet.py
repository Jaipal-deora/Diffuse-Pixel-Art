import torch 
from torch import nn 
import torch.nn.functional as F 

class UNET_ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, is_res:bool=False) -> None:
        super().__init__()
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(8,out_channels),
            nn.SiLU()
        )
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.GroupNorm(8,out_channels),
            nn.SiLU()
        )
        self.is_res = is_res
        if is_res:
            if in_channels == out_channels:
                self.residual_layer = nn.Identity()
            else:
                self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1_layer(x)
            x2 = self.conv2_layer(x1)
            out = x2 + self.residual_layer(x)
            return out/1.414 
        else:
            return self.conv2_layer(self.conv1_layer(x))

class UNET_Upsample(nn.Module):
    def __init__(self,channels:int) -> None:
        super().__init__()
        layers = [UNET_ConvBlock(channels,channels),
                  UNET_ConvBlock(channels,channels)]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x:torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # concat skip connection with input 
        x = torch.cat((x,skip),axis=1)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.model(x)
    
class UNET_Downsample(nn.Module):
    def __init__(self,in_channels:int, out_channels:int) -> None:
        super().__init__()
        
        layers = [UNET_ConvBlock(in_channels,out_channels),
                  UNET_ConvBlock(out_channels,out_channels),
                  nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class EmbedFC(nn.Module):
    def __init__(self,input_dim:int, embed_dim:int) -> None:
        super().__init__()
        self.input_dim = input_dim 

        layers = [nn.Linear(input_dim,embed_dim),
                  nn.SiLU(),
                  nn.Linear(embed_dim,embed_dim)]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.view(-1,self.input_dim)
        return self.model(x)

class ContextUNET(nn.Module):
    def __init__(self, in_channels:int, n_feat:int=64, n_cfeat:int = 5,height:int = 16) -> None:
        super().__init__()

        self.in_channels = in_channels 
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.height = height # height == width (assumption)

        # intitialize conv 
        self.init_conv = UNET_ConvBlock(in_channels,n_feat,is_res=True) # 16x16x3 -> 16x16x64
        
        self.down1 = UNET_Downsample(n_feat,n_feat) # 8x8x64
        self.down2 = UNET_Downsample(n_feat,2*n_feat) # 8x8x64 -> 4x4x128 

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)),
                                    nn.SiLU()) # 1x1x128
        
        # embed timestep & context label with a one layer fcc 
        self.timeembed1 = EmbedFC(1, 2*n_feat) # 1x128
        self.timeembed2 = EmbedFC(1, 1*n_feat) # 1x64 
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat) # 5x128
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat) # 5x64

        # start upsampling 
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, self.height//4,self.height//4),
            nn.GroupNorm(8, 2*n_feat), # normalize 
            nn.GELU()
        ) # --> 4x4x128

        self.up1 = UNET_Upsample(4*n_feat) # 8x8x256
        
        self.up1_red = UNET_ConvBlock(4*n_feat, n_feat)#8x8x64
        self.up2 = UNET_Upsample(2*n_feat) #  16x16x128
        self.up2_red = UNET_ConvBlock(2*n_feat, n_feat) # 16x16x64

        # initialize final conv to map same number of channels as input
        self.out = nn.Sequential(
            nn.Conv2d(2*n_feat,n_feat,3,1,1),
            nn.GroupNorm(8,n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat,self.in_channels,3,1,1)
        ) # 

    def forward(self,x:torch.Tensor, t: torch.Tensor, c: torch.Tensor=None) -> torch.Tensor:
        """
        x: (B,C,H,W) : input image
        t: (B, n_feat): time step 
        c: (B, n_cfeat): context label
        """ 
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0],self.n_cfeat).to(x)
        
        # embed timestep and context 
        cemb1 = self.contextembed1(c).view(-1,self.n_feat*2,1,1) # (B,2*n_feat,1,1)
        temb1 = self.timeembed1(t).view(-1,self.n_feat*2,1,1)
        cemb2 = self.contextembed2(c).view(-1,self.n_feat,1,1)
        temb2 = self.timeembed2(t).view(-1,self.n_feat,1,1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1_red(self.up1(cemb1*up1+temb1,down2))
        up3 = self.up2_red(self.up2(cemb2*up2+temb2, down1))
        out = self.out(torch.cat((up3,x),axis=1))
        return out
    
# if __name__ == '__main__':
#     # params 
#     timesteps = 500
#     n_feat = 64 
#     n_cfeat = 5 
#     height = 16

#     unet = ContextUNET(3,n_feat,n_cfeat,height)
#     x = torch.randn(1,3,height,height)
#     t = torch.tensor([500.0])[:,None,None,None]
#     c = None 
#     out = unet(x,t,c)
#     print(f"INPUT SHAPE: {x.shape} \n {t.shape}")
#     # convBlock = UNET_ConvBlock(3,64,True)
#     # out = convBlock(x)
#     print(f'OUTPUT SHAPE: {out.shape}')