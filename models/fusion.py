# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TransformerBlock(nn.Module):
#     def __init__(self, in_channels, num_heads=1, dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.norm1 = nn.LayerNorm(in_channels)
#         self.self_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)
#         self.norm2 = nn.LayerNorm(in_channels)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(in_channels, in_channels * 4),
#             nn.GELU(),
#             nn.Linear(in_channels * 4, in_channels)
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = x + self.self_attn(x, x,x)[0]
#         x = self.norm1(x)
#         x = x + self.feed_forward(x)
#         x = self.norm2(x)
#         return x

# class FusionModel(nn.Module):
#     def __init__(self, ratio=0.6):
#         super(FusionModel, self).__init__()
#         c1 = int(64 * ratio)
#         c2 = int(128 * ratio)
#         c3 = int(256 * ratio)
#         c4 = int(512 * ratio)
#         # cT=int (512)

#         self.block1 = Block([c1, c1, 'M'], in_channels=3, first_block=True)
#         self.block2 = Block([c2, c2, 'M'], in_channels=c1)
#         self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
#         self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
#         self.block5 = Block([c4, c4, c4, c4], in_channels=c4)

#         # 在 FusionModel 的 __init__ 方法中添加一个 1x1 卷积层
#         # self.conv_adjust = nn.Conv2d(307, 512, kernel_size=1)

#         # self.rgb_transformer = TransformerBlock(in_channels=cT)
#         # self.t_transformer = TransformerBlock(in_channels=cT)

#         self.reg_layer = nn.Sequential(
#             nn.Conv2d(c4, c3, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c3, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 1, 1)
#         )
#         self._initialize_weights()

#     def forward(self, RGBT):
#         RGB = RGBT[0]
#         T = RGBT[1]

#         RGB, T, shared = self.block1(RGB, T, None)
#         RGB, T, shared = self.block2(RGB, T, shared)
#         RGB, T, shared = self.block3(RGB, T, shared)
#         RGB, T, shared = self.block4(RGB, T, shared)
#         _, _, shared = self.block5(RGB, T, shared)
#         x = shared

#         x = self.reg_layer(x)
#         return torch.abs(x)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

# class Block(nn.Module):
#     def __init__(self, cfg, in_channels, first_block=False, dilation_rate=1):
#         super(Block, self).__init__()
#         self.seen = 0
#         self.first_block = first_block
#         self.d_rate = dilation_rate

#         self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
#         self.t_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
#         if first_block is False:
#             self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

#         channels = cfg[0]
#         self.rgb_msc = MSC(channels)
#         self.t_msc = MSC(channels)
#         if first_block is False:
#             self.shared_fuse_msc = MSC(channels)
#         self.shared_distribute_msc = MSC(channels)

#         self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
#         self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
#         self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
#         self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

#         # 添加 TransformerBlock
#         # self.rgb_transformer = TransformerBlock(channels)
#         # self.t_transformer = TransformerBlock(channels)

#     def forward(self, RGB, T, shared):
#         RGB = self.rgb_conv(RGB)
#         T = self.t_conv(T)
#         if self.first_block:
#             shared = torch.zeros(RGB.shape).cuda()
#         else:
#             shared = self.shared_conv(shared)

#         # 在融合之前应用 TransformerBlock
#         # RGB = self.rgb_transformer(RGB.flatten(2).permute(2, 0, 1)).permute(1, 2, 0).view_as(RGB)
#         # T = self.t_transformer(T.flatten(2).permute(2, 0, 1)).permute(1, 2, 0).view_as(T)


#         new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)
#         return new_RGB, new_T, new_shared
############################################################################################################
    # def __init__(self, cfg, in_channels, first_block=False, dilation_rate=1):
    #     super(Block, self).__init__()
    #     self.first_block = first_block
    #     self.d_rate = dilation_rate

    #     self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
    #     self.t_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
    #     if first_block is False:
    #         self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

    #     channels = cfg[0]

    #     # 多尺度特征提取
    #     self.rgb_msc = MSC(channels)
    #     self.t_msc = MSC(channels)
    #     if first_block is False:
    #         self.shared_fuse_msc = MSC(channels)
    #     self.shared_distribute_msc = MSC(channels)

    #     # 1x1 卷积用于跨模态信息融合
    #     self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
    #     self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
    #     self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
    #     self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    #     # 1x1 卷积用于通道降维
    #     self.channel_reduction = nn.Conv2d(channels, channels // 2, kernel_size=1)

    #     # Transformer Block（用于增强共享特征）
    #     self.shared_transformer = TransformerBlock(in_channels=channels)

    # def forward(self, RGB, T, shared):
    #     RGB = self.rgb_conv(RGB)
    #     T = self.t_conv(T)

    #     if self.first_block:
    #         shared = torch.zeros(RGB.shape).cuda()
    #     else:
    #         shared = self.shared_conv(shared)

    #     # 计算多尺度特征
    #     RGB_m = self.rgb_msc(RGB)
    #     T_m = self.t_msc(T)
    #     if self.first_block:
    #         shared_m = shared
    #     else:
    #         shared_m = self.shared_fuse_msc(shared)

    #     new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)

        # new_shared_flatten = new_shared.flatten(2).permute(2, 0, 1)  # 调整维度以适应 TransformerBlock
        # new_shared_attn = self.shared_transformer(new_shared_flatten)
        # new_shared_attn = new_shared_attn.permute(1, 2, 0).view_as(new_shared)  # 恢复原始维度

        # return new_RGB, new_T, new_shared
        # 单独对 new_shared 进行 Transformer 的处理
        # original_shape = new_shared.shape  # 记录原始形状
        # new_shared_flatten = new_shared.flatten(2).permute(2, 0, 1)  # 调整维度以适应 TransformerBlock 输入要求
        # new_shared_transformed = self.shared_transformer(new_shared_flatten)  # 进行 Transformer 处理
        # new_shared = new_shared_transformed.permute(1, 2, 0).view(original_shape)  # 恢复原始形状

        # return new_RGB, new_T, new_shared
############################################################################################################

#     def fuse(self, RGB, T, shared):
#         RGB_m = self.rgb_msc(RGB)
#         T_m = self.t_msc(T)
#         if self.first_block:
#             shared_m = shared
#         else:
#             shared_m = self.shared_fuse_msc(shared)

#         rgb_s = self.rgb_fuse_1x1conv(RGB_m - shared_m)
#         rgb_fuse_gate = torch.sigmoid(rgb_s)
#         t_s = self.t_fuse_1x1conv(T_m - shared_m)
#         t_fuse_gate = torch.sigmoid(t_s)
#         new_shared = shared + (RGB_m - shared_m) * rgb_fuse_gate + (T_m - shared_m) * t_fuse_gate

#         new_shared_m = self.shared_distribute_msc(new_shared)
#         s_rgb = self.rgb_distribute_1x1conv(new_shared_m - RGB_m)
#         rgb_distribute_gate = torch.sigmoid(s_rgb)
#         s_t = self.t_distribute_1x1conv(new_shared_m - T_m)
#         t_distribute_gate = torch.sigmoid(s_t)
#         new_RGB = RGB + (new_shared_m - RGB_m) * rgb_distribute_gate
#         new_T = T + (new_shared_m - T_m) * t_distribute_gate

#         return new_RGB, new_T, new_shared

# class MSC(nn.Module):
#     def __init__(self, channels):
#         super(MSC, self).__init__()
#         self.channels = channels
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

#         self.conv = nn.Sequential(
#             nn.Conv2d(3*channels, channels, kernel_size=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
#         x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
#         concat = torch.cat([x, x1, x2], 1)
#         fusion = self.conv(concat)
#         return fusion

# def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
#     layers = []
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)



###################################################################################################



import torch.nn as nn
import torch
from torch.nn import functional as F


class FusionModel(nn.Module):
    def __init__(self, ratio=0.6):
        super(FusionModel, self).__init__()
        c1 = int(64 * ratio)
        c2 = int(128 * ratio)
        c3 = int(256 * ratio)
        c4 = int(512 * ratio)

        self.block1 = Block([c1, c1, 'M'], in_channels=3, first_block=True)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
        self.block5 = Block([c4, c4, c4, c4], in_channels=c4)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(c4, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self._initialize_weights()

    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]

        RGB, T, shared = self.block1(RGB, T, None)
        RGB, T, shared = self.block2(RGB, T, shared)
        RGB, T, shared = self.block3(RGB, T, shared)
        RGB, T, shared = self.block4(RGB, T, shared)
        _, _, shared = self.block5(RGB, T, shared)
        x = shared

        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block(nn.Module):
    def __init__(self, cfg, in_channels, first_block=False, dilation_rate=1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate

        self.rgb_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

        channels = cfg[0]
        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)
        if first_block is False:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)


        #############################################################
        #加入TransformerBlock层
        # self.shared_transformer = TransformerBlock(in_channels=channels)
        #############################################################
        #卷积注意力by豆包
        self.shared_attention = ConvAttentionBlock(in_channels=channels)
        self.rgb_attention = ConvAttentionBlock(in_channels=channels)
        self.t_attention = ConvAttentionBlock(in_channels=channels)




        ##############################################################
        # #卷积注意力by Copilot
        # self.shared_attention = ConvAttention(in_channels=channels, out_channels=channels)




    def forward(self, RGB, T, shared):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)
        if self.first_block:
            shared = torch.zeros(RGB.shape).cuda()
        else:
            shared = self.shared_conv(shared)

        new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)


        ############################################################################
        # # 对new_shared应用TransformerBlock
        # b, c, h, w = new_shared.shape
        # new_shared = new_shared.flatten(2).transpose(1, 2)  # 转换为 (batch_size, seq_len, in_channels)
        # new_shared = self.shared_transformer(new_shared)
        # new_shared = new_shared.transpose(1, 2).view(b, c, h, w)  # 转换回 (batch_size, in_channels, height, width)
        ############################################################################

        #使用卷积注意力模块处理new_shared
        new_RGB = self.rgb_attention(new_RGB)
        new_T = self.t_attention(new_T)
        new_shared = self.shared_attention(new_shared)

        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T, shared):

        RGB_m = self.rgb_msc(RGB)
        T_m = self.t_msc(T)
        if self.first_block:
            shared_m = shared  # zero
        else:
            shared_m = self.shared_fuse_msc(shared)

        rgb_s = self.rgb_fuse_1x1conv(RGB_m - shared_m)
        rgb_fuse_gate = torch.sigmoid(rgb_s)
        t_s = self.t_fuse_1x1conv(T_m - shared_m)
        t_fuse_gate = torch.sigmoid(t_s)
        new_shared = shared + (RGB_m - shared_m) * rgb_fuse_gate + (T_m - shared_m) * t_fuse_gate

        new_shared_m = self.shared_distribute_msc(new_shared)
        s_rgb = self.rgb_distribute_1x1conv(new_shared_m - RGB_m)
        rgb_distribute_gate = torch.sigmoid(s_rgb)
        s_t = self.t_distribute_1x1conv(new_shared_m - T_m)
        t_distribute_gate = torch.sigmoid(s_t)
        new_RGB = RGB + (new_shared_m - RGB_m) * rgb_distribute_gate
        new_T = T + (new_shared_m - T_m) * t_distribute_gate

        return new_RGB, new_T, new_shared


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion



class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=1, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(in_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.self_attn(x, x,x)[0]
        x = self.norm1(x)
        x = x + self.feed_forward(x)
        x = self.norm2(x)
        return x


    # 定义卷积注意力模块
class ConvAttentionBlock(nn.Module):
    def __init__(self, in_channels,dropout_rate=0.1):
        super(ConvAttentionBlock, self).__init__()
        # 平均池化层，用于计算空间信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层，用于学习通道间的关系
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 对输入进行平均池化，得到全局空间信息
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        # 通过全连接层学习通道注意力权重
        y = self.fc(y)
        # 将通道注意力权重应用到输入上
        return x * y.expand_as(x)


# by Copilot
class ConvAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        conv_out = self.conv(x)
        attention_map = self.softmax(conv_out)
        out = x * attention_map
        return out


def fusion_model():
    model = FusionModel()
    return model


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)