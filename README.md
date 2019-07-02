# TGS-Salt-Identification-Challenge
a tensorflow implementation of a CNN model to predict the correct mask for seismic images referring to the presence or the absence of a salt layer

the network architecture is designed as follows:

-images' original size = (101, 101)

-images' are resized to 192 and padded to 224

-Network Encoder : Resnet34 (without the average pooling layer) + scSE

-Network Decoder : Upsampling block of UNet + scSE 
