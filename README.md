
# Capsule Net in MxNetR

This is a simple example for implementing the Capsule Net by MxNetR. However, I have reduce the number of filter of initial convolutional layer (256 -> 64) and number of capsule channel (32 -> 8). The number of deconvolution filter are also reduced (512, 1024 -> 64, 128)
If you want to get the original Capsule Net reporting in paper, please revise the function parameters.

This simplified Capsule Net only includes 338,368 parameters and achieve 98.75% accuracy in testing set.

