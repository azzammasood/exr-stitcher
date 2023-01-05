# exrstitcher
Tool for stitching exrs. Features include:

● The application has a tiny memory footprint. This is achieved by
writing out the output image one scan line at a time. So as long as the
filesystem supports it, it should be able to handle any size images or grids.

● It supports both single channel and multi channel EXR images, of
course all the input images need to have the same channel configuration.

● Depending on the input image names, it is able to generate a grid of
any size e.g. 2 x 2 or 4 x 1 or 6 x 8 etc.

● Different rows can have varying heights and different columns can have
varying widths, of course all the images for the same row need to have the
same height and all the images for the same column need to have the same
width. This is adequately illustrated by the images in the test folder.

● It has a validation step to identify any missing files or incorrect/corrupt
input images before the processing actually starts.

