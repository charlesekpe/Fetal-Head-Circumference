# Fetal-Head-Circumference
_Automated measurement of fetal head circumference_

![alt text](./images/401db3_826b0e5521f5446c8e5307c8ca949fcd_mv2.png?raw=true)

During pregnancy, ultrasound imaging is used to measure fetal biometrics. One of these measurements is the **fetal head circumference (HC)**.

The HC can be used to estimate the gestational age and monitor growth of the fetus. The HC is measured in a specific cross section of the fetal head, which is called the standard plane.

The dataset for this model contains a total of 999 two-dimensional (2D) ultrasound images of the standard plane that can be used to measure the HC.

We will train a model to predict the HC of any ultrasound image.

The size of each 2D ultrasound image is 800 by 540 pixels with a pixel size ranging from 0.052 to 0.326 mm.
One challenge of this task, is the low pixel size, and hence treating this as a regression task will have alot of shortcomings.

We will attempt a **PyTorch** implementation of a **U-Net** model for the segmentation of ultrasound images in order to estimate the HC.

The approach employed to solve the problem was to **segment** the **fetal head** followed by using **Hough transform** to fit the ellipse. The **Unet** model was trained to segment the ultrasound images and ellipse fitting was performed using the **OpenCV** implementation of **Hough transforms**.
