This code includes the detailed implementation of the paper:

Reference:
Xue, S., Qiu, W., Liu, F., et al., (2017). Double Weighted Truncated Nuclear 
Norm Regularization for Efficient Matrix Completion. IEEE Transactions on 
Information Theory, submitted.

It is partially composed of TNNR code implementation. We would like to thank 
Dr. Debing Zhang for sharing his code.

Reference:
Hu, Y., Zhang, D., Ye, J., Li, X., & He, X. (2013). Fast and accurate matrix 
completion via truncated nuclear norm regularization. IEEE Transactions on 
Pattern Analysis and Machine Intelligence, 35(9), 2117-2130.

The code contains:
|--------------
|-- DW_TNNR_main.m            entrance to start the experiment
|-- function/                 functions of DW-TNNR algorithm
    |-- DW_TNNR_algorithm.m   main part of DW-TNNR implementation
    |-- PSNR.m                compute the PSNR and Erec for recovered image
    |-- weight_exp.m          compute weight matrix using exponential function	
    |-- weight_matrix.m       compute weight matrix in an increasing order
    |-- weight_sort.m         sort the sequence of weight value according to
                                  observed elements; rows with more observed 
                                  elements are given smaller weights
|-- image/                    directory for original images
|-- mask/                     directory for various mask types, 300x300
|-- result/                   directory for saving experimental results
|-------------

For algorithm interpretation, please read our Xue et al. (2017) paper, in 
which more details are demonstrated.

If you have any questions about this implementation, please do not hesitate 
to contact me.

Shengke Xue, 
College of Information Science and Electronic Engineering,
Zhejiang University, Hangzhou, P. R. China,
e-mail: (either one is o.k.)
xueshengke@zju.edu.cn, xueshengke1993@gmail.com.