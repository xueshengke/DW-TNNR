function [ W ] = weight_matrix( m, L, theta )
%--------------------------------------------------------------------------
% Shengke Xue, Zhejiang University, April 2017. 
% Contact information: see readme.txt.
%
% Xue et al. (2017) DW-TNNR paper
%--------------------------------------------------------------------------
%     compute an inceasing weight matrix
% 
%     Inputs:
%         m              --- row number of image
%         L              --- a threshold, 1 <= L <= m
%         theta          --- a threshold, controls the increasing ratio
% 
%     Outputs: 
%         W              --- weight matrix
%--------------------------------------------------------------------------

Wv = ones(m, 1);
Wv(L+1 : m) = Wv(L+1 : m) + (theta-1)/(L-1) .* (1:(m-L))';
W = diag(Wv);

end