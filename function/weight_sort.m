function [ W_sort ] = weight_sort( known, W_inc )
%--------------------------------------------------------------------------
% Xue Shengke, Zhejiang University, April 2017.
% Contact information: see readme.txt.
%
% Xue et al. (2017) DW-TNNR paper, IEEE Transactions on Information Theory.
%--------------------------------------------------------------------------
%     compute an sorted weight matrix according to known elements, rows 
%     with more observed elements are given smaller weights
% 
%     Inputs:
%         known              --- index matrix of known elements
%         W_inc              --- increasing weight matrix
% 
%     Outputs: 
%         W_sort             --- sorted weight matrix
%--------------------------------------------------------------------------

N_ori = sum(known, 2);
[N_sort, index] = sort(N_ori, 'descend');
[~, index_back] = sort(index, 'ascend');

inc_weight = diag(W_inc);
sorted_weight = inc_weight(index_back);
W_sort = diag(sorted_weight);

end

