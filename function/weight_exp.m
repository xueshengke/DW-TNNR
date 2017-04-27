function [ W_row, W_col ] = weight_exp(mask, theta)
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
%         theta              --- increasing weight matrix
% 
%     Outputs: 
%         W_sort             --- sorted weight matrix
%--------------------------------------------------------------------------

[m, n] = size(mask);
row_known = sum(mask, 2);
col_known = transpose(sum(mask, 1));

w_r = exp( -theta * (row_known / m - 1) );
w_c = exp( -theta * (col_known / n - 1) );

% w_r(:) = ones(size(w_r));
% w_c(:) = ones(size(w_c));

W_row = diag(w_r);
W_col = diag(w_c);

% row_inc = exp( theta * (1:m)' / m ) - 1;
% col_inc = exp( theta * (1:n)' / n ) - 1;
% [row_sort, idx_r] = sort(row_known, 'descend');
% [col_sort, idx_c] = sort(col_known, 'descend');
% [ ~ , idx_r_back] = sort(idx_r, 'ascend');
% [ ~ , idx_c_back] = sort(idx_c, 'ascend');
% w_r = row_inc(idx_r_back);
% w_c = col_inc(idx_c_back);
% W_row = diag(w_r);
% W_col = diag(w_c);

X = ones(m, n);
Y = W_row * X * W_col;
figure; imagesc(Y); colorbar;

end
