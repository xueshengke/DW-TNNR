function [ W_row, W_col ] = weight_exp(known, theta1, theta2)
%--------------------------------------------------------------------------
% Shengke Xue, Zhejiang University, April 2017. 
% Contact information: see readme.txt.
%
% Xue et al. (2017) DW-TNNR paper
%--------------------------------------------------------------------------
%     compute weight matrices using exponential function, rows with more 
%     observed elements are given smaller weights
% 
%     Inputs:
%         known              --- index matrix of known elements
%         theta1             --- control the weight for rows
%         theta2             --- control the weight for columns
% 
%     Outputs: 
%         W_row              --- generated weight matrix for rows
%         W_col              --- generated weight matrix for columns
%--------------------------------------------------------------------------

[m, n] = size(known);
row_known = sum(known, 2);
col_known = transpose(sum(known, 1));

% use the exponential function to compute the weight, the row (column)
% with more observed elements is offered with a smaller value of weight
% w_r = exp( -theta1 * (row_known / n - 1) ) - 1;
% w_c = exp( -theta2 * (col_known / m - 1) ) - 1;

w_r = - row_known / n + 2;
w_c = - col_known / m + 2;

w_r = theta1 * w_r / max(w_r);
w_c = theta2 * w_c / max(w_c);

if theta1 == 0
    w_r = 1.0 * ones(size(w_r));
end
if theta2 == 0
    w_c = 1.0 * ones(size(w_c));
end

W_row = diag(w_r);
W_col = diag(w_c);

% visualize the weight in an image
X = ones(m, n);
Y = W_row * X * W_col;
Y = Y .* (ones(m,n) - known);
figure; imagesc(Y); colorbar;

end
