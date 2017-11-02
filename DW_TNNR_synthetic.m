% Shengke Xue, Zhejiang University, April 2017. 
% Contact information: see readme.txt.
%
% Reference: 
% S. Xue, W. Qiu, F. Liu, et~al., Double Weighted Truncated Nuclear 
% Norm Regularization for Low-Rank Matrix Completion, , submitted, 2017.

% Partially composed of Hu et al. (2013) TNNR implementation, written by 
% Dr. Debing Zhang, Zhejiang Universiy, November 2012.

%% add path
close all; clear ; clc;
addpath image ;
addpath mask ;
addpath function;

%% read image files directory information
result_dir = './result/synthetic';
if ~exist(result_dir, 'dir'),   mkdir(result_dir);    end

%% parameter configuration
para.lost = 0.50;        % ratio of lost elements in matrix
para.save_eps = 0;       % save eps figure in result directory
para.min_R = 1;          % minimum rank of chosen image
para.max_R = 20;         % maximum rank of chosen image
% it requires to test all ranks from min_R to max_R, note that different
% images have different ranks, and various masks affect the ranks, too.

para.max_iter = 200;     % maximum number of iteration
para.epsilon = 1e-4;     % tolerance

para.alpha = 2e-4;       % 1/apha, positive step size of gradient descent
para.rho   = 1.20;       % rho > 1, scale up the value of alpha
para.theta1 = 1.00;      % compute the weight matrix, theta1 = theta2
para.theta2 = 1.00;      % can obtain the best PSNR
para.eta = 0.00;         % for best robustness
para.progress = 0;       % show the recovered image in each iteration

%% generate synthetic data for experiment
image_name = 'synthetic_data';
m = 200;
n = 200;
dim = 3;
r = 10;
sigma = 0.3;    % [0.1, 0.9]

%% random loss
rnd_idx = randi([0, 100-1], m, n);
old_idx = rnd_idx;
lost = para.lost * 100;
fprintf('loss: %d%% elements are missing.\n', lost);
rnd_idx = double(old_idx < (100-lost));
mask = repmat(rnd_idx, [1 1 dim]); % index matrix of the known elements

L = randn(m, r);
R = randn(n, r);
noise = sigma * randn(m, n);
M = L * R' + noise * mask(:,:,1);
M = mapminmax(M, 0, 255);
X_full = repmat(M, [1 1 dim]);

%% Double Weighted Truncated Nuclear Norm Regularization
fprintf(['Double Weighted Truncated Nuclear Norm Regularization ' ...
         'for Matrix Completion\n']);
t1 = tic;
[tnnr_res, X_rec]= DW_TNNR_algorithm(result_dir, image_name, X_full, mask, para);
toc(t1);

tnnr_rank = tnnr_res.best_rank;
tnnr_psnr = tnnr_res.best_psnr;
tnnr_erec = tnnr_res.best_erec ./ 255;
tnnr_time_cost = tnnr_res.time(tnnr_rank);
tnnr_iteration = tnnr_res.iterations(tnnr_rank, :);

fprintf('\nrank=%d, psnr=%f, erec=%f, time=%f s, iteration=(%d,%d,%d)\n', ...
    tnnr_rank, tnnr_psnr, tnnr_erec, tnnr_time_cost, tnnr_iteration(1), ...
    tnnr_iteration(2), tnnr_iteration(3));
disp(' ');

figure;
subplot(2, 2, 1);
plot(tnnr_res.Rank, tnnr_res.Psnr, 'o-');
xlabel('Rank');
ylabel('PSNR');

subplot(2, 2, 2);
plot(tnnr_res.Rank, tnnr_res.Erec ./ 255, 'diamond-');
xlabel('Rank');
ylabel('Recovery error');

subplot(2, 2, 3);
plot(tnnr_res.Psnr_iter, 'square-');
xlabel('Iteration');
ylabel('PSNR');

subplot(2, 2, 4);
plot(tnnr_res.Erec_iter ./ 255, '^-');
xlabel('Iteration');
ylabel('Recovery error');

%% show the recovered image in each iteration
if para.progress
    figure('NumberTitle', 'off', 'Name', 'TNNR-WRE progress');
    num_iter = min(tnnr_iteration);
    X_rec = X_rec / 255;
    for i = 1 : num_iter
        imshow(X_rec(:, :, :, i));
        title(['iter ' num2str(i)]);
    end    % better set a breakpoint here, to display image step by step
end

%% record test results
outputFileName = fullfile(result_dir, 'parameters.txt'); 
fid = fopen(outputFileName, 'a') ;
fprintf(fid, '****** %s ******\n', datestr(now,0));
fprintf(fid, '%s\n', ['image: '           image_name               ]);
fprintf(fid, '%s\n', ['loss ratio: '      num2str(para.lost)       ]);
fprintf(fid, '%s\n', ['min rank: '        num2str(para.min_R)      ]);
fprintf(fid, '%s\n', ['max rank: '        num2str(para.max_R)      ]);
fprintf(fid, '%s\n', ['max iteration: '   num2str(para.max_iter)   ]);
fprintf(fid, '%s\n', ['tolerance: '       num2str(para.epsilon)    ]);
fprintf(fid, '%s\n', ['alpha: '           num2str(para.alpha)      ]);
fprintf(fid, '%s\n', ['rho: '             num2str(para.rho)        ]);
fprintf(fid, '%s\n', ['theta1: '          num2str(para.theta1)     ]);
fprintf(fid, '%s\n', ['theta2: '          num2str(para.theta2)     ]);
fprintf(fid, '%s\n', ['eta: '             num2str(para.eta)        ]);
fprintf(fid, '%s\n', ['sigma: '           num2str(sigma)           ]);

fprintf(fid, '%s\n', ['rank: '            num2str(tnnr_rank)       ]);
fprintf(fid, '%s\n', ['psnr: '            num2str(tnnr_psnr)       ]);
fprintf(fid, '%s\n', ['recovery error: '  num2str(tnnr_erec)       ]);
fprintf(fid, '%s\n', ['time cost: '       num2str(tnnr_time_cost)  ]);
fprintf(fid, 'iteration: %d, %d, %d\n',   tnnr_iteration(1), ...
    tnnr_iteration(2), tnnr_iteration(3));

fprintf(fid, '--------------------\n');
fclose(fid);
