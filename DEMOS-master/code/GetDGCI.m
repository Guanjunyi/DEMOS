% Please kindly cite the paper Junyi Guan, Sheng li, Xiaojun Chen, Xiongxiong He, and Jiajia Chen 
% "DEMOS: clustering by pruning a density-boosting cluster tree of density mounts" 
% IEEE Transactions on Knowledge and Data Engineering,2023

% The code was written by Junyi Guan in 2022.

function [DGCI]=GetDGCI(rho,delta,re_cts,n)
%DGCI (an F1-score-based Decision Graph Clarity Index) is designed to
%quantify the clarity of a decision graph.

rho=(rho-min(rho))./(max(rho)-min(rho));
rho(isnan(rho))=0;
delta=(delta-min(delta))./(max(delta)-min(delta));
delta(isnan(delta))=0;

gap = 0.1; %% gap is adjustable (the smaller the finer)
rho_inf=0:gap:1; delta_inf=0:gap:1;
t_r=length(rho_inf);
t_d=length(delta_inf);

figure('Position',[850 500 140 140]);
colormap autumn
cmap = colormap;

F1_img=ones(t_r,t_d,3);
F1_img(:,:,1) = cmap(1,1);
F1_img(:,:,2) = cmap(1,2);
F1_img(:,:,3) = cmap(1,3);
F1_inf = [];

for i=1:t_r
    for j=1:t_d
        thr_rho = rho_inf(j);
        thr_delta = delta_inf(i);
        aaa = find(rho>thr_rho);
        bbb = find(delta>thr_delta);
        centers = intersect(aaa,bbb);
        if length(centers)~=1
            [~,~,F1] = PRE_REC_F1(re_cts,centers,n);
        else
            F1 = 0;
        end
        F1_img(i,j,:)=cmap(max(1,round(64*F1)),:);
        F1_inf = [F1_inf;F1];
    end
end

F1_img = imresize(F1_img,[400,400]);
F1_img_0 = padarray(F1_img,[60 60],0.8,'both');
DGCI = mean(F1_inf(F1_inf>0));
imshow(flipud(F1_img_0));

colormap autumn
colorbar('position',[0.92 0.10 0.04 0.4],'FontWeight','bold');
text(480,240,'F1','FontWeight','bold');
end