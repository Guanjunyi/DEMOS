% Please kindly cite the paper Junyi Guan, Sheng li, Xiaojun Chen, Xiongxiong He, and Jiajia Chen 
% "DEMOS: clustering by pruning a density-boosting cluster tree of density mounts" 
% IEEE Transactions on Knowledge and Data Engineering,2023

% The code was written by Junyi Guan in 2022.
function [CL,rho,delta,centers,runtime] = DEMOS(data,k_input) 
close all;
tic;
fprintf('DEMOS Clustering :)!\n');
%% data preprocessing
data=(data-min(data))./(max(data)-min(data)); % Normalization
data(isnan(data))=0;
[n,d]  = size(data);

%% parameters

if nargin>1
    k = k_input;
else
    k = ceil(sqrt(n));
end

k_v = min(k,2*floor(log(n))); 

%%%default parameters
cv_thr = 0.4; % smoothness threshold
eta = 0.25; % ratio parameter for obatin n_s
lambda = 2; % amplification factor

%% Fast search of KNN matrix based on kd-tree (when dimension is not large than 10)
if d<=10
    [knn,knn_dist] = knnsearch(data,data,'k',k);
else
    dist = pdist2(data,data,'euclidean');
    [knn_dist,knn] = sort(dist,2);
end

%% KNN Density Evaluation method with smooth control
m = 0; %selt-truning paramter
for i = 1:200
    m = 0.05+m; 
    rho=sum(knn_dist(:,2:k).^m,2).^-1; %% KNN-based density estimation
    cv = std(rho)/mean(rho); %% coefficient of variation
    if  cv> cv_thr
        break
    end
end

%% density peak detection and peak-representiveness calculation

theta = ones(n,1); %%% initialize peak-representiveness
n_descendants = zeros(n,1); %%% record the number of descendants 

[~,OrdRho]=sort(rho,'descend');  
for i=1:n
    for j=2:k
        neigh=knn(OrdRho(i),j);
        if(rho(OrdRho(i))<rho(neigh))
            nneigh(OrdRho(i))=neigh;
            theta(OrdRho(i)) = theta(neigh)* (rho(OrdRho(i))/ rho(neigh)); % representiveness learning
            n_descendants(neigh) = n_descendants(neigh)+1;
            break
        end
    end
end

pks = find(theta==1);%% find density peaks
n_p = length(pks);%% the number of density peaks
rho_p = rho(pks);%% density of density peaks

%%% generate density mounts(mt)

mt_l=-1*ones(n,1); %% initialize mt labels for all points.
mt_l(pks) = (1:n_p); %% give unique mt labels to density peaks.

for i=1:n
    if (mt_l(OrdRho(i))==-1)
        mt_l(OrdRho(i))=mt_l(nneigh(OrdRho(i)));%% inherit sub-labels from NPN
    end
end

%% find the number of edges
for i = 1:n_p
    n_descendants_mt = n_descendants(mt_l==i);
    n_edges(i) = length(find(n_descendants_mt==0)); %% number of edges of a density mount mt
end

%% The identification of cross-mount valley point pairs
pair = []; %cross-mount valley point pair
for i=1:n
    label_i = mt_l(i);
    for j = 2:k_v
        i_neigh = knn(i,j);
        dist_i_to_neig = knn_dist(i,j);
        label_neig = mt_l(i_neigh);
        if label_i ~= label_neig & find(knn(i_neigh,2:k_v)==i)
            pair = [pair;[i i_neigh dist_i_to_neig]];
            break
        end
    end
end

%% The identification of valley link "vlink"
if isempty(pair)
    CL = mt_l;
    vlink = [];
    C = n_p;
else
    pair(:,1:2) = sort(pair(:,1:2),2);
    [~,index] = unique(pair(:,3));
    pair = pair(index,:);
    pair = sortrows(pair,3);
    n_pair = size(pair,1);
    
    %% The identification of links from border point pairs
    vlink = [];
    
    for i = 1:n_pair
        pair_1 = pair(i,1:2);
        if isempty(intersect(pair_1,vlink))
            vlink = [vlink;pair_1];
        end
    end 
    %% The collection of connectivity messages
    n_vlink = size(vlink,1);
    message_matrix = cell(n_p,n_p);
    for i = 1:n_vlink
        ii = vlink(i,1);
        jj = vlink(i,2);
        p1 = mt_l(ii);
        p2 = mt_l(jj);
        message = message_matrix(p1,p2);
        msg = theta(ii)*theta(jj);
        message{1} = [message{1};msg];
        message_matrix(p1,p2) = message;
        message_matrix(p2,p1) = message;
    end
    
    %% the connectivity estimaiton of density peaks
    conectivity = zeros(n_p,n_p);
    for p1=1:n_p-1
        for p2 =p1+1:n_p  
            message = message_matrix(p1,p2);
            message = message{:};
            max_msg = max(message);
            n_edges_p1 = n_edges(p1);
            n_edges_p2 = n_edges(p2);
            n_s = ceil(min(n_edges_p1,n_edges_p2)*eta);
            s_vector = [message;zeros(n_s,1)];
            s_vector = sort(s_vector,'descend'); %% s_vector: connectivity message vector
            message = s_vector(1:n_s);
            if max_msg>0
                w_vector = 1/(n_s*(n_s+1)/2)*(1:n_s)'; %% w_vector: an equally decreasing weight vector for 's_vector'
                message = sort(message);
                con = sum(message.* w_vector);%% con: a conectivity value
                conectivity(p1,p2) = con;
                conectivity(p2,p1) = con;
            end
        end
    end
    
    %% Peak-Graph building
    G=sparse(n_p,n_p);
    for i=1:n_p
        for j = 1:n_p
            if conectivity(i,j)~=0 & conectivity(j,i)~=0
                G(i,j) = 1-conectivity(i,j);
                G(j,i) = 1-conectivity(j,i);
            end
        end
    end
    [~,aOrder_ldps]=sort(rho_p,'ascend'); 
    max_con = zeros(n_p,1); % max_con_p: max connectivity from desity mount 'mt(p)' with others higher density areas
    cur_cl = (1:n_p); % cur_mt_l: current cluster label during desity mount merging
    nneigh_p = (1:n_p); % nneigh_p: hihger density peaks with strongest connection 
    region_max_rho = rho_p; % region_max_rho: max density of the mountain that a density peak is located
    
    for i = 1:n_p-1
        low_p = aOrder_ldps(i); % low_p: a density peak with a low density
        for j = i+1:n_p
            high_p = aOrder_ldps(j); % high_p: a density peak a with high density      
            [~,path,~] = graphshortestpath(G,low_p,high_p); % path: a path from 'low_p' to 'high_p'
            if length(unique(cur_cl(path)))==2  % make sure that 'path' is within the clusters of 'low_p' and 'high_p'
                vlink_path = [path(1:end-1)' path(2:end)']; %vlink_path: vlinks along 'path'  
                max_member_con = max(diag(conectivity(vlink_path(:,1),vlink_path(:,2)))); % max_member_con: max member connectivity on 'path' 
                if max_con(low_p) < max_member_con
                    max_con(low_p) = max_member_con;
                    nneigh_p(low_p) = high_p;
                end
            end
            if region_max_rho(low_p) < max(rho_p(path))
                region_max_rho(low_p) = max(rho_p(path));
            end
        end
        cur_cl(cur_cl==low_p) = nneigh_p(low_p); %change current cluster label after merging
        conectivity(cur_cl==nneigh_p(low_p),cur_cl==nneigh_p(low_p)) = 0; %% set conectivity value as 0 within each new cluster
    end
    %% obtain delta value of density peaks
    delta_p = 1-max_con;
    
    %% Clarity-enhancing 
    delta_p = delta_p.^lambda;
    rho_p = rho_p./region_max_rho;
    
    time_1 = toc;
    
    rho_p(rho_p==1) = 1:0.015:1+(length(find(rho_p==1))-1)*0.015; %% visual enhancement of cluster centers in the decision graph
    %% draw decision grpah
    fprintf('...please select the appropriate centers in the decision graph!\n');
    figure; 
    plot(rho_p(:), delta_p(:),'o','MarkerSize',4,'MarkerFaceColor','k','MarkerEdgeColor','k')
    hold on
    grid on;
    axis([0 max(rho_p) 0 max(delta_p)]);
    title ('Decision Graph','FontSize',15.0);
    xlabel ('\rho');
    ylabel ('\delta');
    rect = getrect;
    rhomin = rect(1);
    deltamin=rect(2);
    %% center confirm
    tic;
    NC=0;
    for i=1:n_p
        CL_p(i)=-1;
    end
    
    for i=1:n_p
        if rho_p(i)>rhomin & delta_p(i) > deltamin
            NC=NC+1;
            CL_p(i)=NC;
            center_p(NC)=i;
        end
    end
    
    %% allocation
    [~,dOrder_ldps]=sort(rho_p,'descend');
    for i=1:n_p
        if (CL_p(dOrder_ldps(i))==-1)
            CL_p(dOrder_ldps(i))=CL_p(nneigh_p(dOrder_ldps(i)));
        end
    end
    for i=1:n_p
        CL(mt_l==i) = CL_p(i);
    end
    time_2 = toc;
end
runtime = time_1 + time_2;

rho = zeros(n,1);
rho(pks) = rho_p;
delta = zeros(n,1);
delta(pks) = delta_p;

%% real centers
centers = pks(center_p);
fprintf('Finished !!!!\n');








