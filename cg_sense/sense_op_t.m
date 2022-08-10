function [new_kspace] = sense_op_t(coils_ind,x,loc_mask,m,n,no_c,ksb,slices);

%% put the appr. locations of the conc. kspace with the incoming data
kk1 = zeros(m,n,no_c,'single');
kk1(loc_mask) = x;

%% go back to image domain

kspace_to_im = @(x) ifft2c(x);%* sqrt(size(x,1) * size(x,2));
% kk2 = kspace_to_im(kk1);

for abc = 1:no_c
    kk2(:,:,abc) = kspace_to_im(kk1(:,:,abc));
end


for klm = 1:slices
    
    %% conjugate sens multiplication
    for abc = 1:no_c
        ev1(:,:,abc) = conj(squeeze(coils_ind(:,:,abc,klm))).*kk2(:,:,abc);
    end
    
    ev2((klm-1)*ksb + 1:klm*ksb,:) = sum(ev1,3);
    
end
new_kspace = ev2(:);
end




