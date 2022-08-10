function [res] = cs_main(kspace,maps,iteration,TV_weight,mu)



sense_maps = maps;

ksb = size(kspace,1);
slices = size(sense_maps,4);

[m,n,no_c] = size(kspace);

non_acq_p = kspace==0;
acq_p = ones(m,n,no_c,'single')-non_acq_p;
non_acq_p = logical(non_acq_p);
loc_mask = logical(acq_p);

y = kspace(loc_mask);
cc =sense_maps(:,:,:,:); %% loading sens maps

E = @(x) sense_op(cc,x,m,n,no_c,loc_mask,ksb,slices);
ET = @(x) sense_op_t(cc,x,loc_mask,m,n,no_c,ksb,slices);

ATA = @(x) ET(E(x)) + mu*x;

inp = reshape(ET(y),[slices*m,n]);
z = TV_denoise(reshape(inp,[m n]), TV_weight);
ATb = ET(y) + mu*z(:);
inpx = z;


for ii = 1:iteration-1
    display(strcat('Iteration: ',num2str(ii), ' out of  ',num2str(iteration)))
    ATb = ET(y) + mu*z(:);
    %inp = reshape(z,[slices*m,n]);

    [imm] = conjgrad(5,ATA, ATb, inpx(:),0);

    z = TV_denoise(reshape(imm,[m n]), TV_weight);
    inpx = z;%reshape(imm,[slices*m,n]);
end

res = z;
end