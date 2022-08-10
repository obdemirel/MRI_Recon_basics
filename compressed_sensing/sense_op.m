function [new_kspace] = sense_op(coils_ind,mb_kspace,m,n,coil_n,acq_p,ksb,slices)
kk1 = reshape(mb_kspace,slices*m,n);
ev3 = 0;
for abc = 1:slices
%% sens multiplication
ev1 = coils_ind(:,:,:,abc).*repmat(kk1((abc-1)*ksb + 1:abc*ksb,:),[1 1 coil_n]);

%% go back to k-space
im_to_kspace = @(x) fft2c(x);% / sqrt(size(x,1) * size(x,2));

   for abc2 = 1:coil_n
        ev2(:,:,abc2) = im_to_kspace(ev1(:,:,abc2));
    end

%% taking the acq points from k-space
ev3 = ev3 + ev2(acq_p);

end
new_kspace = ev3;
end

