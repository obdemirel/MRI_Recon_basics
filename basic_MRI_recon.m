clear all
clc
addpath utils
%% first let's load the data 
%% k-space and sensitivity maps are in the example .mat file
%% Sensitivity maps are generated using ESPIRiT
%% Following paper is used to generate these maps
%% Uecker, Martin, et al. "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA." Magnetic resonance in medicine 71.3 (2014): 990-1001.



load dental_example.mat

%% let's visualize the data
figure, imshow(log(kspace(:,:,1)),[]), title('Fully sampled k-space Channel 1') %% 1st channel k-space in log domain
figure, 
for ii = 1:size(maps,3)
    subplot(2,5,ii), imshow(abs(squeeze(maps(:,:,ii))),[]), title(strcat('Map of Chnanel: ',num2str(ii))) % shows each channel sensitivity profile
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%since the data is fully sampled let's generate the SENSE-1 image as the
%reference image
reference = sum(ifft2c(kspace).*conj(maps),3);
figure, imshow(abs(reference),[]), title('Fully Sampled Reference Image')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%now let's sample it by R=4 and keep the center 24 lines as ACS
PE_R = 2;
kspace_r4 = kspace*0;
kspace_r4(:,1:PE_R:end,:) = kspace(:,1:PE_R:end,:); % R4 sampling
kspace_r4(:,86-12:86+12-1,:) = kspace(:,86-12:86+12-1,:); % keeping 24 ACS

figure, imshow(log(kspace_r4(:,:,1)),[]), title('Undersampled k-space Channel 1') 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Iterative SENSE reconstruction using CG-SENSE which is also a basis of the CS recon
%% Pruessmann, Klaas P., et al. "Advances in sensitivity encoding with arbitrary k‐space trajectories." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 46.4 (2001): 638-651.
cd cg_sense
cgsense_result = cgsense_main(kspace_r4,maps,10,0); % 10 iterations
cd ..
figure, imshow(abs(cgsense_result),[]), title('CG-SENSE Reconstructed Image')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% GRAPPA reconstruction
%% Griswold, Mark A., et al. "Generalized autocalibrating partially parallel acquisitions (GRAPPA)." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 47.6 (2002): 1202-1210.
ims =1;
gui_on = 0; %make it 1 if you want to see a progress window
par_on = 0; %make it 1 if you have multi-cpu support
kernels = [5,4]; % GRAPPA kernel size
grappa_acs = kspace_r4(:,86-12:86+12-1,:); %% 24 ACS lines for GRAPPA kernel calibration
%% GRAPPA codes are not tested for R>4!!!!

cd GRAPPA
tic
[hf_kspace] = grappa_main(1,PE_R,ims,kspace_r4,grappa_acs,kspace_r4,kernels,gui_on,par_on);
toc
grappa_result = sum(ifft2c(squeeze(hf_kspace)).*conj(maps),3);
figure, imshow(abs(grappa_result),[]), title('GRAPPA Reconstructed Image')
cd ..
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Compressed Sensing reconstruction
%% Lustig, Michael, David Donoho, and John M. Pauly. "Sparse MRI: The application of compressed sensing for rapid MR imaging." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 58.6 (2007): 1182-1195.
%% Yang, Junfeng, Yin Zhang, and Wotao Yin. "A fast alternating direction method for TVL1-L2 signal reconstruction from partial Fourier data." IEEE Journal of Selected Topics in Signal Processing 4.2 (2010): 288-297.
%% Still developing, please wait until this line vabishes!
cd compressed_sensing

[cs_result] = cs_main(kspace_r4,maps,10,5e-5,0.2);
figure, imshow(abs(cs_result),[]), title('Compressed Sensing Reconstructed Image')
cd ..
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% plotting all

figure, imshow([flipud(abs(reference).') flipud(abs(cgsense_result).') flipud(abs(grappa_result).') flipud(abs(cs_result).')],[0 .8]), title('Reference || CG-SENSE || GRAPPA || CS')
figure, imshow([abs(flipud(abs(reference).') - flipud(abs(reference).'))...
                abs(flipud(abs(reference).') - flipud(abs(cgsense_result).'))....
                abs(flipud(abs(reference).') - flipud(abs(grappa_result).'))....
                abs(flipud(abs(reference).') - flipud(abs(cs_result).'))],[0 .2]), title('Difference to Reference: Reference || CG-SENSE || GRAPPA || CS')


%% calculating the PSNR and SSIM metric to see how good are the reconstructions
for ii = 1:3
    
    if(ii==1)
        inp_im = cgsense_result;
    elseif(ii==2)
        inp_im = grappa_result;
    elseif(ii==3)
        inp_im = cs_result;
    end

    inp_im = abs(inp_im./max(abs(inp_im(:))));
    ref_im = abs(reference./max(abs(reference(:))));

    psnr_val(ii) = psnr(inp_im,ref_im);
    ssim_val(ii) = ssim(inp_im,ref_im);

end

disp('PSNR values (dB): CG-SENSE || GRAPPA || CS')
disp(psnr_val)
disp('SSIM values (%): CG-SENSE || GRAPPA || CS')
disp(ssim_val*100)
