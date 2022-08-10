function [x,error] = conjgrad(niter,ATA,b, x,gui_on)
r = b - ATA(x);
%r = b;
p = r;
rsold = r' * r;

for i = 1:length(b)
    
%         asd = reshape(x,[320 368]);
%     figure, imshow(abs(asd),[])
%     

    
    if(gui_on==1)
    progressbar(i/(niter+1))
    end
    Ap = ATA(p);
    alpha = rsold / (p' * Ap);
    x = x + alpha * p;
    r = r - alpha * Ap;
    rsnew = r' * r;
    if sqrt(rsnew) < 1e-25 || i>niter
        %disp(['Ended!'])
        break;
    end
    p = r + (rsnew / rsold) * p;
    error(i) = 1;%norm(b-ATA(x))/norm(b);
    rsold = rsnew;
    %disp(['iteration: ' num2str(i) ' and error: ' num2str(error(i))])
    %%%%%%%%%%%PLOT%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end
if(gui_on==1)
progressbar(1)
end
end