function [E]=A2E(A)
E=[];
N=max(size(A));
for i=1:N
    for j=1:N
        if(i~=j)
            if(A(i,j)==1)
                E=[E; [i j];];
            end
        end
    end
end

end