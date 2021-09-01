function [TE]=sample_hawkes(E,Nsamples,theta)
% E is Nx2 array that contains the index of the edges.  Should be symmetric
% where if [i j] is a row then [j i] is also a row (and now duplicate rows)
%
% Nsamples is number of times to simulate branching process
% theta is the probability an event at a node triggers an event at neighbor
% output TE is the total expected number of events 

N=length(unique(E(:,1)));
TE=ones(Nsamples,1);

for k=1:Nsamples
   i=randsample(1:N,1);
   node_list=[i];
   while(length(node_list)>0)
       [l,TE(k)]=propagate(node_list(1),theta,E,TE(k));
       node_list=node_list(2:end);
       node_list=[node_list; l;];
   end
end
TE=mean(TE);
end

function [l,TE]=propagate(i,theta,E,TE)
l=E(E(:,1)==i,2);
N=length(l);
if(N>0)
   v=rand(N,1)<theta; 
   l=l(v);
end
TE=TE+length(l);
end