function [TE]=exact_hawkes(A,max_gen,theta)
% A is adjacency matrix
% max_gen is the cutoff for maximum branching generations
% theta is the probability an event at a node triggers an event at neighbor
% output TE is the total expected number of events 

M=zeros(size(A));
for i=1:max_gen
     M=M+(theta*A)^(i-1);
end 

N=size(A,1);
TE=sum(M*ones(N,1)/N);

end