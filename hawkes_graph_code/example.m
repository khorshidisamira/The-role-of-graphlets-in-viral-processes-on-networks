theta=.015;

%[TE_exact]=exact_hawkes(A,100,theta);

%E=A2E(A);
  
formatSpec = '%s%f';
T = readtable('C:\Users\Samira\Dropbox\Dr_Hasan\Facebook\average_degree.txt','Delimiter','\t','Format',formatSpec);
%average_degrees = cell2mat(a_out);

S = table2struct(T);
x = S(1).Var1;
other_directory = 'C:\Users\Samira\Dropbox\facebook';

myFiles = dir(fullfile(other_directory,'*_GUISE.txt')); %gets all wav files in struct
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  name = strsplit(baseFileName, '.');
  index = name{1,1}; 
  thetas = zeros(length(S));
  n = length(S);
  for j = 1:n
      var1 = S(j).Var1;
      var2 = S(j).Var2; 
      new_theta = 2;
      thetas(j) =  strcmp(string(index), string(var1));
      
      if strcmp(string(index), string(var1))
          fprintf(1, 'network %s\n', var1);
          fprintf(1, 'Average degree %4f\n', var2);
          
          new_theta = new_theta / var2;
          break;
      else
          new_theta = theta; 
      end
  end
 
  fullFileName = fullfile(other_directory, baseFileName);
  
  fid = fopen(fullFileName);
  out = textscan(fid,'%d%d','delimiter','\t');
  fclose(fid);

  edges = cell2mat(out);
  %%adjacency_matrix = E2A(edges);
  %%G = digraph(edges(:,1),edges(:,2));
  %%adjacency_matrix = adjacency(G);
  %%fName = strcat(baseFileName,'adj.txt');         
  %%csvwrite(fName, full(adjacency_matrix));
 
  if new_theta > 1
      new_theta = .9999;
  end
  fprintf(1, 'New Theta%4f\n', new_theta);
  [TE_sample]=sample_hawkes(edges,5000,new_theta);
   
  fprintf('simulated total events %.4f\n', TE_sample);
end
