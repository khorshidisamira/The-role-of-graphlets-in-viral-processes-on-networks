# VD
import sys;
import random;
import subprocess
import os

def main(): 

	print sys.argv[0]
	print sys.argv[1]
	i = 0
	for file in os.listdir(sys.argv[1]):
		#print(os.path.join(sys.argv[1], file))
		if file.endswith("_GUISE.txt"):
			#if i < 5:

			print(os.path.join(sys.argv[1], file))
	
			dataset_file = os.path.join(sys.argv[1], file)
			filename = dataset_file + "result.txt"
			command = "~/Dropbox/Dr_Hasan/Project_graphlet/GUISE-Source-master/GUISE -d " + dataset_file + " -i 20000"
			file_ = open(filename, "w")
			subprocess.Popen(command, stdout= file_,stderr=subprocess.STDOUT, shell=True)
			i = i+1

if __name__ == '__main__':
	main()
