# VD
import sys;
import random;
import subprocess
import os

def main(): 

	print sys.argv[0]
	print sys.argv[1]
	output = ""
	for file in os.listdir(sys.argv[1]):
		#print(os.path.join(sys.argv[1], file))
		file_ = open("res_format.txt", "w")
		if file.endswith(".csv"):

			print(os.path.join(sys.argv[1], file))
			t_file = os.path.join(sys.argv[1], file) 
			command = "python ~/Dropbox/Dr_Hasan/convert_format.py " + t_file
			
			subprocess.Popen(command, stdout= file_,stderr=subprocess.STDOUT, shell=True)
			
if __name__ == '__main__':
	main()
