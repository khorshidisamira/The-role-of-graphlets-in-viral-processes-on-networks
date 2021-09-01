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
		
		if file.endswith("result.txt"):

			print(os.path.join(sys.argv[1], file))
			t_file = os.path.join(sys.argv[1], file) 
			
			with open(os.path.join(sys.argv[1], file)) as input_data:
			    # Skips text before the beginning of the interesting block:
			    	for line in input_data:
					if line.startswith('GFD'): 
						output += t_file 
						output += line
						output += "\n"
						print line  

	print "######################################"
	print output
	filename = "GFD_result.txt"
	file_ = open(filename, "w")
	file_.write(output)  # python will convert \n to os.linesep
	file_.close()


if __name__ == '__main__':
	main()

