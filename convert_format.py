# VD
import sys;
import random;
def read_write(file_name):
	filetoread = open(file_name);
	file_name1 = file_name;
	file_name1 += "_GUISE.txt";
	file_wr = open(file_name1,'w');
	index = 0;
	for line in filetoread:
		#if index == 0:				# ignore 1st line, headings
		#	index += 1;
		#	continue;

		line = line.strip().split(",");
		file_wr.write(line[0]+"\t"+line[1]);
		file_wr.write("\n");
		index += 1;

def main():
	read_write(sys.argv[1]);

if __name__ == '__main__':
	main()
