from os import listdir
from os.path import isfile, join
import os
import glob
import textract
from tqdm import tqdm
import docx2txt
import tika
from tika import parser


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def read_files_tika(file_list, max_read = -1):
	count_read_file = 0
	resume_string_list = []
	valid_file_list = []
	valid_extension = ('.pdf', '.docx')
	for file in tqdm(file_list, desc="Read File: Tika"):
		file_extension = os.path.splitext(file)[1]
		# print(file_extension)
		if file_extension in valid_extension:
			parsed = parser.from_file(file)
			textString = parsed["content"]
			resume_string_list.append(textString)
			valid_file_list.append(file)
			count_read_file = count_read_file + 1
			if count_read_file == max_read:
				break
	# print(resume_string_list)
	return valid_file_list, resume_string_list

def read_files(file_list, max_read = -1):
	count_read_file = 0
	resume_string_list = []
	valid_file_list = []
	valid_extension = ('.pdf', '.docx')
	for file in tqdm(file_list, desc="Read File: Textract"):
		file_extension = os.path.splitext(file)[1]
		# print(file_extension)
		if file_extension in valid_extension:
			text = textract.process(file)
			textString = text.decode('ascii', errors='ignore').encode('ascii')
			resume_string_list.append(textString)
			valid_file_list.append(file)
			count_read_file = count_read_file + 1
			if count_read_file == max_read:
				break
	# print(resume_string_list)
	return valid_file_list, resume_string_list

