import os
import glob
import sys


def get_est_type():
	jsonfile = ""
	for file in os.listdir('.'):
		if file.endswith(".json"):
			jsonfile = str(file)
	
	est_type = ''
	for c in jsonfile:
		if(c.isupper() == True):
			est_type += c
	

	return est_type


def get_json_file():
	jsonfile = ""
	for file in os.listdir('.'):
		if file.endswith(".json"):
			jsonfile = str(file)
	
	return jsonfile
