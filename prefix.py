import os
path = 'pdfs/'
for file in os.listdir(path):
	if not file.startswith('kpmg_'):
		os.rename(path + file, path + "pwc_" + file)