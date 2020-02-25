import os

os.mkdir('./food-101/images/train')

for folder in os.listdir('./food-101/images'):
	os.mkdir('./food-101/images/train/' + folder)

with open('./food-101/meta/train.txt', 'r') as train_file:
	for line in train_file.readlines():
		os.rename('./food-101/images/' + line[:-1] + '.jpg', './food-101/images/train/' + line[:-1] + '.jpg')

os.mkdir('./food-101/images/test')

for folder in os.listdir('./food-101/images'):
	os.mkdir('./food-101/images/test/' + folder)

with open('./food-101/meta/test.txt', 'r') as test_file:
	for line in test_file.readlines():
		os.rename('./food-101/images/' + line[:-1] + '.jpg', './food-101/images/test/' + line[:-1] + '.jpg')