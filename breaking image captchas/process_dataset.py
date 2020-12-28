# process_dataset.py
#
# python3 process_dataset.py
# No argparse, just manually edit which captcha directory you want to process
#
# This will take a dataset of plain captchas
# For one label at a time, such as crosswalks
# And extract all of the thumbnails and save them as individual images
# Then you can manually put them into the right directories
# Like, one directory for crosswalks, and another for not_crosswalks
# And then you can train a model with the two classes


from extract_captcha import Extract as ec
from imutils import paths
import cv2


if __name__ == "__main__":
	label = "traffic_lights"
	dataset = f"dataset/{label}/original2"
	save_dir = f"dataset/{label}/processed"
	img_paths = sorted(list(paths.list_images(dataset)))
	itr = 0
	for img_path in img_paths:
		print(img_path)
		thumbs, boxs = ec.extract_thumbs(img_path)
		for thumb in thumbs:
			print(f"{save_dir}/{itr}.png")
			cv2.imwrite(f"{save_dir}/{itr}.png", thumb)
			itr += 1



##
