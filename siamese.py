import Augmentor
from Augmentor.ImageUtilities import scan
from PIL import Image
import uuid
import os
import random

class Siamese_Pipeline(Augmentor.Pipeline):
	def __init__(self,output_directory="output"):
		super(Siamese_Pipeline,self).__init__()

	def add_siamese_dir(self,dir):
		abs_output_directory = os.path.join(dir, 'output')
		siamese_images, self.class_labels = scan(dir, abs_output_directory)
		for i in range(len(self.augmentor_images)):
			self.augmentor_images[i].siamese = siamese_images[i].image_path

	def _execute(self, augmentor_image, save_to_disk=True, multi_threaded=True):
		"""
		Private method. Used to pass an image through the current pipeline,
		and return the augmented image.

		The returned image can then either be saved to disk or simply passed
		back to the user. Currently this is fixed to True, as Augmentor
		has only been implemented to save to disk at present.

		:param augmentor_image: The image to pass through the pipeline.
		:param save_to_disk: Whether to save the image to disk. Currently
		 fixed to true.
		:type augmentor_image: :class:`ImageUtilities.AugmentorImage`
		:type save_to_disk: Boolean
		:return: The augmented image.
		"""

		images = []

		if augmentor_image.image_path is not None:
			images.append(Image.open(augmentor_image.image_path))

		# What if they are array data?
		if augmentor_image.pil_images is not None:
			images.append(augmentor_image.pil_images)

		if augmentor_image.ground_truth is not None:
			if isinstance(augmentor_image.ground_truth, list):
				for image in augmentor_image.ground_truth:
					images.append(Image.open(image))
			else:
				images.append(Image.open(augmentor_image.ground_truth))

		if augmentor_image.siamese is not None:
			if isinstance(augmentor_image.siamese, list):
				for image in augmentor_image.siamese:
					images.append(Image.open(image))
			else:
				images.append(Image.open(augmentor_image.siamese))

		for operation in self.operations:
			r = round(random.uniform(0, 1), 1)
			if r <= operation.probability:
				images = operation.perform_operation(images)

		# TEMP FOR TESTING
		# save_to_disk = False

		if save_to_disk:
			file_name = str(uuid.uuid4())
			try:
				for i in range(len(images)):
					if i == 0:
						save_name = augmentor_image.class_label \
									+ "_original_" \
									+ os.path.basename(augmentor_image.image_path) \
									+ "_" \
									+ file_name \
									+ "." \
									+ (self.save_format if self.save_format else augmentor_image.file_format)

						images[i].save(os.path.join(augmentor_image.output_directory, save_name))

					elif i==1:
						save_name = "_groundtruth_(" \
									+ str(i) \
									+ ")_" \
									+ augmentor_image.class_label \
									+ "_" \
									+ os.path.basename(augmentor_image.image_path) \
									+ "_" \
									+ file_name \
									+ "." \
									+ (self.save_format if self.save_format else augmentor_image.file_format)

						images[i].save(os.path.join(augmentor_image.output_directory, save_name))

					else:
						save_name = augmentor_image.class_label \
									+ "_siamese_" \
									+ os.path.basename(augmentor_image.image_path) \
									+ "_" \
									+ file_name \
									+ "." \
									+ (self.save_format if self.save_format else augmentor_image.file_format)

						images[i].save(os.path.join(augmentor_image.output_directory, save_name))


			except IOError as e:
				print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
				print("You can change the save format using the set_save_format(save_format) function.")
				print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")

		# TODO: Fix this really strange behaviour.
		# As a workaround, we can pass the same back and basically
		# ignore the multi_threaded parameter completely for now.
		# if multi_threaded:
		#   return os.path.basename(augmentor_image.image_path)
		# else:
		#   return images[0]  # Here we return only the first image for the generators.

		# return images[0]  # old method.
		return images[0]


