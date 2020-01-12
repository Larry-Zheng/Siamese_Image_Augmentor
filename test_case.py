from siamese import Siamese_Pipeline

if __name__ == '__main__':
	sia_p = Siamese_Pipeline()
	sia_p.add_further_directory('./original_pic')
	sia_p.ground_truth('./label_pic')
	sia_p.add_siamese_dir('./siamese_pic')
	sia_p.resize(probability=0.5,height=200,width=200)
	sia_p.sample(20)