import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from rake_nltk import Rake
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.image as mpimg
import re
import pickle


########################################################################################################################
## INPUT DATA
########################################################################################################################
ROOT = "/Users/davidhachuel/Google Drive/Cornell Tech/FALL_2017/CS_5785_APPLIED_MACHINE_LEARNING/FINAL/data/"
TRAIN_SET = {
	"descriptions" : {
		"filename" : ["descriptions_train/{}.txt".format(i) for i in range(10000)],
		"data" : []
	},
	"features" : {
		"filename" : [
			"features_train/features_resnet1000_train.csv",
			"features_train/features_resnet1000intermediate_train.csv"
		],
		"data" : []
	},
	"images" : {
		"filename" : ["images_train/{}.jpg".format(i) for i in range(10000)],
		"data" : []
	},
	"tags" : {
		"filename" : ["tags_train/{}.txt".format(i) for i in range(10000)],
		"data" : []
	}
}
TEST_SET = {
	"descriptions" : {
		"filename" : ["descriptions_test/{}.txt".format(i) for i in range(2000)],
		"data" : []
	},
	"features" : {
		"filename" : [
			"features_test/features_resnet1000_train.csv",
			"features_test/features_resnet1000intermediate_train.csv"
		],
		"data" : []
	},
	"images" : {
		"filename" : ["images_test/{}.jpg".format(i) for i in range(2000)],
		"data" : []
	},
	"tags" : {
		"filename" : ["tags_test/{}.txt".format(i) for i in range(2000)],
		"data" : []
	}
}




########################################################################################################################
## DESCRIPTION PRE-PROCESSING
########################################################################################################################
def cleanDescriptionBOW(description: list, nouns_only: bool) -> dict:
	document_tokens, document_keywords = [], []
	for idx, text in enumerate(description):
		# Pre-processing
		text = text.lower() # to lower case
		text = text.strip() # strip white space
		text = re.sub(r'\d+', ' ', text) # remove digits
		text = re.sub(r'[^\w\s]', " ", text) # remove punctuation
		text = re.sub("[ |\t]{2,}", " ", text) # remove tabs

		# Tokenize
		if nouns_only:
			tokens = [
				item[0] \
				for item in nltk.pos_tag(nltk.word_tokenize(text)) \
				if item[1] == "NN"
			]
		else:
			tokens = nltk.word_tokenize(text)


		# Stem
		stemmer=nltk.stem.porter.PorterStemmer()
		tokens = [stemmer.stem(token) for token in tokens]

		# Remove stopwords
		stopwords = nltk.corpus.stopwords.words('english')
		tokens = [token for token in tokens if token not in stopwords]

		# Extract keywords
		r = Rake()
		r.extract_keywords_from_text(text)
		keywords = r.get_ranked_phrases()
		if nouns_only:
			keywords = [
				item[0]
			    for sublist in [
					nltk.pos_tag(nltk.word_tokenize(keyword))
					for keyword in keywords
				]
				for item in sublist if item[1]=="NN"
			]


		document_tokens += tokens
		document_keywords += keywords

	token_count, keyword_count = dict(Counter(document_tokens)), dict(Counter(document_keywords))

	document_bow = {**token_count, **keyword_count}

	return(document_bow)

def getDescriptionBOW(d_set: dict, description_idx: int, nouns_only: bool):
	with open(ROOT + d_set["descriptions"]["filename"][description_idx]) as f:
		raw_lines = f.readlines()

	return (cleanDescriptionBOW(description=raw_lines, nouns_only=nouns_only))

def getBOWTags(d_set: dict, description_idx: int, subcat_multiplier: int= 10):
	with open(ROOT + d_set["tags"]["filename"][description_idx]) as f:
		raw_lines = f.readlines()

	categories, subcategories = [], []
	for idx, text in enumerate(raw_lines):
		# Pre-processing
		text = text.strip()  # strip white space
		text = text.split(":")
		categories.append(text[0])
		subcategories.append(text[1])

	categories_freq, subcategories_freq = dict(Counter(categories)), dict(Counter(subcategories))
	for subcat in subcategories_freq:
		subcategories_freq[subcat] *= subcat_multiplier

	result = {**categories_freq, **subcategories_freq}

	return (result)

##
## RANK TOP CHOICES
##
def rankTopChoices_alt(input_lists):
	if len(set([len(l) for l in input_lists])) != 1:
		raise ValueError("All input lists are not the same length.")

	max_rank = sum([len(l) for l in input_lists])
	output_size = set([len(l) for l in input_lists]).pop()

	presence_score = dict(
		Counter(
			np.concatenate([arr for arr in input_lists])
		)
	)

	rank_score = {}
	for img in presence_score:
		img_rank = np.mean([
			np.where(arr == img)[0][0] if len(np.where(arr == img)[0]) > 0 else max_rank for arr in input_lists
		])
		rank_score[img] = img_rank

	rank = dict(zip(
		list(presence_score.keys()),
		np.array(list(rank_score.values())) / np.array(list(presence_score.values()))
	))

	rank_sorted = [ranked_img[0] for ranked_img in sorted(rank.items(), key=lambda x: x[1])[:output_size]]

	return (rank_sorted)

def rankTopChoices(input_lists, input_weights, output_size=20):
	if len(set([len(l) for l in input_lists])) != 1:
		raise ValueError("All input lists are not the same length.")

	idx_map = list(
		range(
			1,
			set([len(l) for l in input_lists]).pop() + 1
		)
	)[::-1]

	unique_choices = set(np.concatenate(input_lists))

	choice_score = {}
	for choice in unique_choices:
		freq, rank, presence = [], [], []
		for arr, w in zip(input_lists, input_weights):
			if choice in arr:
				freq.append(w)
				rank.append(w * idx_map[np.where(arr == choice)[0][0]])
				presence.append(1)
			else:
				freq.append(0)
				rank.append(0)

		choice_score[choice] = np.dot(freq, rank)

	rank_sorted = [
		ranked_img[0] for ranked_img in sorted(
			choice_score.items(),
			key=lambda x: x[1],
			reverse=True
		)[:output_size]
	]

	return(rank_sorted)





###########################################################################
##                                                                       ##
##                                                                       ##
##                                                                       ##
##               LOAD BOW REPRESENTATION OF DESCRIPTIONS                 ##
##                                                                       ##
##                                                                       ##
##                                                                       ##
###########################################################################
########################################################################################################################
## CREATE BOW ALL TRAIN
########################################################################################################################
bow_all_train_dict_list = []
word_index_all_train = []
for i in range(10000):
	result = getDescriptionBOW(
		d_set=TRAIN_SET,
		description_idx=i,
		nouns_only=False
	)
	bow_all_train_dict_list.append(result)
	word_index_all_train += result.keys()

# Process word index
word_index_all_train = list(set(word_index_all_train))
bow_all_train = np.zeros((10000, len(word_index_all_train)))
for idx, bow_d in enumerate(bow_all_train_dict_list):
	for key in bow_d:
		bow_all_train[idx, word_index_all_train.index(key)] = bow_d[key]
# np.save(ROOT+"bag_of_word/"+"word_index_all_train.npy", word_index_all_train)
# np.save(ROOT+"bag_of_word/"+"bow_all_train.npy", bow_all_train)


########################################################################################################################
## CREATE BOW ALL TEST
########################################################################################################################
bow_all_test_dict_list = []
for i in range(2000):
	result = getDescriptionBOW(
		d_set=TEST_SET,
		description_idx=i,
		nouns_only=False
	)
	bow_all_test_dict_list.append(result)

# Process word index
bow_all_test = np.zeros((2000, len(word_index_all_train)))
for idx, bow_d in enumerate(bow_all_test_dict_list):
	for key in bow_d:
		try:
			bow_all_test[idx, word_index_all_train.index(key)] = bow_d[key]
		except Exception:
			pass
# np.save(ROOT+"bag_of_word/"+"bow_all_test.npy", bow_all_test)


########################################################################################################################
## CREATE BOW NOUN TRAIN
########################################################################################################################
bow_noun_train_dict_list = []
word_index_noun_train = []
for i in range(10000):
	result = getDescriptionBOW(
		d_set=TRAIN_SET,
		description_idx=i,
		nouns_only=True
	)
	bow_noun_train_dict_list.append(result)
	word_index_noun_train += result.keys()

# Process word index
word_index_noun_train = list(set(word_index_noun_train))
bow_noun_train = np.zeros((10000, len(word_index_noun_train)))
for idx, bow_d in enumerate(bow_noun_train_dict_list):
	for key in bow_d:
		bow_noun_train[idx, word_index_noun_train.index(key)] = bow_d[key]
# np.save(ROOT+"bag_of_word/"+"word_index_noun_train.npy", word_index_noun_train)
# np.save(ROOT+"bag_of_word/"+"bow_noun_train.npy", bow_noun_train)


########################################################################################################################
## CREATE BOW NOUN TEST
########################################################################################################################
bow_noun_test_dict_list = []
for i in range(2000):
	result = getDescriptionBOW(
		d_set=TEST_SET,
		description_idx=i,
		nouns_only=True
	)
	bow_noun_test_dict_list.append(result)

# Process word index
bow_noun_test = np.zeros((2000, len(word_index_noun_train)))
for idx, bow_d in enumerate(bow_noun_test_dict_list):
	for key in bow_d:
		try:
			bow_noun_test[idx, word_index_noun_train.index(key)] = bow_d[key]
		except Exception:
			pass
# np.save(ROOT+"bag_of_word/"+"word_index_noun_test.npy", word_index_noun_test)
# np.save(ROOT+"bag_of_word/"+"bow_noun_test.npy", bow_noun_test)


# word_index_noun_train = np.load(ROOT+"bag_of_word/"+"word_index_noun_train.npy")
# bow_noun_train = np.load(ROOT+"bag_of_word/"+"bow_noun_train.npy")
# bow_noun_test = np.load(ROOT+"bag_of_word/"+"bow_noun_test.npy")

# word_index_all_train = np.load(ROOT+"bag_of_word/"+"word_index_all_train.npy")
# bow_all_train = np.load(ROOT+"bag_of_word/"+"bow_all_train.npy")

# bow_all_test = np.load(ROOT+"bag_of_word/"+"bow_all_test.npy")
# tags_index = np.load(ROOT+"bag_of_word/"+"tags_index.npy")

















###################################################################
##                                                               ##
##                                                               ##
##                                                               ##
##               LOAD BOW REPRESENTATION OF TAGS                 ##
##                                                               ##
##                                                               ##
##                                                               ##
###################################################################
########################################################################################################################
## CREATE BOW TRAIN TAGS
########################################################################################################################
bow_train_tags_dict_list = []
tags_index = []
for i in range(10000):
	result = getBOWTags(
		d_set=TRAIN_SET,
		description_idx=i,
		subcat_multiplier=10
	)
	bow_train_tags_dict_list.append(result)
	tags_index += result.keys()

# Process word index
tags_index = list(set(tags_index))
bow_tags_train = np.zeros((10000, len(tags_index)))
for idx, bow_d in enumerate(bow_train_tags_dict_list):
	for key in bow_d:
		bow_tags_train[idx, tags_index.index(key)] = bow_d[key]
# np.save(ROOT+"bag_of_word/"+"tags_index.npy", tags_index)
# np.save(ROOT+"bag_of_word/"+"bow_tags_train.npy", bow_tags_train)


########################################################################################################################
## CREATE BOW TEST TAGS
########################################################################################################################
bow_test_tags_dict_list = []
for i in range(2000):
	result = getBOWTags(
		d_set=TEST_SET,
		description_idx=i,
		subcat_multiplier=10
	)
	bow_test_tags_dict_list.append(result)

# Process word index
bow_tags_test = np.zeros((2000, len(tags_index)))
for idx, bow_d in enumerate(bow_test_tags_dict_list):
	for key in bow_d:
		bow_tags_test[idx, tags_index.index(key)] = bow_d[key]
# np.save(ROOT+"bag_of_word/"+"bow_tags_test.npy", bow_tags_test)



# bow_tags_train = np.load(ROOT+"bag_of_word/"+"bow_tags_train.npy")
# bow_tags_test = np.load(ROOT+"bag_of_word/"+"bow_tags_test.npy")














#############################################################
##                                                         ##
##                                                         ##
##                                                         ##
##               LOAD HIDDEN RESNET LAYERS                 ##
##                                                         ##
##                                                         ##
##                                                         ##
#############################################################
########################################################################################################################
## LOAD FC1000 TRAIN
########################################################################################################################
fc1000_train_raw = pd.read_csv(
	filepath_or_buffer=ROOT+"features_train/features_resnet1000_train.csv",
	header=None
)
fc1000_train_raw[0] = fc1000_train_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
fc1000_train_raw_sorted = fc1000_train_raw.sort_values(by=[0])
fc1000_train_img = fc1000_train_raw_sorted[0].values
fc1000_train = fc1000_train_raw_sorted[list(range(1,1001))].values
# np.save(ROOT+"features_train/"+"fc1000_train_img.npy", fc1000_train_img)
# np.save(ROOT+"features_train/"+"fc1000_train.npy", fc1000_train)


########################################################################################################################
## LOAD FC1000 TEST
########################################################################################################################
fc1000_test_raw = pd.read_csv(
	filepath_or_buffer=ROOT+"features_test/features_resnet1000_test.csv",
	header=None
)
fc1000_test_raw[0] = fc1000_test_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
fc1000_test_raw_sorted = fc1000_test_raw.sort_values(by=[0])
fc1000_test_img = fc1000_test_raw_sorted[0].values
fc1000_test = fc1000_test_raw_sorted[list(range(1,1001))].values
# np.save(ROOT+"features_train/"+"fc1000_test_img.npy", fc1000_test_img)
# np.save(ROOT+"features_train/"+"fc1000_test.npy", fc1000_test)


########################################################################################################################
## LOAD POOL5 TRAIN
########################################################################################################################
pool5_train_raw = pd.read_csv(
	filepath_or_buffer=ROOT+"features_train/features_resnet1000intermediate_train.csv",
	header=None
)
pool5_train_raw[0] = pool5_train_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
pool5_train_raw_sorted = pool5_train_raw.sort_values(by=[0])
pool5_train_img = pool5_train_raw_sorted[0].values
pool5_train = pool5_train_raw_sorted[list(range(1,2049))].values
# np.save(ROOT+"features_train/"+"pool5_train_img.npy", pool5_train_img)
# np.save(ROOT+"features_train/"+"pool5_train.npy", pool5_train)

########################################################################################################################
## LOAD POOL5 TEST
########################################################################################################################
pool5_test_raw = pd.read_csv(
	filepath_or_buffer=ROOT+"features_test/features_resnet1000intermediate_test.csv",
	header=None
)
pool5_test_raw[0] = pool5_test_raw[0].apply(lambda x: int(x.split('/')[1].replace('.jpg','')))
pool5_test_raw_sorted = pool5_test_raw.sort_values(by=[0])
pool5_test_img = pool5_test_raw_sorted[0].values
pool5_test = pool5_test_raw_sorted[list(range(1,2049))].values
# np.save(ROOT+"features_train/"+"pool5_test_img.npy", pool5_test_img)
# np.save(ROOT+"features_train/"+"pool5_test.npy", pool5_test)





# fc1000_train_img = np.load(ROOT+"features_train/"+"fc1000_train_img.npy")
# fc1000_train = np.load(ROOT+"features_train/"+"fc1000_train.npy")
# fc1000_test_img = np.load(ROOT+"features_train/"+"fc1000_test_img.npy")
# fc1000_test = np.load(ROOT+"features_train/"+"fc1000_test.npy")
# pool5_train_img = np.load(ROOT+"features_train/"+"pool5_train_img.npy")
# pool5_train = np.load(ROOT+"features_train/"+"pool5_train.npy")
# pool5_test_img = np.load(ROOT+"features_train/"+"pool5_test_img.npy")
# pool5_test = np.load(ROOT+"features_train/"+"pool5_test.npy")














##############################################
##                                          ##
##                                          ##
##                                          ##
##               PCA ON BOW                 ##
##                                          ##
##                                          ##
##                                          ##
##############################################
########################################################################################################################
## BOW PCA NOUN FC1000
########################################################################################################################
pca_noun_fc1000 = PCA(n_components=1000, svd_solver='auto')
pca_noun_fc1000.fit(bow_noun_train)
bow_noun_train_PCA = pca_noun_fc1000.transform(bow_noun_train)
bow_noun_test_PCA = pca_noun_fc1000.transform(bow_noun_test)
# np.save(ROOT+"bag_of_word/"+"bow_noun_train_PCA.npy", bow_noun_train_PCA)
# np.save(ROOT+"bag_of_word/"+"bow_noun_test_PCA.npy", bow_noun_test_PCA)
# pickle.dump(pca_noun_fc1000, open(ROOT+"models/"+'pca_noun_fc1000.sav', 'wb'))


########################################################################################################################
## BOW PCA ALL POOL5
########################################################################################################################
pca_all_pool5 = PCA(n_components=2048, svd_solver='auto')
pca_all_pool5.fit(bow_all_train)
bow_all_train_PCA = pca_all_pool5.transform(bow_all_train)
bow_all_test_PCA = pca_all_pool5.transform(bow_all_test)
# np.save(ROOT+"bag_of_word/"+"bow_all_train_PCA.npy", bow_all_train_PCA)
# np.save(ROOT+"bag_of_word/"+"bow_all_test_PCA.npy", bow_all_test_PCA)
# pickle.dump(pca_all_pool5, open(ROOT+"models/"+'pca_all_pool5.sav', 'wb'))





# bow_noun_train_PCA =  np.load(ROOT+"bag_of_word/"+"bow_noun_train_PCA.npy")
# bow_noun_test_PCA =  np.load(ROOT+"bag_of_word/"+"bow_noun_test_PCA.npy")
# bow_all_train_PCA =  np.load(ROOT+"bag_of_word/"+"bow_all_train_PCA.npy")
# bow_all_test_PCA = np.load(ROOT+"bag_of_word/"+"bow_all_test_PCA.npy")

# pca_noun_fc1000 = pickle.load(open(ROOT+"models/"+'pca_noun_fc1000.sav', 'rb'))




##########################################
##                                      ##
##                                      ##
##                                      ##
##               MODELS                 ##
##                                      ##
##                                      ##
##                                      ##
##########################################
########################################################################################################################
## PLS REGRESSION NOUN FC1000
########################################################################################################################
# pls_noun_fc1000 = PLSRegression(n_components=1000)
# pls_noun_fc1000.fit(bow_noun_train_PCA, fc1000_train)
# pickle.dump(pls_noun_fc1000, open(ROOT+"models/"+'pls_noun_fc1000_1000.sav', 'wb'))


########################################################################################################################
## MULTILAYER PERCEPTRON REGRESSOR FC1000
########################################################################################################################
# mlp_noun_fc1000 = MLPRegressor(hidden_layer_sizes=(100, 100), activation='tanh')
# mlp_noun_fc1000.fit(bow_noun_train_PCA, fc1000_train)
# pickle.dump(mlp_noun_fc1000, open(ROOT+"models/"+'mlp_noun_fc1000.sav', 'wb'))


########################################################################################################################
## PLS REGRESSION ALL POOL5
########################################################################################################################
pls_all_pool5 = PLSRegression(n_components=2048)
pls_all_pool5.fit(bow_all_train_PCA, pool5_train)
pickle.dump(pls_all_pool5, open(ROOT+"models/"+'pls_all_pool5_2048c.sav', 'wb'))


########################################################################################################################
## MULTILAYER PERCEPTRON REGRESSOR POOL5
########################################################################################################################
mlp_all_pool5 = MLPRegressor(hidden_layer_sizes=(300, 200, 100), activation='tanh')
mlp_all_pool5.fit(bow_all_train_PCA, pool5_train)
# pickle.dump(mlp_all_pool5, open(ROOT+"models/"+'mlp_all_pool5.sav', 'wb'))


########################################################################################################################
## PLS REGRESSION TAGS
########################################################################################################################
pls_noun_tags = PLSRegression(n_components=91)
pls_noun_tags.fit(bow_all_train_PCA, bow_tags_train)
# pickle.dump(pls_noun_tags, open(ROOT+"models/"+'pls_noun_tags_91c.sav', 'wb'))


########################################################################################################################
## MULTILAYER PERCEPTRON REGRESSOR TAGS
########################################################################################################################
mlp_noun_tags = MLPRegressor(hidden_layer_sizes=(200, 100), activation='relu')
mlp_noun_tags.fit(bow_noun_train, bow_tags_train)
# pickle.dump(mlp_all_pool5, open(ROOT+"models/"+'mlp_all_pool5.sav', 'wb'))




from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(random_state=0, verbose=True)
regr.fit(bow_all_train_PCA, pool5_train)

# pls_noun_fc1000 = pickle.load(open(ROOT+"models/"+'pls_noun_fc1000.sav', 'rb'))
# pls_noun_fc1000 = pickle.load(open(ROOT+"models/"+'pls_noun_fc1000_1000.sav', 'rb'))
# mlp_noun_fc1000 = pickle.load(open(ROOT+"models/"+'mlp_noun_fc1000.sav', 'rb'))
# pls_all_pool5 = pickle.load(open(ROOT+"models/"+'pls_all_pool5_200c.sav', 'rb'))
# mlp_all_pool5 = pickle.load(open(ROOT+"models/"+'mlp_all_pool5.sav', 'rb'))
# pls_noun_tags = pickle.load(open(ROOT+"models/"+'pls_noun_tags_91c.sav', 'rb'))


##
## REVERSE APPROACH
##
reverse_approach_nbrs = NearestNeighbors(n_neighbors=10, metric='cosine').fit(pool5_train)
reverse_approach_distances, reverse_approach_indices = reverse_approach_nbrs.kneighbors(pool5_test)

bow_all_test_avg = np.zeros((2000, len(bow_all_train[1])))

for idx in range(reverse_approach_indices.shape[0]):
	for j in reverse_approach_indices[idx]:
		bow_all_test_avg[idx] += bow_all_train[j]
	bow_all_test_avg[idx] = bow_all_test_avg[idx] / 10

















def predict(trained_model, KNN_fit_data, x_to_predict, image_dir, img_index):
	nbrs = NearestNeighbors(n_neighbors=20, metric='cosine').fit(KNN_fit_data)
	distances, indices = nbrs.kneighbors(
		trained_model.predict(x_to_predict)
	)
	del nbrs

	result_images = []
	for img_filename in [img_index[i] for i in indices[0]]:
		result_images.append(mpimg.imread(ROOT+image_dir+str(img_filename)+".jpg"))
	del distances

	plt.figure(1)
	for idx, im in enumerate(result_images):
		plt.subplot(4, 5, idx+1)
		plt.imshow(result_images[idx])
	plt.show()

	return(indices)


## PLS NUN FC1000
predict(
	trained_model=pls_noun_fc1000,
	KNN_fit_data=fc1000_train,
	x_to_predict=np.array([bow_noun_train_PCA[0]]),
	image_dir="images_train/",
	img_index=fc1000_train_img
)
predict(
	trained_model=pls_noun_fc1000,
	KNN_fit_data=fc1000_test,
	x_to_predict=np.array([bow_noun_test_PCA[5]]),
	image_dir="images_test/",
	img_index=fc1000_test_img
)



## PLS POOL5
predict(
	trained_model=pls_all_pool5,
	KNN_fit_data=pool5_train,
	x_to_predict=np.array([bow_all_train_PCA[0]]),
	image_dir="images_train/",
	img_index=pool5_train_img
)
predict(
	trained_model=pls_all_pool5,
	KNN_fit_data=pool5_test,
	x_to_predict=np.array([bow_all_test_PCA[5]]),
	image_dir="images_test/",
	img_index=pool5_test_img
)


## MLP POOL5
predict(
	trained_model=mlp_all_pool5,
	KNN_fit_data=pool5_train,
	x_to_predict=np.array([bow_all_train_PCA[0]]),
	image_dir="images_train/",
	img_index=pool5_train_img
)
predict(
	trained_model=mlp_all_pool5,
	KNN_fit_data=pool5_test,
	x_to_predict=np.array([bow_all_test_PCA[5]]),
	image_dir="images_test/",
	img_index=pool5_test_img
)




## PLS TAGS
predict(
	trained_model=pls_noun_tags,
	KNN_fit_data=bow_tags_train,
	x_to_predict=np.array([bow_all_train_PCA[0]]),
	image_dir="images_train/",
	img_index=list(range(10000))
)
predict(
	trained_model=pls_noun_tags,
	KNN_fit_data=bow_tags_test,
	x_to_predict=np.array([bow_all_test_PCA[5]]),
	image_dir="images_test/",
	img_index=list(range(2000))
)


## PLS TAGS
predict(
	trained_model=regr,
	KNN_fit_data=pool5_train,
	x_to_predict=np.array([bow_all_train_PCA[0]]),
	image_dir="images_train/",
	img_index=list(range(10000))
)
predict(
	trained_model=regr,
	KNN_fit_data=pool5_test,
	x_to_predict=np.array([bow_all_test_PCA[5]]),
	image_dir="images_test/",
	img_index=list(range(2000))
)








########################################################################################################################
## FORMAT SUBMISSION
########################################################################################################################


# pls_noun_fc1000_predictions = pls_noun_fc1000.predict(bow_noun_test_PCA)
# test_nbrs_fc1000 = NearestNeighbors(n_neighbors=80, metric='cosine').fit(fc1000_test)
# pls_noun_fc1000_predictions_distances, pls_noun_fc1000_predictions_indices = test_nbrs_fc1000.kneighbors(
# 	pls_noun_fc1000_predictions
# )

# dist, indic = test_nbrs_pool5.kneighbors(
# 	np.mean([
# 		pls_all_pool5_predictions,
# 		mlp_all_pool5_predictions
# 	], axis=0)
# )


rf_noun_tags_predictions = regr.predict(bow_all_test_PCA)
test_nbrs_rf_noun_tags = NearestNeighbors(n_neighbors=80, metric='cosine').fit(pool5_test)
rf_noun_tags_predictions_distances, rf_noun_tags_predictions_indices = test_nbrs_rf_noun_tags.kneighbors(
	rf_noun_tags_predictions
)



pls_all_pool5_predictions = pls_all_pool5.predict(bow_all_test_PCA)
test_nbrs_pool5 = NearestNeighbors(n_neighbors=80, metric='cosine').fit(pool5_test)
pls_all_pool5_predictions_distances, pls_all_pool5_predictions_indices = test_nbrs_pool5.kneighbors(
	pls_all_pool5_predictions
)

pls_noun_tags_prediction = pls_noun_tags.predict(bow_all_test_PCA)
test_nbrs_tags = NearestNeighbors(n_neighbors=80, metric='cosine').fit(bow_tags_test)
pls_noun_tags_prediction_distances, pls_noun_tags_prediction_indices = test_nbrs_tags.kneighbors(
	pls_noun_tags_prediction
)


reverse_approach_nbrs_end = NearestNeighbors(n_neighbors=80, metric='cosine').fit(bow_all_test_avg)
reverse_approach_distances_end, reverse_approach_indices_end = reverse_approach_nbrs_end.kneighbors(bow_all_test)





final_indices = []

for i in range(2000):
	final_indices.append(
		rankTopChoices(
			input_lists=[
				pls_all_pool5_predictions_indices[i],
				pls_noun_tags_prediction_indices[i],
				reverse_approach_indices_end[i],
				rf_noun_tags_predictions_indices[i]
			],
			input_weights=[
				28,
				24,
				25,
				11
			]
		)
	)
final_indices=np.array(final_indices)






submission = pd.DataFrame({
	"Descritpion_ID":["{}.txt".format(i) for i in list(range(2000))],
	"Top_20_Image_IDs":[" ".join(["{}.jpg".format(i) for i in  indx]) for indx in rf_noun_tags_predictions_indices]
})
submission.to_csv(
	ROOT+"submissions/all.csv",
	index=False
)


## SCORES
# pls_noun_fc1000 : 0.18744
# mlp_noun_fc1000 : 0.12617
# pls_all_pool5 : 0.28027
# mlp_all_pool5 : 0.22942
# pls_mpl_fc1000_pool5_raw_combo : 0.22053
# pls_mlp_pool5_average : 0.23819
# pls_noun_tags : 0.20374
# reverse_approach_indices_end : 0.25222




#
# ## PAGE RANK
# from itertools import combinations
# import networkx as nx
# g = nx.DiGraph()
# g.add_nodes_from(list(range(2000)))
# for tag_idx in range(91):
# 	connected_imgs = list(np.where(bow_tags_test[:,tag_idx]>0)[0])
# 	edges = combinations(connected_imgs, 2)
# 	for edge in edges:
# 		g.add_edge(edge[0], edge[1])
# 		g.add_edge(edge[1], edge[0])
# pr = nx.pagerank(G=g, alpha=1/7.)
#
#
# import copy
#
# final_indices_pr = copy.deepcopy(final_indices)
# for top_choices_idx in range(final_indices.shape[0]):
# 	top_choices = final_indices[top_choices_idx]
# 	top_choices_pr = [pr[choice] for choice in top_choices]
# 	choice_dict = dict(zip(
# 		top_choices,
# 		top_choices_pr
# 	))
# 	sorted_choices = sorted(choice_dict.items(), key=lambda x: x[1], reverse=False)
# 	final_indices_pr[top_choices_idx] = np.array([choice[0] for choice in sorted_choices])
#























