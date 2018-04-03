import random
from base64 import b64decode
from json import loads
import numpy as np
import matplotlib.pyplot as plt

# read MNIST data
def read_in_data(json_file):
	
	# to parse the line of the digits into tuple

	json_object = loads(json_file)
	json_data = b64decode(json_object["data"])
	digit_vector = np.fromstring(json_data,dtype=np.ubyte)
	digit_vector = digit_vector.astype(np.float64)
	return (json_object["label"],digit_vector)

# read in the digits file
with open("digits.base64.json","r") as f:
	digits = map(read_in_data,f.readlines())

# split the data into training set and validation set
training_size = int(len(digits)*0.25)
validation = digits[:training_size]
training = digits[training_size:]

# take a datapoint and display the digit
def display_digit(digit, labeled = True, title = ""):
	
	if labeled:
		digit = digit[1]
	image = digit
	plt.figure()
	fig = plt.imshow(image.reshape(28,28))
	fig.set_cmap('gray_r')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	if title != "":
		plt.title("Inferred label: " + str(title))


### implementing lloyd's algorithm

### 1. Randomly pick some k centers from the data as the starting values for centroids. Remove all labels.
###	2. Sum a list of arrys
### 3. Compute the mean of a list of vectors, take the sum and then divide by the size of the cluster.
###	4. Form cluster (Given some data and centroids for the data, allocate each datapoints to its closest centroid. This forms cluster)
###	5. Move centroid (Return list of mean centroids corresponding to clusters)
###	6. Form cluster around centroid then keep moving the cnetroid until the moves are no longer significant.
###	7. Run k-means clustering on the data
###	8. Assign a digit label to each centroid 


def init_centroids(labelled_data,k):
    
    return map(lambda x: x[1], random.sample(labelled_data,k))

def sum_cluster(labelled_cluster):
    
    # assumes len(cluster) > 0
    sum_ = labelled_cluster[0][1].copy()
    for (label,vector) in labelled_cluster[1:]:
        sum_ += vector
    return sum_

def mean_cluster(labelled_cluster):
    
    sum_of_points = sum_cluster(labelled_cluster)
    mean_of_points = sum_of_points * (1.0 / len(labelled_cluster))
    return mean_of_points

def form_clusters(labelled_data, unlabelled_centroids):
    
    centroids_indices = range(len(unlabelled_centroids))

    clusters = {c: [] for c in centroids_indices}

    for (label,Xi) in labelled_data:
        # for each datapoint, pick the closest centroid.
        smallest_distance = float("inf")
        for cj_index in centroids_indices:
            cj = unlabelled_centroids[cj_index]
            distance = np.linalg.norm(Xi - cj)
            if distance < smallest_distance:
                closest_centroid_index = cj_index
                smallest_distance = distance
        # allocate that datapoint to the cluster of that centroid.
        clusters[closest_centroid_index].append((label,Xi))
    return clusters.values()

def move_centroids(labelled_clusters):
  
    new_centroids = []
    for cluster in labelled_clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids

def repeat_until_convergence(labelled_data, labelled_clusters, unlabelled_centroids):
    
    previous_max_difference = 0
    while True:
        unlabelled_old_centroids = unlabelled_centroids
        unlabelled_centroids = move_centroids(labelled_clusters)
        labelled_clusters = form_clusters(labelled_data, unlabelled_centroids)
        
        differences = map(lambda a, b: np.linalg.norm(a-b),unlabelled_old_centroids,unlabelled_centroids)
        max_difference = max(differences)
        difference_change = abs((max_difference-previous_max_difference)/np.mean([previous_max_difference,max_difference])) * 100
        previous_max_difference = max_difference
        
        if np.isnan(difference_change):
            break
    return labelled_clusters, unlabelled_centroids

def cluster(labelled_data, k):
    centroids = init_centroids(labelled_data, k)
    clusters = form_clusters(labelled_data, centroids)
    final_clusters, final_centroids = repeat_until_convergence(labelled_data, clusters, centroids)
    return final_clusters, final_centroids

def assign_labels_to_centroids(clusters, centroids):
    
    labelled_centroids = []
    for i in range(len(clusters)):
        labels = map(lambda x: x[0], clusters[i])
        # pick the most common label
        most_common = max(set(labels), key=labels.count)
        centroid = (most_common, centroids[i])
        labelled_centroids.append(centroid)
    return labelled_centroids

def classify_digit(digit, labelled_centroids):
    
    mindistance = float("inf")
    for (label, centroid) in labelled_centroids:
        distance = np.linalg.norm(centroid - digit)
        if distance < mindistance:
            mindistance = distance
            closest_centroid_label = label
    return closest_centroid_label

def get_error_rate(labelled_digits,labelled_centroids):
   
    classified_incorrect = 0
    for (label,digit) in labelled_digits:
        classified_label =classify_digit(digit, labelled_centroids)
        if classified_label != label:
            classified_incorrect +=1
    error_rate = classified_incorrect / float(len(digits))
    return error_rate

k = 16
clusters, centroids = cluster(training, k)
labelled_centroids = assign_labels_to_centroids(clusters, centroids)

for (label,digit) in labelled_centroids:
    display_digit(digit, labeled=False, title=label)






