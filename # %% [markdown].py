# %% [markdown]
# 
# Compressing images using Python
# 
# Compressing images is a neat way to shrink the size of an image while maintaining the resolution. In this tutorial we’re building an image compressor using Python, Numpy and Pillow. We’ll be using machine learning, the unsupervised K-means algorithm to be precise.
# 
# If you don’t have Numpy and Pillow installed, you can do so using the following command:
# 
# pip3 install pillow
# pip3 install numpy
# 
# Start by importing the following libraries:

# %%
import os 
import sys
from PIL import Image
import numpy as np

# %% [markdown]
# The K-means algorithm
# 
# K-means, as mentioned in the introduction, is an unsupervised machine learning algorithm. Simplified, the major difference between unsupervised and supervised machine learning algorithms is that supervised learning algorithms learn by example (labels are in the dataset), and unsupervised learning algorithms learn by trial and error; you label the data yourself.
# 
# I’ll be explaining how the algorithm works based on an example. In the illustration below there are two features,
# x1 and x2.
# 
# We want to assign each item to one out of two clusters. The most natural way to do this would be something like this:
# 
# Colored points on a 2 plane
# 
# K-means is an algorithm to do exactly this. Note that K is the number of clusters we want to have, hence the name K means.
# 
# Note: Follow documentation here "https://rickwierenga.com/blog/machine%20learning/image-compressor-in-Python.html"

# %% [markdown]
# Implementing K-means
# 
# We start by implementing a function that creates initial points for the centroids. This function takes as input X, the training examples, and chooses
# distinct points at random.

# %%
def intialize_K_centroids(X,K):
	"""Choose K points from X at random"""
	m = len(X)
	return X[np.random.choice(m,K, replace=False), :]

# %% [markdown]
# Then we write a function to find the closest centroid for each training example. This is the first step of the algorithm. We take X and the centroids as input and return the the index of the closest centroid for every example in c, an m-dimensional vector.

# %%
def find_closest_centroids(X, centroids):
	m = len(X)
	c = np.zeros(m)
	for i in range(m):
		# Find distance
		distance = np.linalg.norm(X[i] - centroids, axis=1)

		# Asign closest cluster to c[i]
		c[i] = np.argmin(distance)

	return c
	

# %% [markdown]
# or the second step of the algorithm, we compute the distance of each example to ‘its’ centroid and take the average of distance for every centroid muk. Because we’re looping over the rows, we have to transpose the examples.

# %%
def compute_means(X, idx, K):
	_, n = X.shape
	centroids = np.zeros((K,n))
	for k in range(K):
		examples = X[np.where(idx == k)]
		mean = [np.mean(column) for column in examples.T]
		centroids[k] = mean
	return centroids

# %% [markdown]
# 

# %% [markdown]
# Finally, we got all the ingredients to complete the K-means algorithm. We set max_iter, the maximum number of iterations, to 10. Note that if the centroids aren’t moving anymore, we return the results because we cannot optimize any further.

# %%
def find_k_means(X, K, max_iters=10):
	centroids = intialize_K_centroids(X, K)
	previous_centroids = centroids
	for _ in range(max_iters):
		idx = find_closest_centroids(X, centroids)
		centroids = compute_means(X, idx, K)
		if(centroids == previous_centroids).all():
			# The centroids aren't moving anymore.
			return centroids
		else:
			previous_centroids = centroids
	return centroids, idx

# %% [markdown]
# Getting the image
# 
# In case you’ve never worked with Pillow before, don’t worry. The api is very easy.
# 
# We start by trying to open the image, which is defined as the first (and last) command line argument like so:

# %%
try:
	image_path = sys.argv[1]
	assert os.path.isfile(image_path)
except (IndexError, AssertionError):
	print('Please specify an image')

# %% [markdown]
# Pillow gives us an Image object, but our algorithm requires a NumPy array. So let’s define a little helper function to convert them. Notice how each value is divided by 255 to scale the pixels to 0…1 (a good ML practice).

# %%
def load_image(path):
	""" Load image from path. Return a numpy array """
	image = Image.open(path)
	return np.asarray(image) / 255

# %% [markdown]
# Here’s how to use that:

# %%
pathOfImage = "D:\corner_stone\ocr\IMG_1333.JPG"
image = load_image(pathOfImage)
w, h, d = image.shape
print('Image found with width : {}, height: {}, depth: {}'.format(w,h,d))

# %% [markdown]
# Then we get our feature matrix X. We’re reshaping the image because each pixel has the same meaning (color), so they don’t have to be presented as a grid.

# %%
X = image.reshape((w * h, d))
K = 20 # the desired number of the colors in the compressed image

# %% [markdown]
# Finally we can make use of our algorithm and get the K colors. These colors are chosen by the algorithm.

# %%
colors, _ = find_k_means(X, K,max_iters=10)

# %% [markdown]
# Because the indexes returned by the find_k_means function are 1 iteration behind the colors, we compute the indexes for the current colors. Each pixel has a value in 0...K corresponding, of course, to its color.

# %%
idx = find_closest_centroids(X, colors)

# %% [markdown]
# Once we have all the data required we reconstruct the image by substituting the color index with the color and resphaping the image back to its original dimensions. Then using the Pillow function Image.fromarray we convert the raw numbers back to an image. We also convert the indexes to integers because numpy only accepts those as indexes for matrices.

# %%
idx = np.array(idx, dtype=np.uint8)
X_reconstructed = np.array(colors[idx, :] * 255, dtype= np.uint8).reshape((w,h,d))
compressed_image= Image.fromarray(X_reconstructed)

# %% [markdown]
# Finally, we save the image back to the disk like so:

# %%
compressed_image.save('out.png')


