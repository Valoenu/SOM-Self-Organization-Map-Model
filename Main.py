# Self Organization Map ("Easiest Unsupervised DeepLearning model")
# Example - "fraud detector for company"
# Fraud - Outline neurons


#Imports the main libraries for:
	#•	pandas: data loading and manipulation
	#•	numpy: numerical computations
	#•	matplotlib.pyplot: plotting and visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Data preprocessing

# Loads the dataset into a Pandas DataFrame from a CSV file.
data = pd.read_csv("Credit_Card_Application.csv")

# Set x and y, 	•	x: all columns except the last (features). y: the last column (approval result: 1 = approved, 0 = rejected)
y = data.iloc[: -1].values 
x = data.iloc[:, :-1].values


# Scaling Feature
# Import libraries
from sklearn.preprocessing import MinMaxScaler

# Set Scaller
scaller = MinMaxScaler(feature_range = (0, 1))
x = scaller.fit_transform(x)

# Now we're ready to train SOM 
# Both Option) Inform SOM from scratch or import from other developer, (make sure that they have license to use!)

# import file, class from another developer (I'm decide to use MiniSom)
from minisom import MiniSom

# Set paremeters
som = MiniSom(x=10, y=10, input_len=15, sigma=0, learning_rate=0.5) # sigma parameter (radius of the different neighbourhoods in the grid). learning_rate (how much time we need)

# Initialise the weights to small numbers close to 0, but not exactly zero
som.random_weights_init(x) # x data which our model will be trained

som.train_random(x, num_iterations=85) # num_iterations (how much iterations we want)

# Visualize our data to find outliers, find MID (Mean interneuron distance) (mean of the distance of the winning neuron around the neighborhooh) The more the MID = Highest chance to be outlier

# import important libraries to start building our map
from pylab import bone, pcolor, colorbar, plot, show

# Start Structuring our map, from scratch
bone()

# Set colors for MID
pcolor(som.distance_map().T) # Take transponde for the MID matrix (Take value in correct order)

# Now use colorbar() to se legend of our color, Small MID ≈ Dark Color, High MID ≈ White color
colorbar()

markers = ['o', 's'] # Two Markers, Defines two marker shapes for plotting customers: 	'o' = circle 	's' = square

# Color our markers
colors = ['r', 'g'] # 'r' - red, 'g' - green, 	•	x: all columns except the last (features)	y: the last column (approval result: 1 = approved, 0 = rejected)

# for each customer we'are going get the winning node and depending on wheter the customer get approveal or not, 
# I - all the index for our customers
# X - All the vectors of the customers at itterations
#Iterates over every customer. i is index, x is the input vector for a customer.

for i, x in enumerate(x):
  # Get winning node
  winning_node = som.winner(x)
  # Plot the marker, depends on wheter the customer get approveal
  plot(winning_node[0] + 0.5, winning_node[1] + 0.5, #Coordinate of our node x, y axis, + 0.5 to put the marker at the center of the square
      markers[y[i]], #Depended value of the customer, give us marker, "i" if customer get approveal that it will return either 1 or 0 (e.x marker at position 1 = 's', and the same way with 0 value)
      markeredgecolor = color[y[i]],
      markerfacecolor = "None",
      markersize = 10,
      markeredgewidth = 2)
show()

# get the explicit list of the customers who potentialy cheating

# Dictionary, all the mappings for our winning node, it will give us a list of customers
mappings = som.win_map(x) # X = our dataset

# Fraud (Use mapping function)
# Get cordinate outlier winning node
#Concatenates vectors from two neurons (identified as suspicious) into one array of potential fraud cases.
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
# axis = 0 verticaly, 


# Inverse data (scaling). Returns the scaled features of fraud suspects back to their original values (useful for reviewing raw customer data).
frauds = scaller.inverse_transform(frauds)
