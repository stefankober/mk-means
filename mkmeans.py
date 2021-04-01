from numba import njit
import numpy as np
from sklearn.cluster import KMeans
from numba.typed import List

# for k-lines-plots
from statistics import mode
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


## mk-means all steps

@njit
def create_radius_list(X, cluster_centers, labels):
    """
    Takes X, cluster centers and labels as input, and outputs
    a list of radii for the cluster centers.
    Has been tested with k-means data fitting from scikit-learn.
    Will not work with algorithms that detect outlier, has no
    filter for label -1.
    """
    radius_list = np.empty(cluster_centers.shape[0], dtype="float64")
    for i in np.arange(cluster_centers.shape[0]):
        radius_list[i] = np.amax(np.sqrt((X[labels == i][:, 0]-cluster_centers[i][0])**2 + 
                      (X[labels == i][:, 1]-cluster_centers[i][1])**2))
    return radius_list

@njit
def create_adjacency_list(cluster_centers, radius_list):
    """
    Takes the cluster centers and radii.
    Creates a list of arrays of cluster_centers where the distances
    is smaller than the sum of their radii.
    """
    close_points = []
    for i in np.arange(cluster_centers.shape[0]):
        adjacency_matrix = (np.sqrt((cluster_centers[i][0]-cluster_centers[:, 0])**2 + 
                      (cluster_centers[i][1]-cluster_centers[:, 1])**2) < radius_list[i]+radius_list)
        close_points.append(np.where(adjacency_matrix == True)[0])
    return close_points

@njit
def join_clusters(adjacency_list):
    """
    Takes an adjacency list, creates clusters.
    """
    open_numbers = [np.int64(x) for x in range(0)]
    closed_numbers = [np.int64(x) for x in range(0)]
    clusters = []
    indices = list(range(len(adjacency_list)))
    while indices:
        open_numbers.append(indices.pop())
        while open_numbers:
            number = open_numbers.pop()
            for i in indices:
                if number in adjacency_list[i]:
                    for j in adjacency_list[i]:
                        if not (j in open_numbers or j in closed_numbers or j == number): 
                            open_numbers.append(j)
                    indices.remove(i)
            closed_numbers.append(number)
        clusters.append(closed_numbers)
        closed_numbers = [np.int64(x) for x in range(0)]
    return clusters 

@njit
def generate_new_labels(labels, clusters):
    """
    Takes the old labels and clusters, and
    generates macrocluster labels.
    """
    new_labels = np.empty(labels.shape[0], dtype="int64")
    for i in np.arange(len(clusters)):
        for j in clusters[i]:
            new_labels[labels == j] = -(i+2)
    return new_labels


## mk-means complete function

def generate_macroclusters(X, cluster_centers, labels):
    """
    One function that contains all subfunctions.
    """
    radius_list = create_radius_list(X, cluster_centers, labels)
    a_list = create_adjacency_list(cluster_centers, radius_list)
    typed_a_list = List()
    [typed_a_list.append(x) for x in a_list]
    clusters = join_clusters(typed_a_list)
    typed_clusters = List()
    [typed_clusters.append(x) for x in clusters]
    return generate_new_labels(labels, typed_clusters)


## Prerequisites for k-line-plots

def check_mode_clusters_and_members(X, k, times=10):
    """
    Checks max, mode and min clusters for a certain k value
    by sampling, using generate_macroclusters
    """
    max_clusters=0
    min_clusters=k
    all_clusters=[]
    cluster_members = []
    cluster_members_all = []
    
    for j in np.arange(times):
        model = KMeans(n_clusters=k).fit(X)
        new_labels = generate_macroclusters(X, model.cluster_centers_, model.labels_)
        clusters_found = set(new_labels)
        number_clusters_found = len(clusters_found)
        cluster_numbers_pre = [] 
        for i in clusters_found: 
            cluster_numbers_pre.append(len(X[new_labels == i])) 
        if max_clusters < number_clusters_found:
            max_clusters = number_clusters_found
        cluster_members_all.append(cluster_numbers_pre)
        if min_clusters > number_clusters_found:
            min_clusters = number_clusters_found
        
        all_clusters.append(number_clusters_found)
    mode_clusters = mode(all_clusters)
    for i in cluster_members_all:
        if len(i) == mode_clusters:
            cluster_members = i
            break
    
    return max_clusters, mode_clusters, min_clusters, cluster_members 


def create_pairwise_assignment_difference(X, k_min, k_max):
    """
    Calculates the assignment difference between macroclusters formed from
    model A and model B of X
    """
    model_A = KMeans(n_clusters=k_min).fit(X)
    model_B = KMeans(n_clusters=k_max).fit(X)
    labels_A = generate_macroclusters(X, model_A.cluster_centers_, model_A.labels_)
    labels_B = generate_macroclusters(X, model_B.cluster_centers_, model_B.labels_)
    clusters = []
    if len(set(labels_A)) < len(set(labels_B)):
        labels_A, labels_B = labels_B, labels_A
    all_misaligned = 0
    for label in set(labels_A):
        label_filter = np.where(labels_A == label) # create a filter for the labels in A
        B_filtered = labels_B[label_filter] 
        B_label = stats.mode(B_filtered)[0][0]
        B_filter = B_filtered != B_label
        misaligned = sum(B_filter) # check for misalignments in B
        all_misaligned+=misaligned
    assignment_error=all_misaligned/len(labels_A)
    return assignment_error


## generate k-line data

def generate_k_line_data(X, k_min, k_max, step, times=10):
    """
    Generates all data needed for a k-line plot
    """
    cluster_members = []
    cluster_numbers = []
    mode_cluster_members = []
    min_cluster_members = []
    x_axis = np.arange(k_min, k_max, step)
    alignment_error_list=[]
    for i in x_axis: # this should be parallelized
        clusters, mode_clusters, min_clusters, members  = check_mode_clusters_and_members(X, i, times=times)
        cluster_numbers.append(clusters)
        cluster_members.append(sorted(members))
        mode_cluster_members.append(mode_clusters)
        min_cluster_members.append(min_clusters)
        for i in np.arange(k_min, k_max-step, step):
            alignment_error_list.append(create_pairwise_assignment_difference(X, i, i+step)) 
    alignment_error = np.mean(alignment_error_list)
    return x_axis, cluster_numbers, mode_cluster_members, min_cluster_members, cluster_members, alignment_error


def plot_k_line_data(k_line_data):   
    """
    Plots the k-line data.
    """
    k_range = k_line_data[0]
    max_clusters = k_line_data[1]
    mode_clusters = k_line_data[2]
    min_clusters = k_line_data[3]
    cluster_members = k_line_data[4]
    error = k_line_data[5]*100
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].plot(k_range, max_clusters)
    ax[0].plot(k_range, mode_clusters)
    ax[0].plot(k_range, min_clusters)
    ax[0].legend(["max", "mode", "min"])
    ax[0].set_ylabel("# of clusters")
    ax[0].set_xticks(ticks=k_range)
    
    len_max=0
    for i in cluster_members:
        if len_max < len(i):
            len_max = len(i)
            
    values = np.zeros((len(cluster_members), len_max))
    
    for i in np.arange(len(cluster_members)):
        for j in np.arange(len(cluster_members[i])):
            values[i,j+values.shape[1]-len(cluster_members[i])
                  ] = cluster_members[i][j]
    
    ax[1].stackplot(k_range, values.T)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax[1].set_xlabel("k value")
    ax[1].text(.5, .8, "Mean assignment error:" + str(np.round(error, 2)) + "%",
               horizontalalignment='center',
               verticalalignment='center',
               transform = ax[1].transAxes)
    ax[1].set_ylabel("size mode clusters")
    
    ax[1].set_xticks(ticks=k_range)
    plt.show()
