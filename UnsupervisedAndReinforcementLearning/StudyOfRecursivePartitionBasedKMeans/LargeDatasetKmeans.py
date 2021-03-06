import numpy as np

#If libraries are not installed, visualization would not be available
try:
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt
    visualizable = True
except:
    visualizable = False

class RecursivePartitionKmeans:
    """
    Recursive Partition based K-Means clustering (RPKM).
    Read more in :ref:  Basu, S.; Banerjee, A. & Mooney, R. J.
                        Semi-supervised Clustering by Seeding Proceedings
                        of the Nineteenth International Conference on Machine Learning,
                        2002, 27-34.
    Parameters
    ----------
    n_clusters : int, default=4
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default=300
        Maximum number of iterations of the Recursive Partition based K-means
        (RPKM) algorithm for a each Weighted Lloyd's Algorithm step.

    error_th : float, default=1e-4
        Min distance between C_{i-1} and C_i to consider convergence.
    verbose : int, default=False
        Plot graphics at each step (It will provoke stops at each step).

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point in the training set.
    n_iter_ : int
        Number of iterations run.
    Notes
    -----
    The RPKM is solved running weighted Lloyd's Algorithm.
    The initialization proposed by his original paper is always Forgy's (random),
    it is not dangerous given that the partitions are distributed in the space.
    The RPKM algorithm updates k-means in order to solve the problem when
    datasets are extremely large.
    It uses Forgy's inizialization for setting the first K random representatives.
    The algorithm will iterate while until iterations (i) = max_iters or
    stopping criterion is met.
    Stopping criterion is defined by a threshold of a displacement measure over
    the centroids between iterations.
    The overall complexity in the worst case is O(max(d(n+sum(len(P_{2..m})),len(P_m)Kd)),
    where m refers to max_iters, n to the instances on the datast, P to the subsets
    generated at this iteration, K to the n_clusters parameter, and d to the dimensions
    of the dataset.
    The partitions of the Dataset (D to P_i) are done using quadtrees.
    """

    def __init__(self, n_clusters=4, max_iter=300, error_th = 1e-5,
                 verbose=False, count_operations = True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.error_th = error_th
        self.distance_operations = 0
        self.count_operations = count_operations

        if verbose and (not visualizable):
            self.verbose = False
            import warnings
            warnings.warn("Verbose set to False given an error when importing"
                          "sklearn or matplotlib libraries")
        else:
            self.verbose = verbose

    def fit(self, X, y=None):
        """
        Compute Recursive Partition based k-means (RPKM) clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        y : Ignored
            Not used, present here for API (scikit-learn) consistency by convention.
        Returns
        -------
        self : Fitted estimator.
        """
        #--------------------------------------------------------------------------------------------
        #Step 1: Construct an initial partition of X, P, and define an initial set of K centroids, C.
        #Step 1.1: Spatial Partition: Based on grid based RPKM (see section 2.4 (original paper))

        #Delimite the initial bounding box
        self.bounding_box = np.array([[np.min(X[:,d]), np.max(X[:,d])]
                                      for d in range(X.shape[-1])])
        #Trick for avoiding to consider the limit point in a new partition (max_limit = max_limit-epsilon)
        epsilon = np.finfo(X.dtype).resolution*10
        X[X[..., [0, 1]] == self.bounding_box[:, 1]] -= epsilon
        #Constraint: Partitions (with representatives) > K
        weights = []
        iter = 1
        while(len(weights) < self.n_clusters):
            # Get the partition correspondences (P) the representative of each centroid, and the weights (|S|)
            P, representatives, weights = self.partitionate_dataset(X, iter=iter)
            iter += 1

        #Step 1.2: Construct the initial set of K centroids
        C = self.forgys_cluster_initialization(representatives,self.n_clusters)
        if self.verbose:
            G_x, G_p = self.WL_assignment_step(P=X, C=C), self.WL_assignment_step(P=representatives, C=C,
                                                                                  count_operations=False)
            self.visualize_state(X=X, P=representatives, C=C, G_x=G_x, G_p=G_p, iter=iter-1)
        # --------------------------------------------------------------------------------------------
        # Step 2: Initial Weighted Lloyd
        C, G = self.weighted_lloyd(P=representatives,C=C,weights=weights,thr=self.error_th)
        distance = self.error_th + 1
        while iter < self.max_iter and distance > self.error_th:
            # ----------------------------------------------------------------------------------------
            # Step 3: Construct a dataset partition P' thinner than P
            P, representatives, weights = self.partitionate_dataset(X, iter=iter)

            # ----------------------------------------------------------------------------------------
            # Step 4: Weighted Lloyd
            C_new, _ = self.weighted_lloyd(P=representatives, C=C, weights=weights, thr=self.error_th)
            distance  = np.sum(np.abs(C_new-C))
            C = C_new
            # ----------------------------------------------------------------------------------------
            #Visualization if enabled
            if self.verbose:
                G_x, G_p = self.WL_assignment_step(P=X, C=C), self.WL_assignment_step(P=representatives, C=C,count_operations=False)
                self.visualize_state(X=X, P=representatives, C=C, G_x=G_x, G_p=G_p, iter=iter)
            iter += 1

        #Algorithm finished
        self.cluster_centers = C
        self.labels = self.WL_assignment_step(P=X, C=C, count_operations = False)
        self.n_iter = iter-1

        return self

    def get_inertia(self, X, labels = None):
        """
        Compute the sum of square errors resulting from the fitting.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        labels : Labels assigned to each point of X. If None (default value)
                 it calculates when called.
        Returns
        -------
        Float value of inertia
        """
        self.check_is_fitted()
        if labels is None:
            labels = self.WL_assignment_step(P=X, C=self.cluster_centers, count_operations=False)
        distances = [np.sum(self.euclidean_distance(points=X[labels==i], centroid=centroid, count_operations=False))
                              for i, centroid in enumerate(self.cluster_centers)]
        return np.sum(distances)

    def partitionate_dataset(self, X, iter):
        """
        Compute the dataset partition for dataset X at iteration iter.
        At first iteration, partition is defined by the grid obtained after dividing each side
        of the smallest bounding box of X by half. In following iterations it will be 2**i partitions.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        iteration : iteration in which the partitioning is done.
        Returns
        -------
        Array of arguments of len(X) where each positions refers to the Partition P in which the
        sample is assigned.
        """
        """
        Gives the correspondant position on the division of 2**i parts for each axis with flatten indexes.
        Operation is performed for minimazing the memory allocation.
        Notes
        -----
        (X[:,d]-min_d)/((max_d-min_d)/(2**iter)): It gives in which segment of the dimension d, the
        point is placed.
        ((2**iter)**d): It transforms the identifier to a flatten space ([1,1] --> [1,2]).
        Sum over axis 0: i.e [0,0] --> [0,0] = 0. [0,1] --> [0,2] = 2. [1,1] --> [1,2] = 3.
        """
        d_labels = np.sum(
                        [((X[:,d]-min_d)/((max_d-min_d)/(2**iter))).astype(np.int)*((2**iter)**d)
                        for d, (min_d, max_d) in enumerate(self.bounding_box)],
                    axis=0)
        filled_segments, inverse_idx, weights = np.unique(d_labels, return_inverse=True, return_counts=True)
        # Optimization: Compute faster the cases where a partition has only one point (usual when dimensions scales)
        representatives = np.empty((len(filled_segments), X.shape[-1]), dtype=np.float32)
        unique_partions_mask = weights == 1
        # Correspondencies in X
        unique_partions_X_mask = np.isin(d_labels, filled_segments[unique_partions_mask])
        representatives[unique_partions_mask] = X[unique_partions_X_mask]

        # Compute the rest of partitions in an efficient way
        non_unique_partitions_X_mask = unique_partions_X_mask == False
        non_unique_partitions_mask = unique_partions_mask==False
        non_unique_inverse_idx = inverse_idx[non_unique_partitions_X_mask]
        unique_X_ordered = np.argsort(non_unique_inverse_idx)
        _, cuts = np.unique(non_unique_inverse_idx[unique_X_ordered], return_index=True)
        cuts = cuts.tolist()+[len(unique_X_ordered)]
        non_unique_inverse_args = np.arange(len(X),dtype=np.int)[non_unique_partitions_X_mask][unique_X_ordered]
        if np.sum(non_unique_partitions_mask) > 0:
            representatives[non_unique_partitions_mask] = np.array(
                [np.mean(X[non_unique_inverse_args[cuts[i-1]:cuts[i]]], axis=0)
                 for i in range(1,len(cuts))],
                dtype=X.dtype)
        """
        representatives[non_unique_partitions_mask] = np.array(
                                    [np.mean(X[d_labels==segment], axis=0)
                                    for segment in filled_segments[non_unique_partitions_mask]],
                          dtype=X.dtype)
        """
        return d_labels, representatives, weights

    def weighted_lloyd(self, P, C, weights, thr = 0., count_operations = True):
        """
        Compute the Weighted Lloyd algorithm for the Partition P, using C as initial clusters.
        For the PRKM case, weights of P are defined as his cardinality (|S|)
        and its representative as its center of mass (Sr = sum(X)/|S|).
        Complexity of this proces is O(|P_i|Kd).
        Parameters
        ----------
        P : array-like or sparse matrix, shape=(n_partitions, n_features)
            Array of representatives of each P_i at the current iteration.
        C : array-like or matrix, shape=(K, n_features)
            Clusters of the last iteration.
        weights : array-like or matrix, shape=(len(P))
                  |S|.
        thr : distance thr over the evolution of cluster for considered as converged.
              If set to negative value, no stop condition other than max_iterations.
        Returns
        -------
        Clusters (C) which minimize the error Associations (G).
        """
        #Step 0 : Initial Assignment
        G = self.WL_assignment_step(P, C=C, count_operations=count_operations)

        distance, r = thr+1., 0
        while r<self.max_iter and distance>thr:
            #Step 1: Update Step (C_r <- G_{r-1})
            C_new = self.WL_update_step(P=P, G=G, weights=weights, C=C)

            #Step 2: Assignment Step (G_r <- C_r)
            G = self.WL_assignment_step(P, C=C_new,count_operations=count_operations)

            #Update variables and constraints
            distance  = np.sum(np.abs(C_new-C))
            r += 1
            C = C_new

        return C, G

    def WL_update_step(self,P, G, weights, C):
        """
        Compute the assignment step of Weighted Lloyd algorithm for the Partition P, using C as initial clusters.
        Parameters
        ----------
        P : array-like or sparse matrix, shape=(n_partitions, n_features)
            Array of representatives of each P_i at the current iteration.
        G : array-like, shape=(n_partitions)
            Array with the closest cluster id for each point.
        weights : array-like or matrix, shape=(len(P))
                  |S|.
        C : array-like or matrix, shape=(K, n_features)
            Clusters of the last iteration.
        Returns
        -------
        Clusters (C) updated.
        """
        C_new = np.array(C, copy=True)
        for moved_cluster_id in np.unique(G):
            mask = G==moved_cluster_id
            c_points, c_weights = P[mask], weights[mask]
            C_new[moved_cluster_id] = np.sum((c_points.T*c_weights),axis=1)/np.sum(c_weights)
        return C_new

    def WL_assignment_step(self, P, C, count_operations = True):
        """
        Compute the assignment step of Weighted Lloyd algorithm for the Partition P, using C as initial clusters.
        Parameters
        ----------
        P : array-like or sparse matrix, shape=(n_partitions, n_features)
            Array of representatives of each P_i at the current iteration
        weights : array-like or matrix, shape=(len(P))
                  |S|.
        C : array-like or matrix, shape=(K, n_features)
            Clusters of the last iteration.
        Returns
        -------
        New Associations (G)
        """

        distances = np.array([self.euclidean_distance(points=P, centroid=centroid, count_operations=count_operations) for centroid in C])
        G = np.argmin(distances, axis=0)
        return G

    def euclidean_distance(self, points, centroid, count_operations = True):
        """
        Compute the euclidean distance between a set of points and a centroid.
        Parameters
        ----------
        points : array-like or sparse matrix, shape=(n_points, n_features)
                Array of points.
        centroid : array-like or matrix, shape=(n_features)
                   Given cluster.
        Returns
        -------
        Array of distances.
        """
        if count_operations:
            self.distance_operations += len(points)
        return np.sqrt(np.sum(np.square((points - centroid)), axis=1))

    def forgys_cluster_initialization(self,X, K):
        """Get the Forgy's inicialization of clusters. Which is basically to select
        randomly K Centroids from the given dataset in order to use it as centroids.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        K : Integer
            Number of expected centroids,
        Returns
        -------
        Array of K x dimensions(X) features which represents the selected clusters.
        """
        return X[np.random.choice(len(X), size=K,replace=False)]

    def fit_predict(self, X, y=None):
        """
        Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API (scikit-learn) consistency by convention.
        Returns
        -------
        labels : array, shape [n_samples,]
                 Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        self.check_is_fitted()
        return self.WL_assignment_step(P=X, C=self.cluster_centers)

    def check_is_fitted(self):
        """
        Check if the algorithm is already fitted. If not, raises an error.
        Returns
        -------
        No return
        """
        if not hasattr(self,'cluster_centers'):
            raise Exception("Algorithm not fitted")

    def visualize_state(self,X, P, C, G_x, G_p, iter):
        """
        Visualize the state of the algorithm. His points, his divisions, his representatives,
        and the current clusters.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_points, n_features)
            Array of points (original dataset).
        P : array-like or matrix, shape=(n_representatives, n_features)
            Array of points (representatitives).
            weights : array-like or matrix, shape=(n_representatives)
            The current weight correspondant to each representative (P).
        C : array-like or matrix, shape=(n_clusters)
            Current centroids.
        G_x : array-like of integers, shape=(n_points)
            Labels of each X point, corresponding to the cluster where
            each point belongs.
        G_p : array-like of integers, shape=(n_representatives)
              Labels of each P point, corresponding to the cluster where
              each point belongs.
        iter : The current iteration.
        Returns
        -------
        Plot a 2D graphic of the current state.
        In case of the data being greater than 2D apply PCA over points and do not show C.
        """
        #Reduce the dimensionality of data for visualization through PCA
        bounding_box = self.bounding_box
        if X.shape[-1]>2:
            pca = PCA(n_components=2).fit(X)
            X, P, bounding_box = pca.transform(X), pca.transform(P), pca.transform(bounding_box)
        #Plot all points and the clusters where they belong, as well as representatives
        for i, centroid in enumerate(C):
            points_of_cluster, representatives_of_cluster = X[G_x==i], P[G_p==i]
            plt.plot(points_of_cluster[:,0], points_of_cluster[:,1],
                     'o', color='C'+str(i%10), alpha=0.4)
            plt.plot(representatives_of_cluster[:, 0], representatives_of_cluster[:, 1],
                     'o', color='C'+str(i%10), alpha=1)
        #Plot the lines which divides the Partitions (P)
        y_axis_cuts, x_axis_cuts = [np.linspace(box[0],box[1],(2**iter)+1) for box in bounding_box]
        for line in y_axis_cuts:
            plt.plot([x_axis_cuts[0], x_axis_cuts[-1]],[line, line], color='black')
        for line in y_axis_cuts:
            plt.plot([line, line], [y_axis_cuts[0], y_axis_cuts[-1]], color='black')

        for i, centroid in enumerate(C):
            plt.plot(centroid[0], centroid[1], 'x', color='black', markeredgewidth=11)
            plt.plot(centroid[0], centroid[1], 'x', color='C'+str(i%10), markeredgewidth=5)
        plt.title("Iteration "+str(iter))
        plt.show()