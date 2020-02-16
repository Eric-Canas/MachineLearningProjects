import numpy as np

class RecursivePartitionKmeans:
    """Recursive Partition based K-Means clustering (RPKM).
    Read more in :ref:  Basu, S.; Banerjee, A. & Mooney, R. J.
                        Semi-supervised Clustering by Seeding Proceedings
                        of the Nineteenth International Conference on Machine Learning,
                        2002, 27-34
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
        Plot graphics at each step (It will provoke stops at each step)

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
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
                 verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.error_th = error_th
        self.verbose = verbose

    def fit(self, X, y=None):
        """Compute Recursive Partition based k-means (RPKM) clustering.
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
        self
            Fitted estimator.
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
        while(len(weights)< self.n_clusters):
            # Get the partition correspondences (P) the representative of each centroid, and the weights (|S|)
            P, representatives, weights = self.partitionate_dataset(X, iter=iter)
            iter += 1

        #Step 1.2: Construct the initial set of K centroids
        C = self.forgys_cluster_initialization(representatives,self.n_clusters)

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
            C_new, G = self.weighted_lloyd(P=representatives, C=C, weights=weights, thr=self.error_th)
            distance  = np.sum(np.abs(C_new-C))
            iter += 1
            C = C_new

        self.cluster_centers_ = C
        self.labels_ = self.WL_assignment_step(P, C=C)
        self.n_iter_ = iter-1

        return self

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
        Operation is performed for minimazing the memory allocation
        Notes
        -----
        (X[:,d]-min_d)/((max_d-min_d)/(2**iter)): It gives in which segment of the dimension d, the
        point is placed
        ((2**iter)**d): It transforms the identifier to a flatten space ([1,1] --> [1,2])
        Sum over axis 0: i.e [0,0] --> [0,0] = 0. [0,1] --> [0,2] = 2. [1,1] --> [1,2] = 3.
        """
        d_labels = np.sum(
                        [((X[:,d]-min_d)/((max_d-min_d)/(2**iter))).astype(np.int)*((2**iter)**d)
                        for d, (min_d, max_d) in enumerate(self.bounding_box)],
                    axis=0)
        filled_segments, weights = np.unique(d_labels, return_counts=True)
        representatives = np.array(
                                    [np.mean(X[d_labels==segment], axis=0)
                                    for segment in filled_segments],
                          dtype=X.dtype)
        return d_labels, representatives, weights

    def weighted_lloyd(self, P, C, weights, thr = 0.):
        """
        Compute the Weighted Lloyd algorithm for the Partition P, using C as initial clusters.
        For the PRKM case, weights of P are defined as his cardinality (|S|)
        and its representative as its center of mass (Sr = sum(X)/|S|).
        Complexity of this proces is O(|P_i|Kd).

        Parameters
        ----------
        P : array-like or sparse matrix, shape=(n_partitions, n_features)
            Array of representatives of each P_i at the current iteration
        C : array-like or matrix, shape=(K, n_features)
            Clusters of the last iteration
        weights : array-like or matrix, shape=(len(P))
                  |S|
        thr : distance thr over the evolution of cluster for considered as converged.
              If set to negative value, no stop condition other than max_iterations.
        Returns
        -------
        Clusters (C) which minimize the error Associations (G)
        """
        #Step 0 : Initial Assignment
        G = self.WL_assignment_step(P, C=C)

        distance, r = thr+1, 0
        while r<self.max_iter and distance>thr:
            #Step 1: Update Step (C_r <- G_{r-1})
            C_new = self.WL_update_step(P=P, G=G, weights=weights, C=C)

            #Step 2: Assignment Step (G_r <- C_r)
            G = self.WL_assignment_step(P, C=C_new)

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
            Array of representatives of each P_i at the current iteration
        G : array-like, shape=(n_partitions)
            Array with the closest cluster id for each point
        weights : array-like or matrix, shape=(len(P))
                  |S|
        C : array-like or matrix, shape=(K, n_features)
            Clusters of the last iteration
        Returns
        -------
        Clusters (C) updated
        """
        C_new = np.array(C, copy=True)
        for moved_cluster_id in np.unique(G):
            mask = G==moved_cluster_id
            c_points, c_weights = P[mask], weights[mask]
            C_new[moved_cluster_id] = np.sum((c_points.T*c_weights),axis=1)/np.sum(c_weights)
        return C_new

    def WL_assignment_step(self, P, C):
        """
        Compute the assignment step of Weighted Lloyd algorithm for the Partition P, using C as initial clusters.
        Parameters
        ----------
        P : array-like or sparse matrix, shape=(n_partitions, n_features)
            Array of representatives of each P_i at the current iteration
        weights : array-like or matrix, shape=(len(P))
                  |S|
        C : array-like or matrix, shape=(K, n_features)
            Clusters of the last iteration
        Returns
        -------
        New Associations (G)
        """

        distances = np.array([self.euclidean_distance(points=P, centroid=centroid) for centroid in C])
        G = np.argmin(distances, axis=0)
        return G

    def euclidean_distance(self, points, centroid):
        """
        Compute the euclidean distance between a set of points and a centroid.
        Parameters
        ----------
        points : array-like or sparse matrix, shape=(n_points, n_features)
                Array of points
        centroid : array-like or matrix, shape=(n_features)
                   Given cluster
        Returns
        -------
        Array of distances
        """
        return np.sqrt(np.sum(np.square((points - centroid)), axis=1))

    def forgys_cluster_initialization(self,X, K):
        """Get the Forgy's inicialization of clusters. Which is basically to select
        randomly K Centroids from the given dataset in order to use it as centroids
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        K : Number of expected centroids
        Returns
        -------
        Array of K x dimensions(X) features which represents the selected clusters.
        """
        return X[np.random.choice(len(X), size=K)]

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
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
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
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

        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return _labels_inertia(X, sample_weight, x_squared_norms,
                               self.cluster_centers_)[0]
        """