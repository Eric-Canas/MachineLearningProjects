class RecursivePartitionKmeans:
    """Recursive Partition based K-Means clustering (RPKM).
    Read more in :ref:  Basu, S.; Banerjee, A. & Mooney, R. J.
                        Semi-supervised Clustering by Seeding Proceedings
                        of the Nineteenth International Conference on Machine Learning,
                        2002, 27-34
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default=300
        Maximum number of iterations of the Recursive Partition based K-means
        (RPKM) algorithm for a single run.

    init : {'k-means++', 'random'} or ndarray of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    precompute_distances : 'auto' or bool, default='auto'
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances.
        False : never precompute distances.
    verbose : int, default=0
        Verbosity mode.
    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.
    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
    See Also
    --------
    MiniBatchKMeans
        Alternative online implementation that does incremental updates
        of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.
    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.
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
    Examples
    --------

    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, X, y=None, sample_weight=None):
        """Compute Recursive Partition based k-means (RPKM) clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        y : Ignored
            Not used, present here for API (scikit-learn) consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        Returns
        -------
        self
            Fitted estimator.
        """

        #Step 1: Construct an initial partition of X, P, and define an initial set of K centroids, C.
            #Step 1.1 Spatial Partition: Based on grid based RPKM (see section 2.4 (original paper))
        P = self.partitionate_dataset(X, iter=1)
        """
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        """
        return self

    def partitionate_dataset(self, X, iter):
        """Compute the dataset partition for dataset X at iteration iter.
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
        self
            Array of arguments of len(X) where each positions refers to the Partition P in which the
            sample is assigned.
        """

        pass

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API (scikit-learn) consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        return self.euclidean_distances(X, self.cluster_centers_)

    def euclidean_distances(X, cluster_centers):
        pass

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
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

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).
        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return -_labels_inertia(X, sample_weight, x_squared_norms,
                                self.cluster_centers_)[1]
        """