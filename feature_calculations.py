from config import *
from helpers.neighbors import spherical_neighborhood, flatten_neighbors

@njit(parallel=True, cache=True)
def ev_numba(xyz, ii_flat, lookup):

    evals = np.zeros((lookup.shape[0]-1, 3))
    evecs = np.zeros((lookup.shape[0]-1, 9))  # Flattened 3x3 matrix

    for i in prange(lookup.shape[0]-1):
        nn = ii_flat[lookup[i]:lookup[i+1]]
        q = xyz[nn, :]
        
        # Compute mean (centroid)
        cx = q[:, 0].mean()
        cy = q[:, 1].mean()
        cz = q[:, 2].mean()
        c = np.array([cx, cy, cz])
        
        # Subtract centroid
        q_centered = q - c

        # Covariance matrix (3x3)
        cov = np.zeros((3, 3))
        cov = q_centered.T @ q_centered / q_centered.shape[0]
        cov /= q_centered.shape[0]

        # Eigen decomposition
        evalue, evector = np.linalg.eigh(cov)
        evalue = evalue[::-1]           # descending
        evector = evector[:, ::-1]      # reorder accordingly

        evals[i, :] = evalue
        evecs[i, :] = evector.flatten()

    return evals, evecs

@njit(cache=True)
def compute_skewness(mean_val, median_val, std_val):
    return 0.0 if std_val == 0 else (mean_val - median_val) / std_val

@njit(cache=True)
def compute_mode(vals):
    counts = Dict.empty(key_type=types.float64, value_type=types.int64)
    max_count = 0
    mode_val = vals[0]
    for v in vals:
        key = float(v)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
        if counts[key] > max_count:
            max_count = counts[key]
            mode_val = key
    return mode_val

@njit(parallel=True, cache=True)
def scattered_echo(echo, ii_flat, lookup):
    n_points, n_features = lookup.shape[0]-1, echo.shape[1]

    echo_mean = np.zeros((n_points, n_features))
    echo_median = np.zeros((n_points, n_features))
    echo_std = np.zeros((n_points, n_features))
    echo_mode = np.zeros((n_points, n_features))
    echo_skew = np.zeros((n_points, n_features))
    echo_range = np.zeros((n_points, n_features))

    for i in prange(n_points):
        nn = ii_flat[lookup[i]:lookup[i+1]]

        for j in range(n_features):
            vals = echo[nn, j]
            n = vals.size

            sum_val = 0.0
            sum_sq = 0.0
            min_val = vals[0]
            max_val = vals[0]

            for v in vals:
                sum_val += v
                sum_sq += v * v
                if v < min_val:
                    min_val = v
                if v > max_val:
                    max_val = v

            mean_val = sum_val / n
            range_val = max_val - min_val
            median_val = np.median(vals)
            mode_val = compute_mode(vals)

            # std
            var_val = sum_sq / n - mean_val * mean_val
            if var_val > 0.0:
                std_val = var_val ** 0.5
            else:
                std_val = 0.0

            # skewness
            if std_val == 0.0:
                skew_val = 0.0
            else:
                skew_val = compute_skewness(mean_val, median_val, std_val)

            # store
            echo_mean[i, j] = mean_val
            echo_median[i, j] = median_val
            echo_std[i, j] = std_val
            echo_mode[i, j] = mode_val
            echo_skew[i, j] = skew_val
            echo_range[i, j] = range_val

    return echo_mean, echo_median, echo_std, echo_mode, echo_skew, echo_range


@njit(parallel=True, cache=True)
def transmissivity(intermediate_returns, ii_flat, lookup):
    n = lookup.shape[0] - 1
    penetrable = np.zeros(n, dtype=np.float32)

    for i in prange(n):
        nn_start = lookup[i]
        nn_end = lookup[i+1]
        nn_sum = 0.0
        count = nn_end - nn_start

        for j in range(nn_start, nn_end):
            nn_sum += intermediate_returns[ii_flat[j]]

        penetrable[i] = nn_sum / count

    return penetrable.reshape(-1, 1)

@njit(cache=True)
def percentileofscore(a, score):
    n = len(a)
    sorted_a = np.sort(a)
    count_less = np.searchsorted(sorted_a, score, side='left')
    count_equal = np.searchsorted(sorted_a, score, side='right') - count_less
    percentiles = (count_less + 0.5 * count_equal) / n * 100
    return percentiles

@njit(parallel=True, cache=True)
def height_features(xyz, ii_flat, lookup, indices):

    height_score = np.zeros(lookup.shape[0]-1)
    local_height_min = np.zeros(lookup.shape[0]-1)
    local_height_max = np.zeros(lookup.shape[0]-1)
    local_height_std = np.zeros(lookup.shape[0]-1)
    local_height_range = np.zeros(lookup.shape[0]-1)

    for i in prange(lookup.shape[0]-1):
        pt = indices[i]
        nn = ii_flat[lookup[i]:lookup[i+1]]
        nnz = xyz[nn, 2]
        height_score[i] = percentileofscore(nnz, xyz[pt, 2])
        local_height_min[i] = xyz[pt, 2] - nnz.min()
        local_height_max[i] = nnz.max() - xyz[pt, 2]
        local_height_std[i] = np.sqrt(((nnz - nnz.mean()) ** 2).sum() / len(nnz))
        local_height_range[i] = nnz.max() - nnz.min()

    return height_score, local_height_min, local_height_max, local_height_std, local_height_range


@njit(parallel=True, cache=True)
def numba_moment(xyz, ii_flat, lookup, evec1, evec2, indices):
    n_points = lookup.shape[0] - 1
    first_mom_1 = np.zeros(n_points)
    first_mom_2 = np.zeros(n_points)
    sec_mom_1 = np.zeros(n_points)
    sec_mom_2 = np.zeros(n_points)

    for i in prange(n_points):
        nn = ii_flat[lookup[i]:lookup[i+1]]
        pt = xyz[indices[i]]

        fm1 = 0.0
        fm2 = 0.0
        sm1 = 0.0
        sm2 = 0.0

        for j in range(nn.size):
            diff0 = xyz[nn[j], 0] - pt[0]
            diff1 = xyz[nn[j], 1] - pt[1]
            diff2 = xyz[nn[j], 2] - pt[2]

            d1 = diff0 * evec1[i, 0] + diff1 * evec1[i, 1] + diff2 * evec1[i, 2]
            d2 = diff0 * evec2[i, 0] + diff1 * evec2[i, 1] + diff2 * evec2[i, 2]

            fm1 += d1
            fm2 += d2
            sm1 += d1 * d1
            sm2 += d2 * d2

        first_mom_1[i] = fm1
        first_mom_2[i] = fm2
        sec_mom_1[i] = sm1
        sec_mom_2[i] = sm2

    return first_mom_1, first_mom_2, sec_mom_1, sec_mom_2

@njit(parallel=True, cache=True)
def numba_verticality(evec3):
    n = evec3.shape[0]
    verticality = np.zeros(n)
    for i in prange(n):
        dot = evec3[i, 2]  # since dot([0,0,1], v) = v[2]
        verticality[i] = 1.0 - abs(dot)
    return verticality

@njit(parallel=True, cache=True)
def flip_eigenvectors(evec):
    n = evec.shape[0]
    flipped = np.empty_like(evec)
    for i in prange(n):
        for j in range(3):  # for each eigenvector
            vec = evec[i, j*3:(j+1)*3]
            s = vec.sum()
            if s < 0:
                flipped[i, j*3:(j+1)*3] = -vec
            else:
                flipped[i, j*3:(j+1)*3] = vec
    return flipped

@njit(parallel=True, cache=True)
def compute_eigen_features(l1, l2, l3):
    n = l1.size
    eigenvalue_sum = np.empty(n)
    omnivariance = np.empty(n)
    eigenentropy = np.empty(n)
    anisotropy = np.empty(n)
    planarity = np.empty(n)
    linearity = np.empty(n)
    sphericity = np.empty(n)
    PCA1 = np.empty(n)
    PCA2 = np.empty(n)
    PCA3 = np.empty(n)
    surface_curvature = np.empty(n)

    for i in prange(n):
        lsum = l1[i] + l2[i] + l3[i]
        eigenvalue_sum[i] = lsum
        omnivariance[i] = (l1[i]*l2[i]*l3[i])**(1.0/3.0)
        eigenentropy[i] = -(l1[i]*np.log(l1[i]) + l2[i]*np.log(l2[i]) + l3[i]*np.log(l3[i]))
        anisotropy[i] = (l1[i]-l3[i])/l1[i]
        planarity[i] = (l2[i]-l3[i])/l1[i]
        linearity[i] = (l1[i]-l2[i])/l1[i]
        sphericity[i] = l3[i]/l1[i]
        PCA1[i] = l1[i]/lsum
        PCA2[i] = l2[i]/lsum
        PCA3[i] = l3[i]/lsum
        surface_curvature[i] = l1[i]/lsum

    return (eigenvalue_sum, omnivariance, eigenentropy, anisotropy,
            PCA1, PCA2, PCA3, surface_curvature, linearity, planarity, sphericity)

def geometric_features(xyz, ii_flat, lookup, indices):

    # get eigenvalues and eigenvectors
    eval, evec = ev_numba(xyz, ii_flat, lookup)

    # l1 >= l2 >= l3
    l1 = eval[:, 0]
    l2 = eval[:, 1]
    l3 = eval[:, 2]

    # clean numerical precision errors: if the smallest eigenvalue is zero or below, set it to a very small positive value
    l1 = np.where(l1 <= 0, 1e-45, l1)
    l2 = np.where(l2 <= 0, 1e-45, l2)
    l3 = np.where(l3 <= 0, 1e-45, l3)

    evec = flip_eigenvectors(evec)

    # compute geometric features
    eigenvalue_sum, omnivariance, eigenentropy, anisotropy, PCA1, PCA2, PCA3, surface_curvature, linearity, planarity, sphericity = compute_eigen_features(l1, l2, l3)

    # verticality
    verticality = numba_verticality(np.ascontiguousarray(evec[:, 6:9]))

    # moment 
    first_mom_1, first_mom_2, sec_mom_1, sec_mom_2 = numba_moment(xyz, ii_flat, lookup, np.ascontiguousarray(evec[:, :3]), np.ascontiguousarray(evec[:, 3:6]), indices)

    # combine
    geo_features = np.c_[eigenvalue_sum, omnivariance, eigenentropy, anisotropy, PCA1, PCA2, PCA3, surface_curvature, linearity, planarity, sphericity, verticality, first_mom_1, first_mom_2, sec_mom_1, sec_mom_2]
    geo_fnames = np.array(("Sum of Eigenvalues ", "Omnivariance", "Eigenentropy", "Anisotropy", "PCA 1",  "PCA 2",  "PCA 3",  "Surface Curvature ", "Linearity", "Planarity",  "Sphericity", "Verticality", "First_mom_1", "First_mom_2", "Sec_mom_1", "Sec_mom_2"))

    return geo_features, geo_fnames


@njit(parallel=True, cache=True)
def return_based(return_number, number_of_returns, ii_flat, lookup):

    n_points = lookup.shape[0] - 1

    # categories
    single = (number_of_returns == 1)
    multi = (number_of_returns > 1)
    first = (number_of_returns > 1) & (return_number == 1)
    intermediate = (return_number != 1) & (return_number != number_of_returns) # not first and not last
    last = (number_of_returns > 1) & (return_number == number_of_returns)

    # map 4 categories to numbers 0-3
    return_cat = np.zeros(single.shape[0], dtype=np.int8)
    return_cat[single] = 0
    return_cat[first] = 1
    return_cat[intermediate] = 2
    return_cat[last] = 3

    # Allocate outputs
    rutzinger2008 = np.zeros(n_points, dtype=np.float32)
    zhang2013 = np.zeros(n_points, dtype=np.float32)
    kuprowski2023 = np.zeros(n_points, dtype=np.float32)

    # Loop over neighborhoods
    for i in prange(n_points):
        nn = ii_flat[lookup[i]:lookup[i+1]]
        single_count = single[nn].sum()
        if single_count == 0:
             rutzinger2008[i] = (first[nn].sum() + intermediate[nn].sum())
        else:
            rutzinger2008[i] = (first[nn].sum() + intermediate[nn].sum()) / single_count
        zhang2013[i] = multi[nn].sum() / len(nn)
        kuprowski2023[i] = np.var(return_cat[nn])

    return rutzinger2008, zhang2013, kuprowski2023, return_cat


def get_return_based_chunks(xyz, return_number, number_of_returns, radius, chunk_size=10):

    # get chunks
    chunks_x = (xyz[:, 0] // chunk_size).astype(int)
    chunks_y = (xyz[:, 1] // chunk_size).astype(int)
    chunks = chunks_x + chunks_y * (np.max(chunks_x) + 1)

    # built tree for whole dataset
    tree = cKDTree(xyz)

    rutzinger2008 = []
    zhang2013 = []
    kuprowski2023 = []
    point_indices = []

    for c in tqdm.tqdm(np.unique(chunks), desc="Processing Chunks", unit="chunk"):

        # get points from current chunk
        chunk_points_indices = np.where((chunks == c))[0]
        point_indices.append(chunk_points_indices)
        chunk_points_mask = chunks == c

        ii = tree.query_ball_point(xyz[chunk_points_mask], radius, workers=-1)
        ii_flat, lookup = flatten_neighbors(ii)
        r2008, z2013, k2023, __ = return_based(xyz[chunk_points_mask], return_number, number_of_returns, ii_flat, lookup)
        rutzinger2008.append(r2008)
        zhang2013.append(z2013)
        kuprowski2023.append(k2023)

    # concatenate and sort results
    rutzinger2008 = np.concatenate(rutzinger2008)
    zhang2013 = np.concatenate(zhang2013)
    kuprowski2023 = np.concatenate(kuprowski2023)
    point_indices = np.concatenate(point_indices)
    rutzinger2008 = rutzinger2008[point_indices.argsort()]
    zhang2013 = zhang2013[point_indices.argsort()]
    kuprowski2023 = kuprowski2023[point_indices.argsort()]

    return np.c_[rutzinger2008], np.c_[zhang2013], np.c_[kuprowski2023]

def features(xyz, radius, echo, echo_names, neighborhood = None, indices = None, mask = None, tree = None, tree_lr = None, tree_ir = None, indices_lr = None, indices_ir = None):

    t0 = time()

       # calculate neighborhoods
    if neighborhood is not None: # chunk version 
        ii_flat, lookup = neighborhood[0], neighborhood[1]
    else:
        # complete dataset
        ii_flat, lookup = spherical_neighborhood(xyz, radius)


    # calculate features
    echo_mean, echo_median, echo_std, echo_mode, echo_skew, echo_range =  scattered_echo(echo, ii_flat, lookup)
    rutzinger2008, zhang2013, kuprowski2023, __ = return_based(echo[:, 1], echo[:, 2], ii_flat, lookup)
    scatt_echo = np.c_[echo_mean, echo_median, echo_std, echo_mode, echo_skew, echo_range, rutzinger2008, zhang2013, kuprowski2023]
    height_score, local_height_min, local_height_max, local_height_std, local_height_range = height_features(xyz, ii_flat, lookup, indices)
    height = np.c_[height_score, local_height_min, local_height_max, local_height_std, local_height_range]
    geom_features, geom_names = geometric_features(xyz, ii_flat, lookup, indices)

    # transmissivity
    intermediate = np.where(echo[:, 1] != echo[:, 2], 1, 0) # not last return
    t = transmissivity(intermediate, ii_flat, lookup)
    t_names = np.array(("Transmissivity", ))

    echo_sl = echo[mask]

    # combine features
    features = np.c_[echo_sl, 
                    scatt_echo,
                    height,
                    geom_features, 
                    t]
    
    categories = np.zeros(features.shape[1])
    categories[0:echo.shape[1]] = 0  # echo
    categories[echo.shape[1]:echo.shape[1]+scatt_echo.shape[1]] = 1
    categories[echo.shape[1]+scatt_echo.shape[1]:echo.shape[1]+scatt_echo.shape[1]+height.shape[1]] = 2
    categories[echo.shape[1]+scatt_echo.shape[1]+height.shape[1]:echo.shape[1]+scatt_echo.shape[1]+height.shape[1]+geom_features.shape[1]] = 3
    categories[-t.shape[1]:] = 1  # transmissivity


    # create remaining names
    func_names = ["Mean", "Median", "Std", "Mode", "Skew", "Range"]
    scatt_echo_names = np.zeros((len(echo_names)*len(func_names)), dtype='object')
    counter = 0
    for name in func_names:
        for fname in echo_names:
            scatt_echo_names[counter] = f"{fname} ({name})"
            counter += 1
    scatt_echo_names = np.concatenate((scatt_echo_names, np.array(("Rutzinger2008", "Zhang2013", "Kuprowski2023"))))
    height_names = np.array(("Zscore", "Zmin", "Zmax", "Zstd", "Zrange"))
    names = np.concatenate((echo_names, scatt_echo_names, height_names, geom_names, t_names))

    time_taken = time() - t0
    return features, categories, names, time_taken


def chunkswise_features(selected_points, points, chunk_size, radius, echo, echo_names):

    """
    selected_points: boolean array indicating which points to calculate features for
    """

    # select all if not specified
    if selected_points is None:
        selected_points = np.ones(points.shape[0], dtype=bool)
    selected_mask = np.zeros(points.shape[0], dtype=bool)
    selected_mask[selected_points] = True

    # get chunks
    chunks_x = (points[:, 0] // chunk_size).astype(int)
    chunks_y = (points[:, 1] // chunk_size).astype(int)
    chunks = chunks_x + chunks_y * (np.max(chunks_x) + 1)

    # built tree for whole dataset
    tree = cKDTree(points)

    # build return-dependent trees
    last_returns = np.where(echo[:, 1] == echo[:, 2], True, False)
    tree_lr = cKDTree(points[last_returns])
    tree_ir = cKDTree(points[~last_returns])

    # loop over chunks
    feature = []
    point_indices = []
    ii_counts = []  # to store number of neighbors for each point

    for c in tqdm.tqdm(np.unique(chunks), desc="Processing Chunks", unit="chunk"):

        # boolean mask for chunk points 
        chunk_points_mask = (chunks == c) & selected_points

        # skip empty chunks
        if chunk_points_mask.sum == 0:
            continue  

        # get indices of points in chunk for sorting later
        chunk_points_indices = np.where(chunk_points_mask)[0]
        point_indices.append(chunk_points_indices)

        # last return and inter return indices 
        chunk_points_indices_lr = np.where((chunks == c)  & last_returns & selected_points)[0]
        chunk_points_indices_ir = np.where((chunks == c)  & ~last_returns & selected_points)[0]

        # calculate neighborhoods for chunk points
        ii = tree.query_ball_point(points[chunk_points_mask], radius, workers=-1)

        # number of neighbors
        iic = np.array([len(neighbors) for neighbors in ii])

        if len(ii) == 0:
            print("Found point with no neighbors, skipping...")
            continue
        else:
            ii_flat, lookup = flatten_neighbors(ii)

        # calculate features for chunk
        f, echo_dependet, names, ttime = features(points,
                                                radius,
                                                echo, echo_names,
                                                neighborhood=(ii_flat, lookup),
                                                indices=chunk_points_indices,
                                                mask=chunk_points_mask,
                                                tree=tree, tree_lr=tree_lr,
                                                tree_ir=tree_ir,
                                                indices_lr=chunk_points_indices_lr,
                                                indices_ir=chunk_points_indices_ir) 

        # append results
        feature.append(f)
        ii_counts.append(iic)

    # concatenate and sort results
    feature = np.concatenate(feature)
    point_indices = np.concatenate(point_indices)
    ii_counts = np.concatenate(ii_counts)

    print("Final feature shape:", feature.shape, point_indices.shape)
    feature = feature[point_indices.argsort()]
    ii_counts = ii_counts[point_indices.argsort()]

    return feature, echo_dependet, names, ttime, ii_counts

