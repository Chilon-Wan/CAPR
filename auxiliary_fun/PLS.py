"""
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""
import numpy as np
from scipy.linalg import pinv, svd
from scipy.sparse.linalg import svds
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def _nipals_twoblocks_inner_loop(X, Y, mode="A", max_iter=500, tol=1e-06,
                                 norm_y_weights=False):
    """
    提供svd(X'Y)的替代方案; 返回X'Y的第一个左右奇异向量。有关参数的含义,
    请参见PLS。它类似于确定X'Y的特征向量和特征值的幂方法。 
    """
    for col in Y.T:
        if np.any(np.abs(col) > np.finfo(np.double).eps):
            y_score = col.reshape(len(col), 1)
            break

    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = np.finfo(X.dtype).eps

    if mode == "B":
        X_t = X.dtype.char.lower()
        Y_t = Y.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}

        cond_X = factor[X_t] * eps
        cond_Y = factor[Y_t] * eps

    # Inner loop of the Wold algo.
    while True:
        # 1.1 Update u: the X weights
        if mode == "B":
            if X_pinv is None:
                # We use slower pinv (same as np.linalg.pinv) for stability
                # reasons
                X_pinv = pinv(X, check_finite=False, cond=cond_X)
            x_weights = np.dot(X_pinv, y_score)
        else:  # mode A
            # Mode A regress each X column on y_score
            x_weights = np.dot(X.T, y_score) / np.dot(y_score.T, y_score)
        # If y_score only has zeros x_weights will only have zeros. In
        # this case add an epsilon to converge to a more acceptable
        # solution
        if np.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        # 1.2 Normalize u
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) + eps
        # 1.3 Update x_score: the X latent scores
        x_score = np.dot(X, x_weights)
        # 2.1 Update y_weights
        if mode == "B":
            if Y_pinv is None:
                # compute once pinv(Y)
                Y_pinv = pinv(Y, check_finite=False, cond=cond_Y)
            y_weights = np.dot(Y_pinv, x_score)
        else:
            # Mode A regress each Y column on x_score
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        # 2.2 Normalize y_weights
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights.T, y_weights)) + eps
        # 2.3 Update y_score: the Y latent scores
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights.T, y_weights) + eps)
        # y_score = np.dot(Y, y_weights) / np.dot(y_score.T, y_score) ## BUG
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite

def _svd_cross_product(X, Y):
    C = np.dot(X.T, Y)
    U, s, Vh = svd(C, full_matrices=False)
    u = U[:, [0]]
    v = Vh.T[:, [0]]
    return u, v

def _center_scale_xy(X, Y, scale=True):
    """ Center X, Y and scale if the scale parameter==True
    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y = Y- y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std

def svd_flip(u, v, u_based_decision=True):
    """ 符号校正以确保SVD的确定输出。调整u的列和v的行, 以使u中绝对值最大的列中的载荷始终为正。
    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        如果为True, 则使用u列作为符号翻转的基础。否则, 使用v的行。决定基于哪个变量的选择通常取决于算法。 
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

class _PLS():
    def __init__(self, n_components=2, *, scale=True,
                 deflation_mode="regression",
                 mode="A", algorithm="nipals", norm_y_weights=False,
                 max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.norm_y_weights = norm_y_weights
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        if self.algorithm not in ("svd", "nipals"):
            raise ValueError("Got algorithm %s when only 'svd' "
                             "and 'nipals' are known" % self.algorithm)
        if self.algorithm == "svd" and self.mode == "B":
            raise ValueError('Incompatible configuration: mode B is not '
                             'implemented with svd algorithm')
        if self.deflation_mode not in ["canonical", "regression"]:
            raise ValueError('The deflation mode is unknown')
        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(X, Y, self.scale))
        # Residuals (deflated) matrices
        Xk = X
        Yk = Y
        # Results matrices
        self.x_scores_ = np.zeros((n, self.n_components))
        self.y_scores_ = np.zeros((n, self.n_components))
        self.x_weights_ = np.zeros((p, self.n_components))
        self.y_weights_ = np.zeros((q, self.n_components))
        self.x_loadings_ = np.zeros((p, self.n_components))
        self.y_loadings_ = np.zeros((q, self.n_components))
        self.n_iter_ = []

        # NIPALS algo: outer loop, over components
        Y_eps = np.finfo(Yk.dtype).eps
        for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
            if self.algorithm == "nipals":
                # Replace columns that are all close to zero with zeros
                Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                Yk[:, Yk_mask] = 0.0

                x_weights, y_weights, n_iter_ = \
                    _nipals_twoblocks_inner_loop(
                        X=Xk, Y=Yk, mode=self.mode, max_iter=self.max_iter,
                        tol=self.tol, norm_y_weights=self.norm_y_weights)
                self.n_iter_.append(n_iter_)
            elif self.algorithm == "svd":
                x_weights, y_weights = _svd_cross_product(X=Xk, Y=Yk)
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'
            x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            y_weights = y_weights.T
            # compute scores
            x_scores = np.dot(Xk, x_weights)
            if self.norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            # test for null variance
            if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps:
                break
            # 2) Deflation (in place)
            # ----------------------
            # 这里可以减少内存占用：为了避免为秩1近似矩阵分配数据块，然后将其减去Xk，我们建议执行逐列通缩。
            #
            # - regress Xk's on x_score
            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            # - subtract rank-one approximations to obtain remainder matrix
            Xk -= np.dot(x_scores, x_loadings.T)
            if self.deflation_mode == "canonical":
                # - regress Yk's on y_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, y_scores) / np.dot(y_scores.T, y_scores))
                Yk -= np.dot(y_scores, y_loadings.T)
            if self.deflation_mode == "regression":
                # - regress Yk's on x_score, then subtract rank-one approx.
                y_loadings = (np.dot(Yk.T, x_scores) / np.dot(x_scores.T, x_scores))
                Yk -= np.dot(x_scores, y_loadings.T)
            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q
        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        self.x_rotations_ = np.dot( self.x_weights_, 
                                    pinv(np.dot(self.x_loadings_.T, self.x_weights_),
                                        check_finite=False))
        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot( self.y_weights_,
                                        pinv(np.dot(self.y_loadings_.T, self.y_weights_),
                                                     check_finite=False))
        else:
            self.y_rotations_ = np.ones(1)

        if True or self.deflation_mode == "regression":
            # FIXME what's with the if?
            # Estimate regression coefficient
            # Regress Y on T
            # Y = TQ' + Err,
            # Then express in function of X
            # Y = X W(P'W)^-1Q' + Err = XB + Err
            # => B = W*Q' (p x q)
            self.coefOrigin_ = np.dot(self.x_rotations_, self.y_loadings_.T)
            self.coef_ = self.coefOrigin_ * self.y_std_
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self.y_mean_
            Y /= self.y_std_
            y_scores = np.dot(Y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X):
        """Transform data back to its original space.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of pls components.
        Returns
        -------
        x_reconstructed : array-like of shape (n_samples, n_features)
        Notes
        -----
        This transformation will only be exact if n_components=n_features
        """
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T)

        # Denormalize
        X_reconstructed *= self.x_std_
        X_reconstructed += self.x_mean_
        return X_reconstructed

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.
        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        Ypred = np.dot(X, self.coef_)
        return Ypred  + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.
        y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        return {'poor_score': True,
                'requires_y': False}

class PLSRegressionWithoutLib(_PLS):
    def __init__(self, n_components=2, *, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="regression", mode="A",
            norm_y_weights=False, max_iter=max_iter, tol=tol,
            copy=copy)

class PLSCanonicalWithoutLib(_PLS):
    def __init__(self, n_components=2, *, scale=True, algorithm="nipals",
                 max_iter=500, tol=1e-06, copy=True):
        super().__init__(
            n_components=n_components, scale=scale,
            deflation_mode="canonical", mode="A",
            norm_y_weights=True, algorithm=algorithm,
            max_iter=max_iter, tol=tol, copy=copy)

# 使用官方库，可以多线程计算
def PLS_Estim1(X, y, maxLatentVarNum, cv, mpFlag=False):
    '''
    x :光谱矩阵 nxm
    y :浓度阵 （化学值）
    maxLatentVarNum :最大潜变量数
    cv :交叉验证数量
    '''
    if mpFlag:
        n_jobs = -1
    else:
        n_jobs = None
    parameters = {'n_components':[i+1 for i in range(maxLatentVarNum)],}
    clf = PLSRegression()
    GS = GridSearchCV(clf, param_grid=parameters,
                    scoring='neg_root_mean_squared_error', # 以最大负均方误差（最小均方误差）为优化目标
                    cv=cv, n_jobs=n_jobs) # 运用所有线程计算
    GS = GS.fit(X, y)

    latentVarNum = GS.best_params_['n_components'] # 最优主成分
    RMSE = -GS.best_score_  # 最小均方误差
    coef = np.ravel(GS.best_estimator_.coef_) # 最佳回归系数

    result = {
        'RMSE':          RMSE,
        'latentVarNum':  latentVarNum,
        'coef':          coef,
        }
    return result

# 使用自己的库，不用多线程计算
def PLS_Estim2(X, y, maxLatentVarNum, cv, mpFlag=False):
    '''
    x :光谱矩阵 nxm
    y :浓度阵 （化学值）
    maxLatentVarNum :最大潜变量数
    cv :交叉验证数量
    '''
    scores = []
    model = PLSRegressionWithoutLib
    # model = PLSRegression
    for pcn in range(1, maxLatentVarNum+1):
        clf = model(n_components=pcn)
        KF = KFold(n_splits=cv)
        SR = 0
        for train_index, test_index in KF.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pre = clf.predict(X_test)
            # print(np.ravel(y_pre).shape, y_test.shape)
            SR = SR + np.sum((np.ravel(y_pre) - y_test)**2)
            # SR += mean_squared_error(y_pre, y_test)/cv
            # plt.plot((np.ravel(y_pre) - y_test)**2)
            # plt.show()
        scores.append((SR/len(y))**0.5)

    bestIndex = np.argmin(scores)
    latentVarNum = bestIndex + 1 # 最优主成分
    RMSE = scores[bestIndex]  # 最小均方误差
    clf = model(n_components=latentVarNum)
    clf.fit(X, y)
    coef = np.ravel(clf.coef_) # 最佳回归系数

    result = {
        'RMSE':          RMSE,
        'latentVarNum':  latentVarNum,
        'coef':          coef,
        }
    return result


