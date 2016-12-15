def train(gm_train, wm_train, n_wm_pcs=10):
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    
    n_trials, n_wm_voxels = wm_train.shape
    _, n_gm_voxels = gm_train.shape

    pca = PCA(n_components=n_wm_pcs)
    wm_train_pcs = pca.fit_transform(wm_train)
    
    model = LinearRegression()
    model.fit(wm_train_pcs, gm_train)

    return model, pca
