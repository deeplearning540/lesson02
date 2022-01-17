import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def main():
    nfeats = 4
    rng = np.random.default_rng(12345)

    for i in range(10):
        std = rng.uniform(1, 4, nfeats)
        X, y = make_blobs(
            n_samples=1000, centers=4, n_features=4, cluster_std=std, random_state=i
        )

        label = (y > 1).astype(np.int8)

        d = {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "x3": X[:, 2],
            "x4": X[:, 3],
            "label": label,
            "raw": y,
        }
        df = pd.DataFrame(data=d)
        fn = "clustering_data_{:02.0f}.csv".format(i)
        print(df.shape, "->", fn)
        df.to_csv(fn)


if __name__ == "__main__":
    main()
