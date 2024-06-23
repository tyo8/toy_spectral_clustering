import os
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


def run_model(
        model_type="clusters", 
        outdir="toy_models", 
        quantile_thresh=0.1, 
        plot=True,
        N = 250
        ):
    print(f"Running toy model \"{model_type}\"...")
    if model_type == "clusters":
        graph_only=False
        x,y,z = clusters(N=N)
        c = x**2 + y**2 + z**2

    elif model_type == "cloud":
        graph_only=False
        x,y,z = cloud(N=N)
        c = x**2 + y**2 + z**2

    elif model_type == "warped_circle":
        graph_only=False
        x,y,z,c = warped_circle(N=N)

    elif model_type == "k_cycle":
        graph_only=True
        G = nx.cycle_graph(N)

    else:
        raise Exception(f"Unrecogized model type: {model_type}")

    if plot:
        savename=None
    else:
        savename=os.path.join(outdir, f"{model_type}_%s.png")

    if not graph_only:
        plot_3d(
                x, y, z, c=c, 
                plot=plot,
                savename = savename % "pts"
                )

        gram_mtx = euclid_pdist(x, y, z)
        G = VR_graph(
                gram_mtx, 
                quantile_thresh=quantile_thresh
                )

    viz_graph(G, savename = savename % "graph", plot=plot)

    # computes null space of incidence matrix and Laplacian eigenanalysis 
    graph_summary(G, savename, model_type=model_type)

    print("Done.\n")



#####################################################################################################################################################
def plot_3d(x, y, z, c=None, plot=True, savename=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c)
    if plot:
        plt.show()
    else:
        plt.savefig(savename, dpi=300, format='png')
        plt.close()

# computes the pairwise Euclidean distance between a list of points in 3-space
def euclid_pdist(x, y, z):
    agg = np.array([x, y, z])
    gram_mtx = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(agg.T))
    return gram_mtx


# computes the 1-skeleton of the Vietoris-Rips complex for a given (real-valued) Gram matrix,
# where epsilon is given by an (optional) quantile threshold (default = 0.1 distance quantile)
def VR_graph(gram_mtx, quantile_thresh=0.1):
    distvals = gram_mtx[np.triu_indices(gram_mtx.shape[0],1)]
    thresh = np.quantile(distvals, quantile_thresh)
    adj = (gram_mtx > 0) * (gram_mtx < thresh)
    G = nx.convert_matrix.from_numpy_array(adj)
    return G

# vizualizes a graph object G by either plotting it (live/interactive) or saving it as a .png
def viz_graph(G, savename, plot=True):
    nx.draw(G, node_size=15)
    if plot:
        plt.show()
    else:
        plt.savefig(savename, dpi=300, format='png')

    plt.close()


# give a networkx graph object G, pulls the incidence matrix and computes (a) the dimension of 
# its null space and (b) an eigenanalysis of its Laplacian
def graph_summary(G, savename, model_type=None, tol=1e-6):
    S = nx.incidence_matrix(G, oriented=True)
    null_dim = scipy.linalg.null_space(S.todense()).shape[1]

    print(f"Corresponding incidence matrix has a null space of dimension {null_dim} and thus ~{null_dim} cycles (many may be trivial)")

    L = (S @ S.T).todense()

    l, v = np.linalg.eigh(L)
    k = np.count_nonzero(l < tol)

    print(f"Laplacian eigenvalue 0 has multiplicity ~{k} and thus ~{k} components.")

    Fiedler_idx = np.where(l > tol)[0][0]

    plt.plot(l)
    plt.title("Spectrum of graph Laplacian")
    plt.savefig(savename % "specLap", dpi=300, format='png')
    plt.close()

    plt.plot(v[:,Fiedler_idx])
    plt.title("Components of smallest non-null Laplacian eigenvector (Fiedler vector)")
    plt.savefig(savename % "vecFiedler", dpi=300, format='png')
    plt.close()
#####################################################################################################################################################


#####################################################################################################################################################
## WARPED CIRCLE
# generate and plot warped circle (unknot) in R3:
def warped_circle(N=250, snr=10, shape_par=1.1, n_loops=1, amp=1):
    t = np.linspace(0, n_loops*2*np.pi, num=N)
    x = shape_par*np.cos(t) + np.random.randn(N)/snr
    y = np.sin(t) + np.random.randn(N)/snr
    z = np.cos(n_loops*2*t)*np.sin(n_loops*2*t)+amp*t*np.sin(t) + np.random.randn(N)/snr
    return x, y, z, t


## UNSTRUCTURED CLOUD
def cloud(N=250, radius=1, dist='unif'):
    if dist=='unif':
        genfun = np.random.rand
    if dist=='gauss':
        genfun = np.random.randn

    x=radius*genfun(N)
    y=radius*genfun(N)
    z=radius*genfun(N)
    return x, y, z

## CLUSTERS
def clusters(N=250, n_clust=3, avg_radius=1, sep_ratio=2.5):
    centers = []
    partitions = [0]
    for i in range(n_clust):
        center = sep_ratio*np.random.randn(3)
        centers.append(center)
        if i > 0:
            interval = [i*N/(n_clust+1), (i+1)*N/(n_clust+1)]
            clust_idx = np.random.randint(interval[0], interval[1])
            partitions.append(clust_idx)

    partitions.sort()
    partitions.append(-1)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    for i,center in enumerate(centers):
        a = partitions[i]
        b = partitions[i+1]
        if i < (n_clust-1):
            num=b-a
        else:
            num=N-1-a
        x[a:b] = avg_radius*np.random.randn(num) + center[0]
        y[a:b] = avg_radius*np.random.randn(num) + center[1]
        z[a:b] = avg_radius*np.random.randn(num) + center[2]

    return x, y, z
#####################################################################################################################################################

toy_models = ["warped_circle","cloud","clusters","k_cycle"]
# toy_models = ["warped_circle","cloud"]
# toy_models = ["clusters"]
# toy_models = ["k_cycle"]
if __name__=="__main__":
    N = 250
    quantile_thresh=0.15
    for model in toy_models:
        run_model(
                model_type=model, 
                quantile_thresh=quantile_thresh,
                plot=False,
                N=N
                )
