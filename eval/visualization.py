
import numpy as np
import matplotlib.pyplot as plt


def denormalize(im):
	image = im.numpy()
	im = (image - np.min(image)) / (np.max(image) - np.min(image))
	im = np.ascontiguousarray(im * 255, dtype=np.uint8)
	return im


def pil2opencv(pil_img):
    '''
        convert PIL image into opencv for better visualization
    '''
    return np.transpose(pil_img, (1,2,0))


def visualize_triplets(batch):
    '''
        visualize the triplet data, especially image
        Input:
            batch comes from MSLS dataset dataloader
    '''
    # fetch data from single batch
    query, query_pc, positive_img, positive_pc, negatives_imgs, negatives_pcs, idxs = batch
    # batch number, query size should be torch.Size([batch_size, 3, 512, 1024])
    bs = query.shape[0]
    # total number of data(image size same as pc size) in a single batch
    # size should be BxNeg, batch_size sample * number of negatives
    N = len(idxs)
    for bsi in range(bs):
        # initialize plt
        plt.figure(figsize=(80, 60))
        # show query image
        plt.subplot(1, N, 1)
        plt.imshow(denormalize(pil2opencv(query[bsi])))
        plt.title("batch {} => query".format(bsi), fontsize=28)
        # show positive
        plt.subplot(1, N, 2)
        plt.imshow(denormalize(pil2opencv(positive_img[bsi])))
        plt.title("batch {} => positive".format(bsi), fontsize=28)
        # # show negatives
        for ni in range(negatives_imgs.shape[1]):
            plt.subplot(1, N, 3 + ni)
            plt.imshow(denormalize(pil2opencv(negatives_imgs[bsi][ni])))
            plt.title("batch {} => negatives {}".format(bsi, ni), fontsize=28)
        plt.show()

def tsne_visualization(path:str, name:str, feat):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    font = {"color": "black",
            "size": 12, 
            "family" : "serif"}
    #plt.style.use("dark_background")
    tsneData = TSNE(n_components=2, random_state=33).fit_transform(feat)
    N, C = feat.shape
    plt.figure(figsize=(10,10))
    plt.subplot(aspect='equal')
    plt.scatter(tsneData[:,0], tsneData[:,1], c=range(0,N),
            cmap=plt.cm.get_cmap('rainbow', N))
    plt.xlim(-50,50)
    plt.ylim(-50,50) 
    cbar = plt.colorbar() #ticks=range(N)
    cbar.set_label(label='color bar', fontdict=font)
    plt.title("Feature visualization of " + name + " based on t-SNE.", fontdict=font)
    plt.tight_layout()
    plt.savefig(path + "/" + name + ".png")
