##
##
##

from dataset import RefCocoDataset, RefCocoConfig

from torch.utils.data import DataLoader

from skimage import io, draw
import matplotlib.pyplot as plt


def main():
    config = RefCocoConfig.default("data/refcocog")
    dataset = RefCocoDataset(config, "train")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=RefCocoDataset.batchify,
    )

    batch = next(iter(loader))

    img = batch.images[3].permute(1, 2, 0).numpy()
    sentence = batch.sentences[3]
    bbox = batch.bboxes[3].numpy()

    rr, cc = draw.rectangle_perimeter(
        start=(bbox[1], bbox[0]),
        extent=(bbox[3], bbox[2]),
        shape=img.shape,
        clip=True,
    )
    img[rr, cc] = 0

    print(sentence)

    io.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
