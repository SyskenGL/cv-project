import logging
from cv.core.dexpression import DeXpression
from cv.dataset.loader import CKPLoader, MMILoader


if __name__ == "__main__":

    logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)

    choice = None
    while choice not in ["0", "1", "2"]:
        choice = input("\n \u2022 Dataset [0: MMI | 1: CK+ | 2: CK+48]: ")

    loaders = [MMILoader(), CKPLoader(), CKPLoader(version48=True)]
    loader = loaders[int(choice)]
    loader.load()

    model = DeXpression()
    stats = model.cross_validate(loader.dataset, splits=3, epochs=1, output=True)

    print(stats)
