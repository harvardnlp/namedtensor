import os
import sys
import time
import numpy as np
import pickle
import gensim
import data_helpers


def customize_embeddings_from_pretrained_googlenews_w2v(
    pretrained_embedding_fpath
):
    x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {
        rank: word for rank, word in enumerate(vocabulary_inv_list)
    }
    embedding_dim = 300

    directory = "./models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fpath_pretrained_extracted = os.path.expanduser(
        "{}/googlenews_extracted-python{}.pl".format(
            directory, sys.version_info.major
        )
    )
    fpath_word_list = os.path.expanduser("{}/words.dat".format(directory))

    tic = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format(
        pretrained_embedding_fpath, binary=True
    )
    print(
        "Please wait ... (it could take a while to load the file : {})".format(
            pretrained_embedding_fpath
        )
    )
    print("Done.  (time used: {:.1f}s)\n".format(time.time() - tic))

    embedding_weights = {}

    found_cnt = 0
    words = []
    for id, word in vocabulary_inv.items():
        words.append(word)
        if word in model.vocab:
            embedding_weights[id] = model.word_vec(word)
            found_cnt += 1
        else:
            embedding_weights[id] = np.random.uniform(
                -0.25, 0.25, embedding_dim
            )
    with open(fpath_pretrained_extracted, "wb") as f:
        pickle.dump(embedding_weights, f)
    with open(fpath_word_list, "w") as f:
        f.write("\n".join(words))


def main():
    if len(sys.argv) == 1:
        path_to_googlenews_vectors = os.path.expanduser(
            "~/.keras/models/GoogleNews-vectors-negative300.bin"
        )
    else:
        path_to_googlenews_vectors = sys.argv[1]
        if not os.path.exists(path_to_googlenews_vectors):
            print(
                'Sorry, file "{}" does not exist'.format(
                    path_to_googlenews_vectors
                )
            )
            sys.exit()
    print(
        "Your path to the googlenews vector file is: ",
        path_to_googlenews_vectors,
    )
    customize_embeddings_from_pretrained_googlenews_w2v(
        path_to_googlenews_vectors
    )


if __name__ == "__main__":
    main()
