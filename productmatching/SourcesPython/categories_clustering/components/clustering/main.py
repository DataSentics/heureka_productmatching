import logging
import gensim
import mlflow
import os
import argparse
import random
import asyncio

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import typing as t
import plotly.express as px

from collections import Counter
from itertools import chain
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from utilities.notify import notify
from utilities.normalize import normalize_string
from utilities.component import process_input, compress
from utilities.normalize.stemmer import cz_stem
from preprocessing.models.transformer import TransformerModel

os.environ['TOKENIZERS_PARALLELISM'] = "false"


def preprocess(sentence: str) -> list:
    sentence = normalize_string(sentence)
    # stemming
    stemmed = [cz_stem(w) for w in sentence.split(" ")]
    # skip empty words, too short words or words containing only digits (mostly product types)
    stemmed = [w for w in stemmed if len(w) > 2 and sum(le.isalpha() for le in w) > 0]

    return stemmed


def get_characteristic_words(preprocessed_sentences: t.List[str], n_topics: int = 1, n_words_return: int = 10):

    # create corpus
    dictionary = gensim.corpora.Dictionary(preprocessed_sentences)
    # create bag of words
    bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_sentences]

    # build LDA model
    lda_model = gensim.models.LdaModel(
        bow_corpus,
        num_topics=n_topics,
        id2word=dictionary,
        passes=10,
        # workers=4,
        random_state=10
    )

    # get characteristic words
    characteristic_words = []
    for topic_id in range(n_topics):
        topic_words = [dictionary[term[0]] for term in lda_model.get_topic_terms(topic_id, n_words_return)]
        characteristic_words.append(topic_words)

    return characteristic_words


def produce_summary_file(output_dir, max_cluster, pdf_ctc, counts_pdfs):
    output_path = os.path.join(output_dir, "clustering_results_agg.txt")
    with open(output_path, "w") as f:
        f.write("This file serves as a quick overview of aggregated clustering results.\n")
        f.write("For each number of clusters we provide a summary table with following columns:\n")
        f.write("\t- cluster (index): id of a cluster\n")
        f.write("\t- n_products_in_cluster: number of products that fell into corresponding cluster and belong to categories in this cluster\n")
        f.write("\t- n_products: number of products that were used during clustering and belong to categories in this cluster\n")
        f.write("\t- n_categories: number of categories that were assigned to corresponding cluster\n")
        f.write("\t- perc_products_in_clusters: % of products that belong to categories in corresponding cluster and were assigned the cluster during clustering\n")
        f.write("\t- perc_products_total: % of total number of products that were used during clustering that belong to categories in corresponding cluster\n")
        for i in range(max_cluster):
            cc = f"{i}_clusters_cluster"
            if cc in pdf_ctc.columns:
                pdf_ctc_i = pdf_ctc[["category_slug", cc]].drop_duplicates()

                pdf_sizes = counts_pdfs[i][["category_slug", "cluster", "n_products_in_cluster", "n_products"]]

                pdf_merged = pd.merge(pdf_ctc_i, pdf_sizes, left_on=["category_slug", cc], right_on=["category_slug", "cluster"])
                total_products = sum(pdf_merged["n_products"])
                pdf_agg = (
                    pdf_merged
                    .groupby('cluster')
                    .agg({"n_products_in_cluster": "sum",  "n_products": "sum", "cluster": "count"})
                    .rename(columns={"cluster": "n_categories"})
                    .sort_values("n_categories")
                )
                pdf_agg["perc_products_in_clusters"] = (pdf_agg["n_products_in_cluster"] / pdf_agg["n_products"]).apply(lambda x: round(x, 2))
                pdf_agg["perc_products_total"] = (pdf_agg["n_products"] / total_products).apply(lambda x: round(x, 2))

                f.write("\n\n")
                f.write("#" * 120)
                f.write("\n\n")
                f.write(f"aggregated results for {i} clusters:")
                f.write("\n")
                f.write(str(pdf_agg))

    mlflow.log_artifact(output_path)


@notify
def clustering_cluster(args):
    asyncio.run(_main(args))


async def _main(args):
    random.seed(10)
    output_dir = os.path.join(args.data_directory, "clustering_results")
    os.makedirs(output_dir, exist_ok=True)

    # read products
    products = pd.read_csv(os.path.join(args.input_dataset, "clustering_data.csv"), index_col=False)
    logging.info(products.groupby("category_slug").size())

    # count of products per category
    n_products_before_removal = products.groupby("category_slug").size().reset_index(name="products_before_removal")

    products["name"] = products["name"].astype(str)
    category_ids = set(products["category_id"])

    # find caharacteristic words for each category
    characteristic_words = {}

    characteristic_names = pd.DataFrame()
    for category in category_ids:
        logging.info(f"Processing {category}")

        products_sel = products[products["category_id"] == category]
        names = products_sel["name"].to_numpy().astype(str)
        # preprocess names
        preprocessed_names = list(map(preprocess, names))

        # find characteristic words
        typical_words = get_characteristic_words(preprocessed_names, n_words_return=args.n_characteristic_words, n_topics=1)
        characteristic_words[category] = typical_words
        # unpack topics to one for this case
        typical_words = list(chain.from_iterable(typical_words))

        contains_characteristic_words = [len(set(preprocessed_name).intersection(typical_words)) for preprocessed_name in preprocessed_names]

        # set threshold for choosing names with most characteristic words
        thr = np.quantile(contains_characteristic_words, 0.75)
        # append list of items with calculated flag
        characteristic_names = characteristic_names.append(
            pd.DataFrame(
                {
                    "id": products_sel["id"],
                    "typical_name": contains_characteristic_words >= thr
                }
            ),
            ignore_index=True
        )

    # add this info to products df
    products = pd.merge(products, characteristic_names, on="id")

    # work only with products containing characteristic words
    products = products[products["typical_name"]].drop("typical_name", axis=1)

    logging.info(products.groupby("category_slug").size())

    # save and log info about product count per category before and after removal
    n_products_info_path = os.path.join(args.data_directory, "n_products_info.csv")

    n_products_after_removal = products.groupby("category_slug").size().reset_index(name="products_after_removal")
    n_products_info = pd.merge(n_products_before_removal, n_products_after_removal, on="category_slug")
    n_products_info.to_csv(n_products_info_path, index=False)
    mlflow.log_artifact(n_products_info_path)

    # save characteristic words
    characteristic_words_df = pd.DataFrame.from_dict(characteristic_words, orient="index").reset_index()
    characteristic_words_df.columns = ["category_id", "characteristic_words"]
    category_id_to_slug = products[["category_id", "category_slug"]].drop_duplicates()
    characteristic_words_df = pd.merge(characteristic_words_df, category_id_to_slug, on="category_id")
    characteristic_words_df[["category_slug", "category_id", "characteristic_words"]].to_csv(
        os.path.join(output_dir, "characteristic_words.csv"), index=False
    )

    # calculate number of products per category
    products_per_category = products.groupby("category_slug", as_index=False).size().sort_values(by="category_slug")
    products_per_category.columns = ["category_slug", "n_products"]

    # get name embeddings
    transformer_model = TransformerModel(args.input_transformer)

    logging.info("Starting creating name embeddings")
    features = await transformer_model.get_sentence_vector(products["name"].to_numpy(), show_progress_bar=True)
    logging.info("Ending creating name embeddings")
    if not args.use_only_namesimilarity:
        name_len = products["name"].apply(len)
        n_words = products["name"].apply(lambda x: len(x.split(" ")))
        n_digits = products["name"].apply(lambda x: sum(le.isdigit() for le in x))
        avg_price = products["prices"].apply(lambda x: np.mean(x))
        features = np.concatenate(
            [
                np.array([name_len, n_words, n_digits, avg_price]).T,
                features
            ],
            axis=1
        )
        scaler = StandardScaler()
        # scale because we have different types of variables, yes, it mights distort the embeddings a bit
        features = scaler.fit_transform(features)

    # trying to address the "curse of dimensionality" problem
    if args.n_components > 0:
        features = PCA(n_components=args.n_components, random_state=42).fit_transform(features)

    # run clustering
    # we will monitor number of caregories, where percetages of products in most common and second cluster
    # are above specified threshold
    n_diff_pct_above_thr = []

    # number of clusters with some category, it may happen we have some clusters containing a bit from each category
    n_populated_clusters = []

    # limit number of clusters by number of categories
    max_cluster = min(args.max_clusters, len(category_ids)) + 1
    min_cluster = min(args.min_clusters, max_cluster - 1)

    # get category cluster accross various number of clusters
    category_to_clusters = products[["category_slug", "category_id"]].drop_duplicates()

    counts_pdfs = {}

    for n_clusters in range(min_cluster, max_cluster):
        clusters = KMeans(n_clusters=n_clusters, verbose=0, tol=1e-6, algorithm="full", random_state=10)
        clusters = clusters.fit(features)
        # add cluster id to products
        labels = clusters.labels_
        products["cluster"] = labels
        sil_score = silhouette_score(features, labels)
        mlflow.log_metric("silhouette_score", sil_score, step=n_clusters)
        logging.info(f"{n_clusters} clusters: {Counter(products['cluster'])}")
        logging.info(f"Silhouette score: {sil_score}")
        print(f"{n_clusters} clusters: {Counter(products['cluster'])}")
        print(f"Silhouette score: {sil_score}")

        # assign categories to clusters and get more info, we have clustered products now

        output_excel_path = os.path.join(output_dir, f"results_{n_clusters}_product_clusters.xlsx")
        with pd.ExcelWriter(output_excel_path) as writer:
            # number of product per category and per cluster
            counts = products.groupby(["category_slug", "category_id", "cluster"], as_index=False).size()
            counts = pd.merge(counts, products_per_category, on="category_slug")
            # calculate percentage of products from given category in given cluster
            counts["products_in_cluster_pct"] = counts["size"] / counts["n_products"]
            # largest percentage per category
            most_pct = counts.groupby("category_slug", as_index=False).agg({"products_in_cluster_pct": max})
            most_pct.columns = ["category_slug", "max_pct"]
            # add this column
            counts = pd.merge(counts, most_pct, on="category_slug")
            # order alphabetically and within category, order by percentage decreasingly
            counts = counts.sort_values(by=["category_slug", "products_in_cluster_pct"], ascending=[True, False])
            # calculate the differences in percentage between first and second most common cluster within category
            differences = pd.DataFrame(
                {
                    "category_slug": counts["category_slug"],
                    "diff_pct": -1*counts.groupby("category_slug")["products_in_cluster_pct"].diff().fillna(0)
                }
            )
            # take only diff between first and second most common cluster, one if all products in one cluster
            difference = differences.groupby("category_slug").head(2)
            # omit the first cluster where diff is 0
            # if all products were in one cluster, we will omit the row
            difference = difference[difference["diff_pct"] > 0]
            # add diff_pct as new column to counts, 1 if no row in difference i.e. all products in one cluster
            counts = pd.merge(counts, difference, on="category_slug", how="left").fillna(1)
            # find caategories where the difference is above than given threshold
            counts[f"diff_pct_above_{args.diff_pct_thr}"] = counts["diff_pct"] > args.diff_pct_thr

            # assign categories to clusters
            # find row index with most populated cluster for each category
            idx_max = counts.groupby("category_slug")["size"].idxmax()
            # get cluster id, category name and flag about difference of percentages
            # some clusters may not be present, if it is not the most comon cluster for any category
            final_clusters = (
                counts.loc[idx_max][["cluster", "category_slug", "category_id", "diff_pct", f"diff_pct_above_{args.diff_pct_thr}"]]
                .sort_values(by=["cluster", "diff_pct"], ascending=[True, False]))

            counts.rename(columns={"size": "n_products_in_cluster"}, inplace=True)
            counts_pdfs[n_clusters] = counts

            # write results to excel
            counts.to_excel(writer, "category_to_cluster_info", index=False)
            final_clusters.to_excel(writer, "clusters_to_categories", index=False)

            # add info to overall category assignment
            final_clusters_flt = final_clusters[["category_slug", "category_id", "cluster", "diff_pct"]]
            final_clusters_flt.columns = ["category_slug", "category_id", f"{n_clusters}_clusters_cluster", f"{n_clusters}_clusters_diff_pct"]
            category_to_clusters = pd.merge(category_to_clusters, final_clusters_flt, on=['category_slug', 'category_id'])

        n_diff_pct_above_thr.append(sum(final_clusters[f"diff_pct_above_{args.diff_pct_thr}"]))
        n_populated_clusters.append(len(set(final_clusters["cluster"])))

        # random coordinates on a grid for vizualization
        x = np.array(random.sample(range(0, 20, 1), k=n_clusters))
        y = np.array(random.sample(range(0, 20, 1), k=n_clusters))
        z = np.array(random.sample(range(0, 20, 1), k=n_clusters))
        centers = np.vstack([x, y, z]).T
        # join with cluster id
        centers = np.concatenate([np.array(range(n_clusters)).reshape(-1, 1), centers], axis=1)
        centers = pd.DataFrame(centers, columns=["cluster", "x", "y", "z"])
        # append to all categories
        cluster_viz = pd.merge(final_clusters, centers, on="cluster")
        # add some random shift so they are scattered a bit
        cluster_viz.loc[:, ["x", "y", "z"]] = (
            cluster_viz.loc[:, ["x", "y", "z"]] +
            np.random.poisson(size=cluster_viz.loc[:, ["x", "y", "z"]].shape) +
            np.random.normal(scale=0.5, size=cluster_viz.loc[:, ["x", "y", "z"]].shape)
        )
        cluster_viz["cluster"] = cluster_viz["cluster"].astype(str)
        # create 3D scatterplot and save
        # TODO: make this usefull or delete it
        fig = px.scatter_3d(
            cluster_viz,
            x="x",
            y="y",
            z="z",
            color="cluster",
            text="category_slug",
            hover_name="category_slug",
            hover_data=["cluster", "diff_pct"],
            title="Only informational plot. Categories locations are arbitrary, focus on cluster colors and IDs. For more info, see other output files."
        )
        fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color, textposition='top center'))
        fig.write_html(os.path.join(output_dir, f"cluster_vizualization_{n_clusters}_clusters.html"))

    # write overall cluster assignment
    output_excel_path = os.path.join(output_dir, "categories_to_clusters.xlsx")
    cluster_columns = [c for c in category_to_clusters.columns if "clusters_cluster" in c]
    with pd.ExcelWriter(output_excel_path) as writer:
        category_to_clusters.sort_values(by=cluster_columns).to_excel(writer, "clusters_to_categories", index=False)

    produce_summary_file(output_dir, max_cluster, category_to_clusters, counts_pdfs)

    # draw plot with number nonempty category clusters, some may contain
    plt.figure()
    plt.subplot(311)
    plt.plot(range(min_cluster, max_cluster), n_populated_clusters)
    plt.xticks(range(min_cluster, max_cluster))
    plt.xlabel("Number of product clusters", fontsize=8)
    plt.ylabel("Number of nonempty \n category clusters", fontsize=8)

    # draw plot with number of categories above diff threshold
    plt.subplot(313)
    plt.plot(range(min_cluster, max_cluster), n_diff_pct_above_thr)
    plt.xticks(range(min_cluster, max_cluster))
    plt.xlabel("Number of product clusters", fontsize=8)
    plt.ylabel(f"Number of categories with difference \n between first two clusters above {args.diff_pct_thr}", fontsize=8)

    plt.savefig(os.path.join(output_dir, "clustering_diff"))

    tar_file = os.path.join(args.data_directory, "clustering_results.tar.gz")
    compress(tar_file, output_dir)

    mlflow.log_artifact(str(tar_file))


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), force=True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-directory", default="/data")
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--input-transformer", required=True)
    parser.add_argument("--use-only-namesimilarity", default=1)
    parser.add_argument("--min-clusters", default=2, type=int)
    parser.add_argument("--max-clusters", default=5, type=int)
    parser.add_argument("--n-characteristic-words", default=30, type=int)
    parser.add_argument("--diff-pct-thr", default=0.5, type=float)
    parser.add_argument("--n-components", default=-1, type=int)

    args = parser.parse_args()

    args.input_dataset = process_input(args.input_dataset, args.data_directory)

    with mlflow.start_run():
        clustering_cluster(args)
