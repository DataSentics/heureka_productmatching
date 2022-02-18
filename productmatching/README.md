# ProductMatching

Research to match products.

# Components

## NameSimilarity

### Collector

```
make collector
```

### Embedding dataset

```
make preprocessing_embedding_dataset
```

### PMI

```
make preprocessing_pmi
```

### Extract attributes

```
make preprocessing_extract_attributes
```

### Fasttext train

```
make preprocessing_fasttext_train
```

### Static dataset

```
make static_dataset
```
Static dataset is created from collector data. 
During the creation, we search for candidates for all the offers using elastic search and FAISS based on fasttext embeddings, that's why this stage is placed after the training of fasttext. When running in kube, the data is stored at a special location in S3 and the collector uri is replaced by uri of static dataset after this component finishes. When running locally, your collector data folder will be modified and renamed, so be cautios when running it.

## XGBoost ensemble

### Dataset

```
make xgboostmatching_dataset
```

### Train

```
make xgboostmatching_train
```

## Evaluation
```
make evaluation
```

## MatchAPI

### Serve

```
make matchapi_serve
```

### Single component in k9s

Copy `deployment.single.yaml.example` to `deployment.single.yaml` and modify as needed and then start with

```sh
kubectl --context production apply -f deployment.single.yaml
```

or stop with

```sh
kubectl --context production delete -f deployment.single.yaml
```

# How the **** is deployment of jobs from stage working? And how to make it work in new namespace?

Very weirdly, at least.

First we need to make a MR to the manifests with content of `Workflow/yamls/production.yaml` (e.g. https://gitlab.heu.cz/kubernetes/manifests/-/merge_requests/155/diffs).

Then, we need to steal its content.

List secrets with `kubectl --namespace catalogue-mlflow --context production get secrets`.
And steal with `kubectl --namespace catalogue-mlflow --context production get secret ml-workflow-token-vwf7g -o yaml > secret.production.yaml`.
Copy value from data `ca.crt` and `token`. It is encoded with base64, so decode it somehow.

Open `Workflow/yamls/stage.raw.yaml.example`.
`certificate-authority-data: fill_me` should have content of `ca.crt`. But only sections between first `---BEGIN---` and last `---END---`. And it have to be encoded in base64.
`token: fill_me` should have content of `token`, but NOT encoded! Use the decoded version.

Now, encode whole `stage.raw.yaml.example` into base64.

Now, open `Workflow/yamls/stage.yaml.example` and fill encoded `stage.raw.yaml.example` to the `config:` variable.

Now, deploy it to the stage and change name of the secret in `Workflow/charts/templates/deployment.yaml` in `volumes` if needed.

Voilá, it should work now.

Also, do not forget to have gitlab token for helm 3 deployments (e.g. https://gitlab.heu.cz/kubernetes/manifests/-/merge_requests/154/diffs).

And do not forget to deploy `Workflow/yamls/secrets.yaml` to both stage and production, with correct S3 credentials (pay attention to thing like added `\n` at the end of line with password, etc.).
