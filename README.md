# LSH's Bucketed Random Projection Model Implementation

# Table of content

- [LSH's Bucketed Random Projection Model Implementation](#%6C%73%68%27%73%2D%62%75%63%6B%65%74%65%64%2D%72%61%6E%64%6F%6D%2D%70%72%6F%6A%65%63%74%69%6F%6E%2D%6D%6F%64%65%6C%2D%69%6D%70%6C%65%6D%65%6E%74%61%74%69%6F%6E)
- [Table of content](#%74%61%62%6C%65%2D%6F%66%2D%63%6F%6E%74%65%6E%74)
  - [Usage guide](#%75%73%61%67%65%2D%67%75%69%64%65)
    - [Init environment](#%69%6E%69%74%2D%65%6E%76%69%72%6F%6E%6D%65%6E%74)
    - [Generate a random dataset](#%67%65%6E%65%72%61%74%65%2D%61%2D%72%61%6E%64%6F%6D%2D%64%61%74%61%73%65%74)
    - [Initiliaze the index](#%69%6E%69%74%69%6C%69%61%7A%65%2D%74%68%65%2D%69%6E%64%65%78)
    - [Querying index](#%71%75%65%72%79%69%6E%67%2D%69%6E%64%65%78)

## Usage guide

### Init environment

1. (Optional) Initalize a virtual environment:
```sh
py -m venv venv
```
2. (Optional) Activate the venv:
```sh
# On Windows
.\venv\Scripts\activate

# On Unix system
source venv/bin/activate
```
3. Install dependencies:
```sh
pip install -r requirements.txt -r requirements-dev.txt 
```

### Generate a random dataset

Once you are in your virtual env (if created) with the dependencies installed, run the following command:

```sh
python -m brp.generate_dataset <dataset_size> <data_dimension>
```

where:
- `<dataset_size>` is the length of the dataset (ex: 1000);
- `<data_dimension>` is the length of the data embedding (ex: 50).

:pencil: Notice that each entry is created with an identifier like "AAA", "AAB", ... "ZZZ" according to the dataset size.  
You can know the size of the identifier's string with the following formula: $\Big\lfloor\log_{26}(n)\Big\lfloor$ where $n$ is the dataset size.

### Initiliaze the index

Once you are in your virtual env (if created) with the dependencies installed, run the following command:

```sh
python -m brp.init_index <num_hyperplanes> <bucket_size>
```

where:
- `<num_hyperplanes>` is the number of hyperplanes you want to generate (ex: 4);
- `<bucket_size>` is the bucket length (ex: 1.0).

### Querying index

Once you are in your virtual env (if created) with the dependencies installed, run the following command:

```sh
python -m brp.query <bucket_size> <k> <identifier>
```

where:

- `<bucket_size>` is the bucket length specified when initialized the index (ex: 1.0);
- `<k>` is the number of the k nearest neighbors to retrieve (ex: 5);
- `<identifier>` is the data identifier to compare (ex: "ACE").