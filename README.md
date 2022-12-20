## Long-form Question Answering Annotation

Using the Inception platform for long-form question answering annotation.

### Installation

```python
pip install -r requirements.txt
```

### Usage

1. To get the answer preference layer annotations, run the following command:

```python
python meta_annotation.py
```

2. To get the all other feature annotations, run the following command:

```python
python analysis.py
```

3. To get the labels for the answer preference layer annotations, you can find the 
function `collate_metadata` in `utils.py` and run it.


4. To get the final annotations, you can find the function `collate_annotations` in 
`utils.py` and run it.
