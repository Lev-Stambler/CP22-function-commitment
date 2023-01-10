# Numpy Functional Commitments

## Getting setup with the repository
The below should only be done once:

First, initialize the space with venv: ```
python3 -m venv venv
```

## Getting started with development
Initialize the workspace with `venv`:
run
```
source venv/bin/activate
```
in your terminal.

## Style guide
As you can see classes are everywhere. But, you may ask, isn't Lev (one of the authors
of this library) a huge proponent
of functional style programming? Yes, yes, Lev is.
Classes are syntactic sugar which make Python programming and typing easier to use.
Thus, **classes should have only immutable class variables**. This will make everything
much less of a headache. Pretty please follow these style guidelines.

## Running tests
This repo uses `pytest`. To run all tests just run
```
pytest
```

To run a specific file, run 
```
pytest <FILE_PATH>
```