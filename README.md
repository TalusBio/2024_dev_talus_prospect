
# Peptide fragmentation models deployable via ONNX

This repository implements the training/inference logic for the peptide fragmentation models.
The initial training is done using Prospect (Kuster lab) and the main idea is to have a portable
model that can be deployed in other applications.

The idea here is not to have the most accurate model possible but one that is very fast and easy
to deploy (all pre-processing logic should be transferable easily between languages).

## Project organization

The project is organized as a monorepo with the following packages:

- `python_packages`:
  - `elfragmentador_core`: The core logic for the model encoding+decoding.
  - `elfragmentador_train`: The training logic for the model.
    - Depends on `elfragmentador_core`.
  - `elfragmentadonnx`: Reference implementation of the onnx inference.
    - Does not depend on torch or any of the very heavy training dependencies (still depends on numpy though).


## Working with the project

A lot of the logic on how to manage the project is in the `Taskfile.yml` (https://taskfile.dev/) file.

```
task fmt # Will format the code of all the packages
task lint # Will lint the code of all the packages
task test # Will run the tests of all the packages
task build # Will build all the packages
task publish # Will build all the packages and publish them to the S3 bucket
```

## Publishing

Right now wheels are built and pushed to s3, I would like some more stability before I decide to
have anythinng in PyPI.

## Future work

Right now the project is in a very early stage, so the general direction is not totally clear.
If someone finds it useful or has an use case for it, please let me know and within reasonable
limits I will try to make it work for you.

Here are some ideas:

- Implement rust crate for the model + logic
- Add first class support for finetuning the model.
- ... a lot of documentation ...

## Contributing

Contributions are welcome, please open an issue and we can discuss it there.
Valuable form of contributions:
- Adding more tests
- Adding more documentation
- Adding more examples
- Requests for documentation:
  - As an user you mentioning that one aspect is not clear or documented is
    very valuable.
- Re-raising issues that hae not been solved yet.