# Contributing

Install the tested CPU environment from `requirements-lock.txt`, then install this
package with `python -m pip install -e . --no-deps`. Run `python -m pytest` and
`python -m ruff check src tests scripts` before proposing changes.

Keep changes focused and include tests for mathematical or protocol changes.
Never replace a missing dataset with generated data in a research run. Synthetic
fixtures belong only in software tests and must be described that way.

An experiment contribution must include its resolved configuration, dataset
provenance, seed, environment, run manifest, artifact checksums, and validation
output. Report failed and incomplete runs. Do not tune against test labels.
For a proposed scientific conclusion, show independently trained runs and the
paired statistical comparison that supports it. Repeated stochastic predictions
from one checkpoint are not independent training runs.

Do not commit datasets, credentials, environments, or large checkpoints. Agree
on an artifact location before uploading large outputs. Dataset and dependency
licenses remain separate from this project's MIT code license.

The repository has no declared publication venue, institutional affiliation,
or DOI. Do not add these without verification. `CITATION.cff` uses the provisional
collective attribution "NeuralShift Lab contributors"; the owner must confirm
the author list and add the real repository URL before publication. Public
publication and external service deployment require the project owner's
authorization.
