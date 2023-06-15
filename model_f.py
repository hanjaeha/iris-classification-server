from __future__ import annotations
import collections
from dataclasses import dataclass, asdict
from typing import (
    Optional,
    Counter,
    List,
)
import weakref
import sys


@dataclass(frozen=True)
class Sample:
    """Abstract superclass for all samples."""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@dataclass(frozen=True)
class KnownSample(Sample):
    """Represents a sample of testing or training data, the species is set once
    The purpose determines if it can or cannot be classified.
    """
    species: str


@dataclass
class TestingKnownSample(KnownSample):
    sample: KnownSample
    classification: Optional[str] = None


@dataclass(frozen=True)
class TrainingKnownSample:
    """Cannot be classified -- there's no classification instance variable available."""

    sample: KnownSample

@dataclass


class UnknownSample(Sample):
    """A sample provided by a User, to be classified."""
    sample: Sample
    classification: Optional[str] = None


class Distance:
    """A distance computation"""

    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError


@dataclass
class Hyperparameter:
    """A hyperparameter value and the overall quality of the classification."""

    k: int
    algorithm: Distance
    data: weakref.ReferenceType["TrainingData"]

    def classify(self, sample: Sample) -> str:
        """The K-NN algorithm"""
        if not (training_data := self.data()):
            raise RuntimeError("No TrainingData object")
        distances: list[tuple[float, TrainingKnownSample]] = sorted(
            (self.algorithm.distance(sample, Known), Known)
            for Known in training_data.training
        )
        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species


@dataclass
class TrainingData:
    """A set of training data and testing data with methods to load and test the samples."""
    testing: List[TestingKnownSample]
    training: List[TrainingKnownSample]
    tuning: List[Hyperparameter]


# Special case, we don't *often* test abstract superclasses.
# In this example, however, we can create instances of the abstract class.
test_Sample = """
>>> x = Sample(1, 2, 3, 4)
>>> x
Sample(sepal_length=1, sepal_width=2, petal_length=3, petal_width=4)
"""

test_Training_KnownSample = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> s1
KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, purpose=2, species='Iris-setosa')
>>> s1.classification
Traceback (most recent call last):
...
AttributeError: Training samples have no classification
>>> s1.classification("rejected")
Traceback (most recent call last):
...
AttributeError: Training samples have no classification
"""

test_Testing_KnownSample = """
>>> s2 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Testing.value)
>>> s2
KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, purpose=1, species='Iris-setosa')
>>> s2.classification
>>> s2.classification = "wrong"
>>> s2
KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, purpose=1, species='Iris-setosa', classification='wrong')
"""

test_UnknownSample = """
>>> u = UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, )
>>> u
UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, classification=None)
>>> u.classification = "something"
>>> u
UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, classification='something')
"""

test_Hyperparameter = """
>>> td = TrainingData('test')
>>> s2 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Testing.value)
>>> td.testing = [s2]
>>> t1 = KnownSample(**{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa", "purpose": 1})
>>> t2 = KnownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "species": "Iris-versicolor", "purpose": 1})
>>> td.training = [t1, t2]
>>> h = Hyperparameter(k=3, algorithm=Chebyshev(), training=td)
>>> u = UnknownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
>>> h.classify(u)
'Iris-setosa'
>>> h.test()
>>> print(f"data={td.name!r}, k={h.k}, quality={h.quality}")
data='test', k=3, quality=1.0
"""

test_TrainingData = """
>>> td = TrainingData('test')
>>> raw_data = [
... {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa"},
... {"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "species": "Iris-versicolor"},
... ]
>>> td.load(raw_data)
>>> h = Hyperparameter(k=3, algorithm=Chebyshev(), training=td)
>>> len(td.training)
1
>>> len(td.testing)
1
>>> td.test(h)
>>> print(f"data={td.name!r}, k={h.k}, quality={h.quality}")
data='test', k=3, quality=0.0
"""


__test__ = {name: case for name, case in globals().items() if name.startswith("test_")}