# 기말시험범위 6장까지, 6월 13일 시험, 이론같은 경우도 다형성 얘기하면서 오버라이딩 오버리딩 했음
# 과제는 7장까지 여러분이 공부해서 깃헙에 올려주기 -> 기말고사 전 까지
# 각 챕터의 마지막 부분에 use-case가 있는데 이걸 구현하고 실행이 되어야함 (test-case가 통과되어야함) -> 교수님이 돌렸을 때 실행이 되어야함 (테스트케이스)
# Github 주소를 과제에 올리기
# 객체지향프로그래밍을 왜하는건지, 어떤 언어가 있고, 파이썬에 대해 배웠는데 파이썬에 대한 advanced한 내용을 배웠음 이 두가지만 기억을 하시오
# 3요소 5원칙 무조건 시험에 나온다
# 시험은 사지선다 객관식 주관식 손코딩이 나올 것, 문제는 20문제정도 나올 것, 어려운 문제 거의 안나오고 다 책에서 나올 것
# eclass에 ppt 3개 올라간거 이것만 꼭 보면 된다
from __future__ import annotations
import collections
import abc
import csv
import collections.abc
import datetime
import enum
import random
from math import isclose
from pathlib import Path
from typing import (
    cast,
    Any,
    Optional,
    Union,
    Iterator,
    Iterable,
    Counter,
    Callable,
    Protocol,
    TypedDict,
    List,
    overload,
    Tuple
)
import weakref
from model import Sample

class Sample:
    """Abstract superclass for all samples."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def __eq__(self, other: Any) -> bool:
        if type(other) != type(self):
            return False
        other = cast(Sample, other)
        return all(
            [
                self.sepal_length == other.sepal_length,
                self.sepal_width == other.sepal_width,
                self.petal_length == other.petal_length,
                self.petal_width == other.petal_width,
            ]
        )

    @property
    def attr_dict(self) -> dict[str, str]:
        return dict(
            sepal_length=f"{self.sepal_length!r}",
            sepal_width=f"{self.sepal_width!r}",
            petal_length=f"{self.petal_length!r}",
            petal_width=f"{self.petal_width!r}",
        )

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"


class Purpose(enum.IntEnum):
    Classification = 0
    Testing = 1
    Training = 2


class KnownSample(Sample):
    """Represents a sample of testing or training data, the species is set once
    The purpose determines if it can or cannot be classified.
    """

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        purpose: int,
        species: str,
    ) -> None:
        purpose_enum = Purpose(purpose)
        if purpose_enum not in {Purpose.Training, Purpose.Testing}:
            raise ValueError(f"Invalid purpose: {purpose!r}: {purpose_enum}")
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.purpose = purpose_enum
        self.species = species
        self._classification: Optional[str] = None

    def matches(self) -> bool:
        return self.species == self.classification

    @property
    def classification(self) -> Optional[str]:
        if self.purpose == Purpose.Testing:
            return self._classification
        else:
            raise AttributeError(f"Training samples have no classification")

    @classification.setter
    def classification(self, value: str) -> None:
        if self.purpose == Purpose.Testing:
            self._classification = value
        else:
            raise AttributeError(f"Training samples cannot be classified")

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["purpose"] = f"{self.purpose.value}"
        base_attributes["species"] = f"{self.species!r}"
        if self.purpose == Purpose.Testing and self._classification:
            base_attributes["classification"] = f"{self._classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"


class UnknownSample(Sample):
    """A sample provided by a User, to be classified."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self._classification: Optional[str] = None

    @property
    def classification(self) -> Optional[str]:
        return self._classification

    @classification.setter
    def classification(self, value: str) -> None:
        self._classification = value

    def __repr__(self) -> str:
        base_attributes = self.attr_dict
        base_attributes["classification"] = f"{self.classification!r}"
        attrs = ", ".join(f"{k}={v}" for k, v in base_attributes.items())
        return f"{self.__class__.__name__}({attrs})"


class Distance:
    """A distance computation"""

    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError


class Chebyshev(Distance):
    """
    Computes the Chebyshev distance between two samples.

    ::

        >>> from math import isclose
        >>> from model import KnownSample, Purpose, UnknownSample, Chebyshev

        >>> s1 = KnownSample(
        ...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa",
        ...     purpose=Purpose.Training)
        >>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

        >>> algorithm = Chebyshev()
        >>> isclose(3.3, algorithm.distance(s1, u))
        True

    """

    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )


class Minkowski(Distance):
    """An abstraction to provide a way to implement Manhattan and Euclidean."""

    m: int

    def distance(self, s1: Sample, s2: Sample) -> float:
        return (
            sum(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            )
            ** (1 / self.m)
        )


class Euclidean(Minkowski):
    m = 2


class Manhattan(Minkowski):
    m = 1


class Sorensen(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum(
            [
                s1.sepal_length + s2.sepal_length,
                s1.sepal_width + s2.sepal_width,
                s1.petal_length + s2.petal_length,
                s1.petal_width + s2.petal_width,
            ]
        )


class Reduce_Function(Protocol):
    """Define a callable object with specific parameters."""

    def __call__(self, values: list[float]) -> float:
        pass


class Minkowski_2(Distance):
    """A generic way to implement Manhattan, Euclidean, and Chebyshev.

    ::

        >>> from math import isclose
        >>> from model import KnownSample, Purpose, UnknownSample, Minkowski_2


        >>> class CD(Minkowski_2):
        ...     m = 1
        ...     reduction = max

        >>> s1 = KnownSample(
        ...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa",
        ...     purpose=Purpose.Training)
        >>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

        >>> algorithm = CD()
        >>> isclose(3.3, algorithm.distance(s1, u))
        True

    """

    m: int
    reduction: Reduce_Function

    def distance(self, s1: Sample, s2: Sample) -> float:
        # Required to prevent Python from passing `self` as the first argument.
        summarize = self.reduction
        return (
            summarize(
                [
                    abs(s1.sepal_length - s2.sepal_length) ** self.m,
                    abs(s1.sepal_width - s2.sepal_width) ** self.m,
                    abs(s1.petal_length - s2.petal_length) ** self.m,
                    abs(s1.petal_width - s2.petal_width) ** self.m,
                ]
            )
            ** (1 / self.m)
        )


class Hyperparameter:
    """A hyperparameter value and the overall quality of the classification."""

    def __init__(self, k: int, algorithm: "Distance", training: "TrainingData") -> None:
        self.k = k
        self.algorithm = algorithm
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """Run the entire test suite."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError("Broken Weak Reference")
        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Sample) -> str:
        """The k-NN algorithm"""
        training_data = self.data()
        if not training_data:
            raise RuntimeError("No TrainingData object")
        distances: list[tuple[float, KnownSample]] = sorted(
            (self.algorithm.distance(sample, known), known)
            for known in training_data.training
        )
        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species


class TrainingData:
    """A set of training data and testing data with methods to load and test the samples."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[KnownSample] = []
        self.testing: list[KnownSample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_iter: Iterable[dict[str, str]]) -> None:
        """Extract TestingKnownSample and TrainingKnownSample from raw data"""
        for n, row in enumerate(raw_data_iter):
            purpose = Purpose.Testing if n % 5 == 0 else Purpose.Training
            sample = KnownSample(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
                purpose=purpose,
                species=row["species"],
            )
            if sample.purpose == Purpose.Testing:
                self.testing.append(sample)
            else:
                self.training.append(sample)
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:
        """Test this hyperparamater value."""
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: UnknownSample) -> str:
        return parameter.classify(sample)


class BadSampleRow(ValueError):
    pass


class SampleReader:
    """See iris.names for attribute ordering in bezdekIris.data file"""

    target_class = Sample
    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterator[Sample]:
        target_class = self.target_class
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = target_class(
                        sepal_length=float(row["sepal_length"]),
                        sepal_width=float(row["sepal_width"]),
                        petal_length=float(row["petal_length"]),
                        petal_width=float(row["petal_width"]),
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invalid {row!r}") from ex
                yield sample

class TestingKnownSample:
    ...

class TrainingKnownSample:
    ...

class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


class SamplePartition(List[SampleDict], abc.ABC): # 추상클래스 선언, 다중상속을 받음(리스트샘플딕트와 abc)
    @overload
    def __init__(self, *, training_subset: float = 0.80) -> None: # * 은 파티션 역할을 함
        ...

    @overload
    def __init__(
            self,
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
    ) -> None:
        ...

    def __init__(
            self,
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
    ) -> None:
        self.training_subset = training_subset
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__()

    @abc.abstractproperty
    @property
    def training(self) -> list[TrainingKnownSample]:
        ...

    @abc.abstractproperty
    @property
    def testing(self) -> list[TestingKnownSample]:
        ...


class ShufflingSamplePartition(SamplePartition):
    def __init__(
        self,
        iterable: Optional[Iterable[SampleDict]] = None,
        *,
        training_subset: float = 0.80 # 80%만 가져갈것이다
    ) -> None:
        super().__init__(iterable, training_subset)
        self.split: Optional[int] = None

    @property
    def training(self) -> List[TrainingKnownSample]:
        self.shuffle()
        return [TrainingKnownSample(**sd) for sd in self[:self.split]]
    
    @property
    def testing(self) -> List[TestingKnownSample]: # 테스트 할 때는 셔플을 하지 않는다(머신러닝적으로), 책과는 다름
        return [TestingKnownSample(**sd) for sd in self[self.split:]]

    def shuffle(self) -> None:
        if not self.split:
            random.shuffle(self)
            self.split = int(len(self) * self.training_subset)

class DealingPartition(abc.ABC):
    def __init__(
        self,
        items: Optional[Iterable[SampleDict]],
        *, # /는 왼쪽 *는 오른쪽 파티션을 의미함
        training_subset: Tuple[int, int] = (8, 10)
    ) -> None:
        ...

    @abc.abstractproperty
    @property
    def training(self) -> List[TrainingKnownSample]:
        ...
    
    @abc.abstractproperty
    @property
    def testing(self) -> List[TestingKnownSample]:
        ...

    @abc.abstractmethod
    def extend(self, items: Iterable[SampleDict]) -> None: # 리턴은 트레이닝, 테스팅이 해줄것이기 때문에 리턴은 None, 추상클래스는 구현을 해 놓지 않는다
        ...
    
    @abc.abstractclassmethod
    def append(self, item: SampleDict) -> None:
        ...

class CountingDealingPartition(DealingPartition):
    def __init__(
        self,
        items: Optional[Iterable[SampleDict]],
        *,
        training_subset: Tuple[int, int] = (8, 10)
    ) -> None:
        self.training_subset = training_subset
        self.counter = 0
        self._training: list[TrainingKnownSample] = []
        self._testing: list[TestingKnownSample] = []
        if items:
            self.extend(items)

    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)

    def append(self, item: SampleDict) -> None:
        n, d = self.training_subset # 지금 n은 8, d는 10
        if self.counter % d < n:
            self._training.append(TrainingKnownSample(**item))
        else:
            self._testing.append(TestingKnownSample(**item))
        self.counter += 1

    @property
    def training(self) -> list[TrainingKnownSample]:
        return self._training
    
    @property
    def testing(self) -> list[TestingKnownSample]:
        return self._testing        

# 시험
# src(소스)라는 폴더를 만들고 model.py를 src에 집어넣는다 그럼 root폴더 안에 src폴더가 있고 거기에 model이 들어가있다
# 그 다음 classifier.py라는 파일이 src폴더 안에 들어가있어야한다
# 다시 root폴더로 나와서 tests라는 폴더를 만든다
# tests폴더 안에는 test_ 라는 스크립트가 하나 있어야한다 (6장에서는 파티션, test_partition.py)
# 과제를 이런 식으로 제출해주셔야합니다
# 제출할 때 src, tests폴더가 있어야한다
# src안에는 model.py, classifier.py라는 파일이 있어야하고
# tests폴더 안에는 test_ 라는 파일들이 있어야한다
# 이는 교수님이 만든것이아닌 효과적인 방법으로 만들어진 구조체임
# 당연히 이 소스폴더는 아무 이름이나 하셔도 상관없다 하지만 우리는 src라고 얘기를 한다
# 하지만 tests 라는 폴더는 무조건 tests
# cache 생기는 것은 신경쓰지 않아도 된다
# 책에 있는 git을 보면 이미 구조가 이렇게 되어있음 파이썬 객체지향 프로그래밍 4/e 깃헙 가기 github.dev/AcornPublishing/python-oop-4e
# src 안에는 여러가지가 있는데 model과 classifier만
# tests폴더에는 partition만 있으면 되겠네요
# 7장까지 최신화된 코드를 이클래스과제제출에다가 링크만 남겨주면 된다.
# 복붙해서 붙여넣기만 하면 되겠지만 잘 돌아가지도 않을것임
# git diff만 쓰면 다 나오니 손수 하나하나 필사를 해서 제출하기
# 이해가 안가면 연구실로 오거나 카톡을 주기
# 완전체를 돌리라는 것 아님


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

test_Chebyshev = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Chebyshev()
>>> isclose(3.3, algorithm.distance(s1, u))
True
"""

test_Euclidean = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Euclidean()
>>> isclose(4.50111097, algorithm.distance(s1, u))
True
"""

test_Manhattan = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Manhattan()
>>> isclose(7.6, algorithm.distance(s1, u))
True
"""

test_Sorensen = """
>>> s1 = KnownSample(
...     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-setosa", purpose=Purpose.Training.value)
>>> u = UnknownSample(**{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4})

>>> algorithm = Sorensen()
>>> isclose(0.2773722627, algorithm.distance(s1, u))
True
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
