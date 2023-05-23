import math
import datetime
import weakref
import collections
from typing import Optional, Iterable, Union, Counter, cast


class InvalidSampleError(ValueError):
    """"""


class Sample: # 샘플이라는 class는 데이타셋을 학습 테스트, 데이터를 저장하는 클래스다.
    """Abstarct superclass for all sample classes(모든 샘플 클래스를 위한 추상클래스)"""
    def __init__( # 파이썬에선 생성자 __init__으로 초기화를 시킨다
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        species: Optional[str] = None # 기계학습에선 정답이 없으므로 옵셔널이라 쓰고 없으면 None
        # ctrl 누르고 Optional 누르면 vscode의 리팩토리 기능을 쓸 수 있다
    ) -> None: # Return 할 것이 아무것도 없다
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species
        self.classification: Optional[str] = None

    def __repr__(self) -> str: # representation (sample1, 2, 3, ... , 100000 이렇게 다 하긴 힘들어서 만듬)
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f")"
        )
    
    def classify(self, classification: str) -> None:
         self.classification = classification # 이렇게 저장만 시켜줄 것이기 때문에 윗줄에 return이 없다

    def matches(self) -> bool:
        return self.species == self.classification
    

class KnownSample(Sample): # 이 Sample은 위에 만들어놨던 class Sample
    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
        self.species = species

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f")"
        )
    
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "KnownSample": # 클래스 그 자체이기 때문에 self 말고 cls를 넣음
        if row["species"] not in {"Iris-setosa", "Iris-versicolour", "Iris-virginica"}:
            raise InvalidSampleError(f"invalid species in {row!r}") # 에러 방지됨
        try:
            return cls(
                species=row["species"],
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"])
            )
        except ValueError as e:
            raise InvalidSampleError(f"invalid {row!r}")


class UnknownSample(Sample):
    """Sample provided by an user, not yet classified."""
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "UnknownSample":
        if set(row.keys()) != {"sepal_length", "sepal_width", "petal_length", "petal_width"
        }:
            raise InvalidSampleError(f"invalid fields in {row!r}")
        try:
            return cls(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"])
            )
        except (ValueError, KeyError) as e:
            raise InvalidSampleError(f"invalid {row!r}")        


class TrainingKnownSample(KnownSample): # (KnownSample)을 상속받음
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TrainingKnownSample":
        return cast(TrainingKnownSample, super().from_dict(row)) # super는 부모클래스, 즉 KnownSample


class TestingKnownSample(KnownSample):
    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        classification: Optional[str] = None
    ) -> None:
        super().__init__(
            species=species,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f"classification={self.classification!r}"
            f")"
        )

    def matches(self) -> bool:
        return self.species == self.classification
    
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "TestingKnownSample":
        return cast(TestingKnownSample, super().from_dict(row)) # 리턴되는 것을 보여주려고 계속 써주는 것(인풋과 리턴값을 알려줌)

class ClassifiedSample(Sample):
    """for user(유저가 보는 창)"""
    def __init__(self, classification: str, sample: UnknownSample) -> None: # 항상 None으로 리턴합니다.
        super().__init__(
            sepal_length=sample.sepal_length,
            sepal_width=sample.sepal_width,
            petal_length=sample.petal_length,
            petal_width=sample.petal_width
        )
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"classification={self.classification!r}"
            f")"
        )


class Distance:
    def distance(self, s1: Sample, s2: Sample) -> float:
        pass


class ED(Distance): # 유클리드 거리
    def distance(self, s1: Sample, s2: Sample) -> float: # 객체지향의 다향성, 오버리딩 오버라이딩
        return math.hypot( # math 안에 있는 hypot 모듈을 가져다 쓴다)
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width
        )


class MD(Distance): # 맨하탄 거리
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum([
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width),
            abs(s1.petal_length - s2.petal_length),
            abs(s1.petal_width - s2.petal_width)
        ])

# 나머지 거리들은 여러분들이 직접 해서 업로드해주세요.
class CD:
    """TODO"""
    pass


class SD:
    pass


class Hyperparameter:
    def __init__(self, k: int, algorithm : Distance, training: "TrainingData") -> None: # Return이 없기 때문에, 그리고 밑에 k랑 training에 물결표시 떠서 인자 넣음
        self.k = k
        self.algorithm = algorithm
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def classify(self, sample: Union[UnknownSample, TestingKnownSample]) -> str: # 꽃 종류를 이야기할 것 이므로 str
        """K-NN algorithm"""
        training_data = self.data() # 그냥 가지고 오는게 아니라 메소드로 처리
        if not training_data:
            raise RecursionError("No TrainingData object!")
        distances: list[tuple[float, TrainingKnownSample]] = \
            sorted(
                (self.algorithm.distance(sample, known), known)
                for known in training_data.training
            )
        k_nearest: tuple[str] = (known.species for _, known in distances[:self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common() # ("a", 5)
        species, votes = best_fit
        return species

    def test(self) -> None:
        training_data: Optional["TrainingData"] = self.data
        if not training_data:
            raise RuntimeError("")
        pass_count, fail_count = 0, 0
        for sample in self.training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)


class TrainingData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime # 왜 이렇게 하냐? -> 우리는 이걸 가지고 서버를 만들 것이기 때문에
        self.training: list[Sample] = [] # 리스트 안에 샘플이 계속 쌓일 것이다 밑에도 똑같다 그래서 미리 리스트를 선언함
        self.testing: list[Sample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_source: Iterable[dict[str, str]]) -> None:
        for n, row in enumerate(raw_data_source):
            if n % 5 == 0:
                test = TestingKnownSample(
                    species=row["species"],
                    sepal_length=float(row["sepal_length"]),
                    sepal_width=float(row["sepal_width"]),
                    petal_length=float(row["petal_length"]),
                    petal_width=float(row["petal_width"]),
                )
                self.testing.append(test)
            else:
                train = TrainingKnownSample(
                    species=row["species"],
                    sepal_length=float(row["sepal_length"]),
                    sepal_width=float(row["sepal_width"]),
                    petal_length=float(row["petal_length"]),
                    petal_width=float(row["petal_width"]),
                )                
                self.training.append(train)
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc) # for 루프가 다 끝나면 업로드 시간을 저장해준다.
        
    def test(self, parameter: Hyperparameter) -> None:
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: UnknownSample) -> ClassifiedSample:
        return ClassifiedSample(
            classification=parameter.classify(sample), sample=sample
        )
    
test_sameple = """
>>> x = Sample(1.0, 2.0, 3.0, 4.0)
>>> x
UnknownSample(sepal_length=1.0, sepal_width=2.0, petal_length=3.0, petal_width=4.0, species=None) # 시험문제가 이렇게 나올 것이다.
"""

__test__ = {
    name: case for name, case in globals().items() if name.startswith("test_")
}
# class(생성자) 뒤에 줄바꿈 2번 (안하면 감점)

# if __name__ == "__main__":
#     sample = Sample(2.0, 2.0, 20.2, 30.1, "Virginica")
#     print(sample.classify("Sentosa"))
#     print(sample.matches())
# 실행하기 = 터미널 : ctrl+~ 아니면 우상단 삼각형 눌러서 최신버젼
