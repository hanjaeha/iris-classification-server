import datetime
from typing import Optional, Iterable

class Sample:
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
        if self.species is None:
            known_unknown = "UnknownSample"
        else:
            known_unknown = "KnownSample"
        if self.classification is None:
             classification = ""
        else:
             classification = f", classification={self.classification!r}" # format str을 뜻함
        return (
            f"{known_unknown}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f"{classification}"
            f")"
        )
    
    def classify(self, classification: str) -> None:
         self.classification = classification # 이렇게 저장만 시켜줄 것이기 때문에 윗줄에 return이 없다

    def matches(self) -> bool:
        return self.species == self.classification
    

class Hyperparameter:
    def __init__(self, k: int, training: "TrainingData") -> None: # Return이 없기 때문에, 그리고 밑에 k랑 training에 물결표시 떠서 인자 넣음
        self.k = k
        self.data: TrainingData = training
        self.quality: float

    def classify(self):
        """TODO: k-NN 알고리즘"""
        return

    def test(self) -> None:
        pass_count, fail_count = 0, 0
        for sample in self.data.testing:
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

    def load(self, raw_data_source: Iterable[dict[str, str]]):
        for n, row in enumerate(raw_data_source):
            sample = Sample(
                sepal_length=float(row["sepal_length"]),
                sepal_width=float(row["sepal_width"]),
                petal_length=float(row["petal_length"]),
                petal_width=float(row["petal_width"]),
                species=row["species"],
            )
            if n % 5 == 0:
                self.testing.append(sample)
            else:
                self.training.append(sample)
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc) # for 루프가 다 끝나면 업로드 시간을 저장해준다.
        
    def test(self, parameter: Hyperparameter) -> None:
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        classification = parameter.classify(sample)
        sample.classify(classification)
        return sample
    
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
