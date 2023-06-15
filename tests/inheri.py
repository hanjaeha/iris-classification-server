##### 상속 #####
# import abc as abstraction # 라이브러리 불러들이기
from abc import ABC, abstractclassmethod


# 스타일상 두 줄 띄워준다.
#abc.ABC
class Player(ABC): # 관례로 Class만들 때는 앞에 대문자를 쓰고 띄어쓰기를 그냥 대소문자 차이로 정하자고 정함
        # 추상클래스 선언
    def __init__(self) -> None:
        self._name # _로 정보은닉
        self.number
        self.cloth
        self._age

    def eat(self): # 여기선 eat_apple 이런 식으로 띄운다. 이런 스타일 가이드는 구글 python pep8 참고
        pass

    def move(self): # moveLeft가 아니고 move_left 이런 식의 스타일을 사용한다. 우리가 만든 코드는 전부 github에 올려주자 객체지향을 하는 이유는 코드가 몇십만줄씩 되기 때문이다.
                    # 그리고 터미널을 잘 사용해야한다. (회사에서는 터미널을 사용한다.) 터미널에대한 간단한 명령어를 공부해놓자.
                    # 터미널 중에서도 파워셸 말고 리눅스 터미널을 사용해야한다 (우분투) 리눅스 터미널에 대한 명령어를 공부해야한다.
                    # 직장에 들어가게 되면 Docker라는 소프트웨어를 사용하기 때문이다.
        pass


class LeeJungJae(Player):
    pass

jungjae = LeeJungJae() # 인스턴스화
jungjae.move()


class OhIlNam(Player):
    pass

