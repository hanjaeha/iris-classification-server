##### 다형성 #####

##### 오버라이딩 #####
from abc import ABC, abstractclassmethod

class Player(ABC):
#   @abstractclassmethod
#   def move(self, food):
#       pass

    @abstractclassmethod
    def eat(self):
        pass

class LeeJungJae(Player):
    def eat(self, food): # 이정재만 안먹는 특별한 경우, 오버라이딩을 해준다.
        return None

#   def move(self): # 각각 걷는 방식이 다르다.

class OhIlNam(Plyaer):
    def eat(self, food): # eat만 하지 말고 move까지 해줘야한다. 하지만 작동이 되므로 장치를 강제로 만들어야한다. @abstractclassmethod로 설정한다.
        hungry += food
        return hungry # 먹을 때 마다 hungry라는 수치가 늘어난다

jungjae = LeeJungJae()
jungjae.eat("apple")
ilnam = OhIlNam()
ilnam.eat("apple")

##### 오버 로딩 #####
from abc import ABC, abstractclassmethod
from typing import List, Dict

class Player(ABC):
    @abstractclassmethod
    def move(self, food: str):
        pass

    @abstractclassmethod
    def eat(self):
        pass

class LeeJungJae(Player):
    def eat(self, food: str): # 1번을 하게한다
        """ # 협업을 위한 type hint style
        >>> LeejungJae.eat("apple")
        ... apple
        """
        print(food)

    def eat(self, foods: List[str]): # 2번을 하게한다
        """
        >>> LeejungJae.eat(["apple", "juice"])
        ... apple
        ... juice
        """
        for food in foods:
            print(food)

    def eat(self, foods: Dict[str, int]): # 3번을 하게한다 이를 위해 오버로딩을 한다.
        """
        >>> LeejungJae.eat({"apple": 3, "juice": 500})
        ... apple
        ... juice
        """
        for k, _ in foods.items(): # 기존 for k, v in foods.items(): 였으나 v를 사용하지 않으므로 v 대신 _를 사용한다. 하지만 아예 사용하지 않으니 아예 _도 지워줘도 된다.
            print(k)
            # apple 3
            # juice 500

# 이렇게 하면 python은 실행되지 않고 그건 다음 주에 배운다
# 사용하지 않은 변수는 vs code상 연한 색으로 표시된다. 현재는 python 인터프리터를 설정하지 않았기 때문에 색상이 안보인다.

jungjae = LeeJungJae()
jungjae.eat("apple") # 원래라면 이것만 작동되야 하지만 밑에 까지 되게 하기위해 오버로딩을 한다. 1번
jungjae.eat(["apple", "bread", "juice"]) # 자료구조가 리스트 2번
jungjae.eat({"apple": 1, "juice":500}) # 자료구조가 딕셔널 3번
