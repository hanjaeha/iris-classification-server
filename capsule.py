##### 캡슐화 #####
class Capsule:
    def __init__(self) -> None:
        print("Hello, object!")


if __name__ == "__main__":
    capsule = Capsule() # Ctrl + Shift + ~ (터미널 켜기)
# ls (list)
# dir
# python .\capsule.py (뒤는 자동완성) __는 특별메소드를 뜻함

# -------------------------------------------------------------------

class SquidGame:
    def __init__(self) -> None:
        self.name = "Jae-ha Han"
        self.__age = 29 # 이렇게 하지 말고
        self._age = 29 # 이렇게 하기로 함, 앞에 언더바가 붙으면 이 클래스 안에서만 사용하는 속성을 뜻함 (외부에선 사용하지 못한다)

        self._generate_player()

    def _generate_player(self):
        self._age

# 여기가 외부
squid_game = SquidGame()
squid_game.generate_player()
# squid_game._age (외부라서 _age는 사용하지 못한다)

# -------------------------------------------------------------------

# def # 클래스 밖에서 def를 생성하게되면 함수

class SquidGame:
    def __init__(self, name: str, age: int) -> None: # 인자를 설정하여 파라미터를 넣을 수 있게 한다
        """
        Args:
            name (str): e.g. Kyung-Su
            age (int): yout age

        """ # 이렇게 설정하면 다른 사람과 협업할 때 type hint를 줄 수 있어 업무 효율에 좋다.

#       self.name = "Jae-ha Han"
#       self.name: str = "Jae-ha Han" # 이러면 문자열 (str)로 받아들인다.
        self.name: str = name # 이렇게 하면 파라미터를 각각 설정할 수 있다.
        if age > 40: # 나이가 40세 밑으로 설정하게 한다.
            raise ValueError("나이가 40세를 넘었습니다!") # 오류 발생시키려면 raise를 사용한다.
#       self.age: int = 29 # 이러면 정수형 (int)로 받아들인다.
        self.age: int = age # 이렇게 하면 파라미터를 각각 설정할 수 있다.
                        # python에서는 이렇게 설정을 하지 않아도 괜찮지만 다른 언어에서는 직접 지정을 해줘야하고 python에서도 이렇게 지정을 해주는 것이 디자인패턴상 좋다.

    @property
    def age(self) -> int: # -> int로 type hint 넣음
        return self.age

    @age.setter # setter와 getter는 반대되는 기능
    def age(self, age: int) -> None:
        if age <= 40 and isinstance(self.age, str): # isinstance라는 함수로 age에 대한 데이터타입을 체크해줌
            raise ValueError("나이가 40세를 넘었습니다!") 
        self.age = int = age
        
    # 위 @property와 @setter를 사용하여 자료형 검사와 40세 미만의 if문 까지 설정 가능하다.

# half number = 39.2 (C언어의 경우 이런식으로 자료형을 먼저 선언해준다. 왜냐면 python은 동적인 언어이고 C언어는 정적인 언어이기 때문이다.)

squid_game1 = SquidGame("Jae-ha", 29)
squid_game1.age = 150 # Python에서는 150이 잘 넣어지게 된다
squid_game1.age = "150" # Python에서는 "150"이 잘 넣어지게 된다 이 둘을 방지해야한다, 이를 위해 위에 def age(self)문으로 체크를 한다.
squid_game1.name
squid_game2 = SquidGame("Su-jeong", 22)

print(squid_game.age) # 29 정수값 출력
squid_game.age = 10 # 이러면 나이가 10으로 바뀌게 된다
print(squid_game.age) # 10 정수값 출력
# squid_game.age = "10" # 이러면 문자열(str)로 받아들이게 되므로 안된다.

# 코딩도장에서 python 복습하거나 모르는 개념에 대해서 Google에 검색하여 알아보면된다.