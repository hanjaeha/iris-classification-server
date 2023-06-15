class Keyboard(ABC): # OCP를 활용하여 강제로 제조사들에게 규칙을 정해 만들게한다.
    @abstractmethod
    def store_input(self, input)
        pass

    @abstractmethod
    def send_input(self):
        pass

class SamsungKeyboard(Keyboard):
    def store_input(self, input):
        pass

    def send_input(self, input):
        pass

class SamsungKeyboard: # 삼성과 애플은 다른 회사기 때문에 삼성은 삼성만의 변수명, 애플은 애플만의 변수명이 있을 것 이다.
    def __init__(self) -> None:
        self.user_input: str = ""

    def save_user_input(self, input: str) -> None: # 일부러 애플과 명칭에서 차별점을 둠
        self.user_input = input

    def give_user_input(self) -> str:
        return self.user_input


##### 개방 폐쇄의 원칙 (OCP) 을 배우기 위한 예제 #####
class AppleKeyboard:
    """Apple Keyboard"""
    def __init__(self) -> None:
        """keyboard input and touchbar"""
        self.keyboard_input: str = "" # 키보드에 입력한 값을 저장

    def set_keyboard_input(self, input: str) -> None:
        """Store keyboard input"""
        self.keyboard_input = input # input에 새로운 것을 입력 (아래와 달리 return은 안된다.)

    def send_keyboard_input(self) -> str:
        """Send text input method"""
        return self.keyboard_input # 다른 method들 한테 값을 반환(return)해준다.

# 컴퓨터 안에 키보드를 관리할 수 있는 것을 만들어야 하니 새 클래스 생성

class keyboardManager:
    def __init__(self) -> None:
        """Manage a keyboard class"""
        self.keyboard: AppleKeyboard = None # 이 None이

    def connect_to_keyboard(self, keyboard: AppleKeyboard) -> None:  # AppleKeyboard가 되고
        self.keyboard = keyboard

    def get_keyboard_input(self) -> str:
        """get text input from keyboard"""
#        if isinstance(self.keyboard, AppleKeyboard):
#            return self.keyboard.send_keyboard_input()
#        elif isinstance(self.keyboard, SamsungKeyboard):
#            return self.keyboard.give_user_input()
#        return self.keyboard.send_keyboard_input() # input을 사용할 수 있게 된다, 이것만 있으면 애플키보드만 되니까 다른 해결방법이 필요하여 위 4줄을 추가하였으나
                                                    # 그 외의 class 키보드들 (다른 제조사)은 사용이 불가능하지 OCP를 활용한다 (최상단)

keyboard_manager = KeyboardManager()
apple_keyboard = AppleKeyboard()
apple_keyboard.set_keyboard_input("Hello World!")
keyboard_manager.connect_to_keyboard(apple_keyboard)
keyboard_manager.get_keyboard_input()
# 최종적으로 Hello World! 반환 (출력)
