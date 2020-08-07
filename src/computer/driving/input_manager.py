"""
Manages the keyboard inputs to control the car and the recording
"""
from pynput.keyboard import Key, Listener, KeyCode
from variables import Command, Mode
from data_manager import DataManager

class InputManager:
    def __init__(self, data_manager: DataManager, application_mode: Mode):
        """Inits the input manager and starts the listener

        Args:
            data_manager (DataManager): Datamanager
            application_mode (Mode): Mode in which the application runs
        """        
        print("init inputmanager")
        self.data_manager = data_manager
        self.application_mode = application_mode
        
        self.w = False
        self.a = False
        self.s = False
        self.d = False
        self.q = False
        self.e = False
        self.r = False
        self.x = False
        self.up = False
        self.down = False
        self.left = False
        self.right = False

        self.recording = False
        self.self_driving = False

        self.r_pressed = False
        self.space_pressed = False

        self.quit = False

        listener = Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()

    def on_press(self, key: int) -> bool:
        """Sets corresponding var to true or returns False to end the listener.

        Args:
            key (int): Key code of pressed key

        Returns:
            bool: False if Esc or Q is pressed and listener should be stoped, else True
        """
        if key == KeyCode.from_char('w'): self.w = True
        if key == KeyCode.from_char('a'): self.a = True
        if key == KeyCode.from_char('s'): self.s = True
        if key == KeyCode.from_char('d'): self.d = True
        if key == KeyCode.from_char('q'): self.q = True
        if key == KeyCode.from_char('e'): self.e = True
        if key == KeyCode.from_char('x'): self.x = True
        if key == Key.up: self.up = True
        if key == Key.down: self.down = True
        if key == Key.left: self.left = True
        if key == Key.right: self.right = True

        if key == KeyCode.from_char('E'):
            self._reset_data()
        if key == KeyCode.from_char('r') and not self.r_pressed:
            self._toggle_recording()
        if key == Key.space and not self.space_pressed:  
            self._toggle_self_driving()
        if key == Key.esc:
            # Stop listener
            self.quit = True
            return False
        else: 
            return True

    def _reset_data(self) -> None:
        """Calls reset_data() and stops recording
        """        
        self.data_manager.reset_data()
        self.recording = False

    def _toggle_recording(self) -> None:
        """Toggle the recording. If we stop recording also save the recorded files
        """        
        if self.recording:
            self.data_manager.save_data()
            self.self_driving = False

        self.recording = ~self.recording
        self.r_pressed = True

    def _toggle_self_driving(self) -> None:
        """Toggle self_driving
        """        
        if self.application_mode == Mode.inference:
            self.self_driving = ~self.self_driving
            self.space_pressed = True

    def on_release(self, key: int) -> None:
        """Sets the corresponding var to False

        Args:
            key (int): Key code of released key
        """
        if key == KeyCode.from_char('w'): self.w = False
        if key == KeyCode.from_char('a'): self.a = False
        if key == KeyCode.from_char('s'): self.s = False
        if key == KeyCode.from_char('d'): self.d = False
        if key == KeyCode.from_char('q'): self.q = False
        if key == KeyCode.from_char('e'): self.e = False
        if key == KeyCode.from_char('x'): self.x = False
        if key == Key.up: self.up = False
        if key == Key.down: self.down = False
        if key == Key.left: self.left = False
        if key == Key.right: self.right = False

        if key == KeyCode.from_char('r'): self.r_pressed = False
        if key == Key.space: self.space_pressed = False

    def get_action(self) -> Command:
        """Checks all the keys and returns corresponding action

        Args:
            command (Command): Command mapping

        Returns:
            Command: Action
        """
        if self.x:
            return Command.stop
        elif self.a:
            return Command.left
        elif self.s:
            return Command.reverse
        elif self.d:
            return Command.right
        elif self.q:
            return Command.turn_left
        elif self.e:
            return Command.turn_right
        elif self.w:
            return Command.forward
        else: 
            return Command.no_command     

    def get_condition(self, condition: Command) -> Command:
        """Returns condition according to the pressed key

        Args:
            condition (Command): Current condition
            command (Command): Command mapping

        Returns:
            Command: Command
        """
        if self.up:
            condition = Command.forward
        elif self.left:
            condition = Command.left
        elif self.right:
            condition = Command.right

        return condition

