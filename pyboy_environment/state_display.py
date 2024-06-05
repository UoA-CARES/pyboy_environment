from tkinter import *

class StateDisplay():
    def __init__(self):
        
        self.tk = Tk()

        self.tk.title('State Display')
        self.tk.geometry("350x200")
        self.tk.resizable(True,True)

        self.frame = Frame(self.tk)
        self.frame.pack(fill=BOTH)

        self.display_label = Label(self.tk, text="Test", justify=LEFT)
        self.display_label.pack(fill=BOTH, expand=True)

        self.tk.update()

    def update_display(self, state: dict[str, any]):

        result = ""

        for key, value in state.items():
            valueStr = str(value)
            result += f"{key}: {valueStr}\n"

        self.display_label.config(text=result)

        self.tk.update_idletasks()
        self.tk.update()
