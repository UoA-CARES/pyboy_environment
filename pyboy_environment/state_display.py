from tkinter import *


class StateDisplay:
    def __init__(self):
        self.tk = Tk()

        self.tk.title("State Display")
        self.tk.geometry("1200x1200")
        self.tk.resizable(True, True)

        self.frame = Frame(self.tk)
        self.frame.pack(fill=BOTH, expand=True)

        self.text_widget = Text(self.frame, wrap=WORD)
        self.text_widget.pack(side=TOP, fill=BOTH, expand=True, anchor="n")

        self.tk.update()

    def update_display(self, state: dict):
        self.text_widget.delete(1.0, END)  # Clear previous content

        for key, value in state.items():
            valueStr = str(value)
            self.text_widget.insert(END, f"{key}: {valueStr}\n")

        self.tk.update_idletasks()
        self.tk.update()
