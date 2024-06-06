from tkinter import *

class StateDisplay:
    def __init__(self):
        self.tk = Tk()

        self.tk.title('State Display')
        self.tk.geometry("350x400")  # Increased height for better visibility
        self.tk.resizable(True, True)

        self.frame = Frame(self.tk)
        self.frame.pack(fill=BOTH, expand=True)

        self.text_widget = Text(self.frame, wrap=WORD)
        self.text_widget.pack(side=TOP, fill=BOTH, expand=True, anchor='n')

        self.tk.update()

    def update_display(self, state: dict):
        self.text_widget.delete(1.0, END)  # Clear previous content

        def format_value(value, indent=0):
            if isinstance(value, dict):
                formatted_str = ""
                for k, v in value.items():
                    formatted_str += ' ' * indent + f"{k}: {format_value(v, indent + 2)}\n"
                return formatted_str
            elif isinstance(value, list):
                formatted_str = "[\n"
                for item in value:
                    formatted_str += ' ' * (indent + 2) + f"{format_value(item, indent + 2)}\n"
                formatted_str += ' ' * indent + "]"
                return formatted_str
            else:
                return str(value)

        for key, value in state.items():
            value_str = format_value(value, 2)
            self.text_widget.insert(END, f"{key}:\n{value_str}\n\n")

        self.tk.update_idletasks()
        self.tk.update()
