import os

class Printing:

    @staticmethod
    def cout(msg, color="green", icon=None):
        colors = {
            "black": "\033[30m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m"
        }
        icons = {
            "check": "✔️",
            "error": "❌",
            "info": "ℹ️",
            "warn": "⚠️",
            "star": "⭐",
            "rocket": "🚀",
            "test": "🧪"
        }
        color_code = colors.get(color.lower(), colors["green"])
        icon_str = icons.get(icon, "") + " " if icon and icon in icons else ""
        print(f"{color_code}{icon_str}{msg}{colors['reset']}")

    @staticmethod
    def statmsg(msg, kind="info"):
        """
        Print a message with a specific kind (error, success, info, step, warn, test).
        kind: str, one of 'error', 'success', 'info', 'step', 'warn', 'test'.
        """
        mapping = {
            "error": {"color": "red", "icon": "error"},
            "success": {"color": "green", "icon": "check"},
            "info": {"color": "blue", "icon": "info"},
            "step": {"color": "yellow", "icon": "star"},
            "warn": {"color": "yellow", "icon": "warn"},
            "test": {"color": "magenta", "icon": "test"},
        }
        opts = mapping.get(kind, {"color": "blue", "icon": "info"})
        Printing.cout(msg, color=opts["color"], icon=opts["icon"])

    @staticmethod
    def statmsg_with_counter(
        msg: str,
        msg_no: int = 0,
        msg_amount: int = 0,
        kind: str = "info"
    ):
        """
        Print a message with a counter.
        msg: str, the message to print.
        msg_no: int, the current message number.
        msg_amount: int, the total number of messages.
        """
        if msg_amount > 0:
            counter = f"[{msg_no}/{msg_amount}] "
        else:
            counter = ""
        Printing.statmsg(f"{counter}{msg}", kind=kind)

class OutputManager():
    def __init__(
            self,
            output_root_dir: str = "./output",
            dir_weights: str = "weights",
            dir_results: str = "results",
            dir_logs: str = "logs"
    ):
        self.output_root_dir = output_root_dir
        self.dir_weights = dir_weights
        self.dir_results = dir_results
        self.dir_logs = dir_logs

        # check if the output directories exist, if not, create them
        if not self.check_directories():
            Printing.statmsg("Creating output directories...", kind="info")
            self.create_directories()
            Printing.statmsg("Output directories created successfully.", kind="success")
        else:
            #print found 
            Printing.statmsg("Output directories already exist.", kind="info")
            
    def create_directories(self):
        """
        Create the output directories if they do not exist.
        """
        os.makedirs(self.output_root_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_root_dir, self.dir_weights), exist_ok=True)
        os.makedirs(os.path.join(self.output_root_dir, self.dir_results), exist_ok=True)
        os.makedirs(os.path.join(self.output_root_dir, self.dir_logs), exist_ok=True)

    def check_directories(self):
        """
        Check if the output directories exist.
        """
        if not os.path.exists(self.output_root_dir):
            Printing.statmsg(f"Output root directory '{self.output_root_dir}' does not exist.", kind="error")
            return False
        if not os.path.exists(os.path.join(self.output_root_dir, self.dir_weights)):
            Printing.statmsg(f"Weights directory '{self.dir_weights}' does not exist.", kind="error")
            return False
        if not os.path.exists(os.path.join(self.output_root_dir, self.dir_results)):
            Printing.statmsg(f"Results directory '{self.dir_results}' does not exist.", kind="error")
            return False
        if not os.path.exists(os.path.join(self.output_root_dir, self.dir_logs)):
            Printing.statmsg(f"Logs directory '{self.dir_logs}' does not exist.", kind="error")
            return False
        return True
    
    def get_output_path(self, subdir: str, filename: str) -> str:
        """
        Get the full path for a file in the specified subdirectory.
        subdir: str, the subdirectory within the output root directory.
        filename: str, the name of the file.
        """
        if not self.check_directories():
            Printing.statmsg("Output directories do not exist. Please create them first.", kind="error")
            return ""
        
        return os.path.join(self.output_root_dir, subdir, filename)

