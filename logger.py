from termcolor import colored

class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.thread_flags = {}  # Pour gérer les répétitions par thread

    def log_message(self, msg, thread_id=None):
        msg_type = type(msg).__name__
        content = msg.content.strip()

        # Init des flags pour ce thread
        if thread_id not in self.thread_flags:
            self.thread_flags[thread_id] = {
                "sys_printed": False,
                "last_ai_msg": "",
                "final_printed": False
            }

        flags = self.thread_flags[thread_id]

        # Masquer le contenu <think> sauf si verbose=True
        if "<think>" in content and not self.verbose:
            return

        # Mapping des couleurs
        prefix = {
            "SystemMessage": colored("[SYS]", "cyan"),
            "HumanMessage": colored("[USER]", "green"),
            "AIMessage": colored("[AGENT]", "yellow"),
        }.get(msg_type, "[MSG]")

        # ✅ Filtrer les répétitions
        if msg_type == "SystemMessage":
            if flags["sys_printed"]:
                return
            flags["sys_printed"] = True

        if msg_type == "AIMessage":
            # Si déjà affiché et pas verbose, skip
            if not self.verbose and flags["final_printed"]:
                return

            if "<final_answer>" in content:
                # Extraire uniquement le bloc final
                final_text = content.split("<final_answer>")[-1].replace("</final_answer>", "").strip()
                print(f"{prefix} {final_text}")
                flags["final_printed"] = True
                return

            # Si verbose, affiche tout mais évite doublons exacts
            if content == flags["last_ai_msg"]:
                return
            flags["last_ai_msg"] = content

        print(f"{prefix} {content}")

    def log_tool(self, tool_name, args, result_summary):
        print(colored(f"[TOOL]", "magenta"), f"{tool_name}({args}) → Summary: {result_summary}")
