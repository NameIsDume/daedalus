from langchain_core.tools import tool
import httpx

@tool
def linux_doc(command: str) -> str:
    """Fetch Linux manual page for a command."""
    base_command = command.strip().split()[0]  # ✅ On garde seulement la commande
    url = f"http://localhost:9000/get_doc?command={base_command}"
    resp = httpx.get(url, timeout=10)
    data = resp.json()
    if "error" in data:
        return f"No documentation found for '{base_command}'"
    print(f"[TOOL] linux_doc(command={command}) → Summary: {', '.join(data['summary'])}")
    return f"Command: {data['command']}\nSections: {', '.join(data['summary'])}"