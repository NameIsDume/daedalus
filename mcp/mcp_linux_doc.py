# mcp_linux_doc.py
import subprocess
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI(title="Linux Doc MCP", description="MCP server for Linux command documentation")

@app.get("/get_doc")
async def get_doc(command: str = Query(..., description="Linux command")):
    # ✅ Extraire la commande principale (avant les espaces)
    base_command = command.strip().split()[0]

    try:
        # ✅ Récupérer la doc via man
        result = subprocess.run(["man", base_command], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return {"error": f"No documentation found for '{base_command}'"}

        doc_text = result.stdout
        summary = []
        for section in ["NAME", "SYNOPSIS", "DESCRIPTION"]:
            if section in doc_text:
                summary.append(section)

        return {
            "command": base_command,
            "summary": summary,
            "full_doc": doc_text
        }
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
