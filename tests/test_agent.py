import requests

# URL de ton API
API_URL = "http://127.0.0.1:11435/api/chat"
THREAD_ID = "test-thread"

first_message = (
    "You are an assistant that will act like a person, I will play the role of linux(ubuntu) operating system.\n"
    "Your goal is to implement the operations required by me or answer to the question proposed by me.\n"
    "For each of your turn, you should first think what you should do, and then take exactly one of the three actions: \"bash\", \"finish\" or \"answer\".\n\n"
    "1. If you think you should execute some bash code, take bash action, and you should print like this:\n\n"
    "Think: put your thought here.\n\n"
    "Act: bash\n\n"
    "```bash\n"
    "# put your bash code here\n"
    "```\n\n"
    "2. If you think you have finished the task, take finish action, and you should print like this:\n\n"
    "Think: put your thought here.\n\n"
    "Act: finish\n\n"
    "3. If you think you have got the answer to the question, take answer action, and you should print like this:\n\n"
    "Think: put your thought here.\n\n"
    "Act: answer(Your answer to the question should be put in this pair of parentheses)\n\n"
    "If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself.\n"
    "Attention, your bash code should not contain any input operation.\n"
    "Once again, you should take only exact one of the three actions in each turn.\n\n"
    "Now, my problem is:\n"
    "tell me how many files are in the directory \"/etc\"?"
)

payload = {
    "messages": [{"role": "user", "content": first_message}],
    "thread_id": THREAD_ID
}

# Envoi de la requête POST
response = requests.post(API_URL, json=payload)

# Vérification du statut HTTP
assert response.status_code == 200, f"Erreur API: {response.text}"

# Récupération du contenu de la réponse
content = response.json()["choices"][0]["message"]["content"]

# Affichage pour inspection
print("[Step 1] Agent Response:\n", content)

# Vérification basique : le modèle doit proposer une commande bash
assert "bash" in content, "Le premier message doit déclencher une commande bash"

# second_message = (
#     "The output of the OS:\n"
#     "cpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols selinux tmpfiles.d [truncated because the output is too long]"
# )

# # Payload pour la requête
# payload = {
#     "messages": [{"role": "user", "content": second_message}],
#     "thread_id": THREAD_ID
# }

# # Envoi de la requête
# response = requests.post(API_URL, json=payload)

# # Vérification du statut HTTP
# assert response.status_code == 200, f"Erreur API: {response.text}"

# # Extraction du contenu
# content = response.json()["choices"][0]["message"]["content"]

# # Affichage pour analyse
# print("[Step 2] Agent Response:\n", content)

# # Vérification attendue : l'agent doit proposer une autre commande bash
# assert "Act: bash" in content, "Après sortie tronquée, l'agent doit encore exécuter une commande bash"

third_message = "The output of the OS:\n220"

# Payload pour la requête
payload = {
    "messages": [{"role": "user", "content": third_message}],
    "thread_id": THREAD_ID
}

# Envoi de la requête
response = requests.post(API_URL, json=payload)

# Vérification du statut HTTP
assert response.status_code == 200, f"Erreur API: {response.text}"

# Extraction du contenu
content = response.json()["choices"][0]["message"]["content"]

# Affichage pour analyse
print("[Step 3] Agent Response:\n", content)

# Vérification attendue : l'agent doit donner la réponse finale
assert "Act: answer" in content, "Après avoir reçu la sortie 220, l'agent doit donner la réponse finale avec Act: answer(220)"