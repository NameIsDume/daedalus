# import requests

# # URL de ton API
# API_URL = "http://127.0.0.1:11435/api/chat"
# THREAD_ID = "test-thread"

# first_message = (
#     "You are an assistant that will act like a person, I will play the role of linux(ubuntu) operating system.\n"
#     "Your goal is to implement the operations required by me or answer to the question proposed by me.\n"
#     "For each of your turn, you should first think what you should do, and then take exactly one of the three actions: \"bash\", \"finish\" or \"answer\".\n\n"
#     "1. If you think you should execute some bash code, take bash action, and you should print like this:\n\n"
#     "Think: put your thought here.\n\n"
#     "Act: bash\n\n"
#     "```bash\n"
#     "# put your bash code here\n"
#     "```\n\n"
#     "2. If you think you have finished the task, take finish action, and you should print like this:\n\n"
#     "Think: put your thought here.\n\n"
#     "Act: finish\n\n"
#     "3. If you think you have got the answer to the question, take answer action, and you should print like this:\n\n"
#     "Think: put your thought here.\n\n"
#     "Act: answer(Your answer to the question should be put in this pair of parentheses)\n\n"
#     "If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself.\n"
#     "Attention, your bash code should not contain any input operation.\n"
#     "Once again, you should take only exact one of the three actions in each turn.\n\n"
#     "Now, my problem is:\n"
#     "tell me how many files are in the directory \"/etc\"?"
# )

# payload = {
#     "messages": [{"role": "user", "content": first_message}],
#     "thread_id": THREAD_ID
# }

# # Envoi de la requête POST
# response = requests.post(API_URL, json=payload)

# # Vérification du statut HTTP
# assert response.status_code == 200, f"Erreur API: {response.text}"

# # Récupération du contenu de la réponse
# content = response.json()["choices"][0]["message"]["content"]

# # Affichage pour inspection
# print("[Step 1] Agent Response:\n", content)

# # Vérification basique : le modèle doit proposer une commande bash
# # assert "bash" in content, "Le premier message doit déclencher une commande bash"

# third_message = "The output of the OS:\n220"

# # Payload pour la requête
# payload = {
#     "messages": [{"role": "user", "content": third_message}],
#     "thread_id": THREAD_ID
# }

# # Envoi de la requête
# response = requests.post(API_URL, json=payload)

# # Vérification du statut HTTP
# assert response.status_code == 200, f"Erreur API: {response.text}"

# # Extraction du contenu
# content = response.json()["choices"][0]["message"]["content"]

# # Affichage pour analyse
# print("[Step 3] Agent Response:\n", content)

# # Vérification attendue : l'agent doit donner la réponse finale
# # assert "Act: answer" in content, "Après avoir reçu la sortie 220, l'agent doit donner la réponse finale avec Act: answer(220)"

# fourth_message = "Now, I will start a new problem in a new OS. My problem is:\n\nTell me whether npm is installed or not. If so, return 'installed'. If not, return 'not-yet'"

# # Payload pour la requête
# payload = {
#     "messages": [{"role": "user", "content": fourth_message}],
#     "thread_id": THREAD_ID
# }

# response = requests.post(API_URL, json=payload)

# assert response.status_code == 200, f"Erreur API: {response.text}"

# fifth_message = "The output of the OS is:\nnot-yet"

# payload = {
#     "messages": [{"role": "user", "content": fifth_message}],
#     "thread_id": THREAD_ID
# }
# response = requests.post(API_URL, json=payload)
# assert response.status_code == 200, f"Erreur API: {response.text}"