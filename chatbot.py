# chatbot.py
from bot import pred_class, get_response, bert_model, model, label_encoder, responses_dict

print("Press 0 If you don't want to chat with our ChatBot.")

while True:
    message = input("You: ")
    if message == "0":
        print("ChatBot: Take care! We're here if you need us.")
        break
    intent = pred_class(message, bert_model, model, label_encoder)
    response = get_response(intent, responses_dict)
    print("ChatBot:", response)