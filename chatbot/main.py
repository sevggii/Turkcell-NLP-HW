# Chatbot geliştirmek ve bunu bir arayüz ile kullanıcıya sunmak.


# Veri -> İşin en zor kısmı.
# LLM = Large Language Model


# 1. teknik -> Herhangi bir modeli olduğu gibi kullanmak
# 2. teknik -> Modeli alıp özel veriyle donatma -> Parametre sayısı


# Transfer-Learning -> Daha önceden büyük çaplı bir veriyle eğitilmiş bir modeli alıp kendi verimizle çalışabilecek duruma getirmek.
# Derin Öğrenme => Sıfırdan bir beyin eğitmek.
# Transfer Learning -> Benzer verilerle eğitilmiş bir beyni kendi verimize adapte etmek..


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat_with_bot(user_input):
    prompt = (
        "User:\tDoes money buy happiness?\n"
        "Bot:\tDepends how much money you spend on it .\n"
        "User:\tWhat is the best way to buy happiness ?\n"
        "Bot:\tYou just have to be a millionaire by your early 20s, then you can be happy .\n"
        "User:\tThis is so difficult !\n"
        f"User:\t{user_input}\nBot:"
    )
    response = generator(prompt, max_length=100, pad_token_id=tokenizer.eos_token_id)
    generated_text = response[0]['generated_text']
    if "Bot:" in generated_text:
        return generated_text.split("Bot:")[-1].strip()
    return generated_text

iface = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(lines=2, placeholder="Bir soru yazın..."),
    outputs="text",
    title="DialoGPT Chatbot",
    description="Sorunuzu yazın, model cevaplasın."
)

if __name__ == "__main__":
    iface.launch(share=True)
    ## .git deneme