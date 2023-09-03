import cohere
import streamlit as st
api_key = st.secrets['api_key']
co = cohere.Client(api_key)
def generate(prom):
    prompt = prom


    response = co.generate(  
        model='command-nightly',  
        prompt = prompt,  
        max_tokens=1000,  
        temperature=0.9)

    intro_paragraph = response.generations[0].text
    st.write(intro_paragraph)




def main():
    st.title('Generate Legal contracts with ease!!')
    prompt = st.text_input('Enter the prompt for the legal contract you want to generate please be specific')
    if prompt:
        generate(prompt)

if __name__ == '__main__':
    main()