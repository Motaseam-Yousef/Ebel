import os
from langchain import PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import openai,OpenAIChat
from gtts import gTTS
import streamlit as st
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

import os
#os.environ["OPENAI_API_KEY"] = "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"

#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "key"

#llm = OpenAI(temerature=0.9, model_name="text-davinci-003")

# Print the content
#print(response_dict["choices"][0]["message"]["content"])

st.title("🐫 Camel 911 🐫")

prefix_messages = [{"role": "system", "content": "You are a Camels stuff assisst"}]

llm = OpenAIChat(model_name='gpt-3.5-turbo',
                openai_api_key="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",
                temperature=0,
                prefix_messages=prefix_messages,
                max_tokens = 500)

template = """{مدخل}
أنت مساعد ذكي لمالكي وهواة الإبل تجيب على كل الأسئلة المتعلقة بحياة الإبل وصحة الإبل وجميع أسماء الإبل 
لا يجب عليك أن تخترع معلومات أو تتخيل أي معلومات غير المعلومات الحقيقة الموجودة في الداتا التي قمت بالتدريب عليها 
يجب أن تكون جميع أجوبتك مختصرة وبلغة سهلة يفهمها الجميع
إذا كانت هناك مدخلات ليست موجودة في قاعدة البيانات الخاصة بك لا تقم بإختراع أي إجابات فقط أجب "لا أعرف الإجابة يجب عليك سؤال شخص مختص "
يجب أن تكون جميع أجاباتك باللغة العربية دون إستخدام أي كلمة بلغة أخرى 

اعرف أن:
شداد الخرج أو الخرك و هو ما يوضع على ظهر الجمل و يوضع به المعدات الخفيفة بالسفر
"""

prompt = PromptTemplate(template=template, 
                        input_variables=["مدخل"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

مدخل = st.text_input("What's Your issue?!") 

response = llm_chain.run(مدخل)
responseSound = gTTS(text=response, lang="ar", slow=False)
responseSound.save("res.mp3")

audio_file = open('res.mp3', 'rb')
audio_bytes = audio_file.read()

# Use Streamlit to play the audio file
st.write(response)
st.audio(audio_bytes, format='audio/mp3')

# Convert mp3 file to mp3
# audio = AudioSegment.from_mp3("res.mp3")
# audio.export("res.mp3", format="mp3")

# Open the mp3 file
# audio_file = open('res.mp3', 'rb')
# audio_bytes = audio_file.read()
# audio = AudioSegment.from_file("res.mp3")
# fast_audio = audio.speedup(playback_speed=2.0)

# st.audio(fast_audio.raw_data, format='audio/mp3')
# sample_rate = 44100  # 44100 samples per second
# seconds = 1  # Note duration of 2 seconds
# frequency_la = 500  # Our played note will be 440 Hz
# # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
# t = np.linspace(0, seconds, seconds * sample_rate, False)
# # Generate a 440 Hz sine mp3e
# note_la = np.sin(frequency_la * t * 2 * np.pi)

# st.audio(note_la, sample_rate=sample_rate)

#if response: 
###    res= chain.run(result)

#    st.write(response_dict["choices"][0]["message"]["content"])
