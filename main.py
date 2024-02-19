import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from transformers import pipeline


st.set_page_config(page_title="Test",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    body{
    background-color: #27221e;
    }
    [data-testid="stAppViewContainer"]{
        background-color: #27221e;
    }
    [data-testid="stHeader"]{
        background-color: #27221e;
    }
    [data-testid="stToolbar"]{
        color: ##f3d8ba;
    }
    [data-testid="baseButton-secondary"]{
        background-color: #f2ae72;
        color:#27221e;
    }
    [data-testid="stWidgetLabel"]{
        color:#f2ae72
    }
    [data-testid="baseButton-header"]{
        background-color: #f2ae72;
        color:#27221e;
    }
    [data-testid="baseButton-headerNoPadding"]{
        color:#27221e;
        background-color: #f2ae72;
    }
    [data-testid="stStatusWidget"]{
        color:#27221e;
        background-color: #f2ae72;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(f'<h1 style="color:#f3d8ba;font-size:24px;text-align:center;font-size:2.75rem;font-weight:700;font-family:"Source Sans Pro", sans-serif">{"Welcome to the Named Entity Recognition App!⚡"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#f3d8ba;font-size:24px">{"Token Classification"}</h1>', unsafe_allow_html=True)
input_text=st.text_input("Input: ",key="input")

submit=st.button("Compute")

@st.cache_resource
def classifier(text):
    # Use a pipeline as a high-level helper
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large",add_prefix_space=True)
    model = TFAutoModelForTokenClassification.from_pretrained("Astral7/roberta-large-finetuned-ner")
    pipe = pipeline("token-classification", model=model,tokenizer=tokenizer )
    return pipe(text)
entities=[] 

if input_text:
    entities=classifier(input_text)

entity_tag = {
 'B-eve':'EVE',
 'I-eve':'EVE',
 'B-org':'ORG',
 'I-org':'ORG',
 'B-gpe':'GPE',
 'I-gpe':'GPE',
 'B-geo':'GEO',
 'I-geo':'GEO',
 'B-nat':'NAT',
 'I-nat':'NAT',
 'B-per':'PER',
 'I-per':'PER',
 'B-art':'ART',
 'I-art':'ART',
 'B-tim':'TIM',
 'I-tim':'TIM',
}
tag_out_styles = {
 'B-eve':"eve_out",
 'I-eve':'eve_out',
 'B-org':'org_out',
 'I-org':'org_out',
 'B-gpe':'gpe_out',
 'I-gpe':'gpe_out',
 'B-geo':'geo_out',
 'I-geo':'geo_out',
 'B-nat':'nat_out',
 'I-nat':'nat_out',
 'B-per':'per_out',
 'I-per':'per_out',
 'B-art':'art_out',
 'I-art':'art_out',
 'B-tim':'tim_out',
 'I-tim':'tim_out',
}
tag_in_styles = {
 'B-eve':'eve_in',
 'I-eve':'eve_in',
 'B-org':'org_in',
 'I-org':'org_in',
 'B-gpe':'gpe_in',
 'I-gpe':'gpe_in',
 'B-geo':'geo_in',
 'I-geo':'geo_in',
 'B-nat':'nat_in',
 'I-nat':'nat_in',
 'B-per':'per_in',
 'I-per':'per_in',
 'B-art':'art_in',
 'I-art':'art_in',
 'B-tim':'tim_in',
 'I-tim':'tim_in',
}

custom_style_tag=f"""
<style>
.fl{{
    width:fit-content;
    display:flex;
    align-items:center;
    background-color:#27221e;
}}
.tag_out{{
border-radius:0.25rem;
padding-left: 0.25rem;
padding-right: 0.25rem;
display:flex;
width:max-content;
align-items:center;
margin-right:0.25rem;
}}
.tag_in{{
font-weight:600;
font-size:.75rem;
line-height: 1rem;
border-radius:0.25rem;
margin-left:0.25rem;
height:18px;
padding-left:0.25rem;
padding-right:0.25rem;
margin-top:0.20rem;
margin-bottom:0;
}}
.org_out{{
color:rgb(17 94 89);
background-color: rgb(204 251 241);
}}
.org_in{{
color: rgb(204 251 241);
background-color:rgb(20 184 166);
}}
.per_out{{
color:rgb(91 33 182);
background-color:rgb(237 233 254);
}}
.per_in{{
color: rgb(237 233 254);
background-color:rgb(139 92 246);
}}
.gpe_out{{
color:rgb(134 25 143);
background-color:rgb(250 232 255);
}}
.gpe_in{{
color:rgb(250 232 255);
background-color: rgb(217 70 239);
}}
.eve_in{{
color: rgb(254 226 226);
background-color:rgb(239 68 68) ;
}}
.eve_out{{
color: rgb(239 68 68);
background-color: rgb(254 226 226);
}}
.nat_in{{
color: rgb(255, 239, 213);
background-color: rgb(255, 165, 0);
}}
.nat_out{{
color: rgb(255, 165, 0);
background-color: rgb(255, 239, 213);
}}
.tim_in{{
color: rgb(224 242 254);
background-color: rgb(14 165 233);
}}
.tim_out{{
color: rgb(14 165 233);
background-color:  rgb(224 242 254);
}}
.art_in{{
color: rgb(255, 240, 245);
background-color: rgb(255, 192, 203);
}}
.art_out{{
color: rgb(255, 192, 203);
background-color: rgb(255, 240, 245);
}}
.geo_out{{
color:rgb(157 23 77);
background-color:rgb(252 231 243);
}}
.geo_in{{
color: rgb(252 231 243);
background-color:rgb(236 72 153);
}}

</style>

<div class="fl">
{"".join([f"<span class='tag_out {tag_out_styles[entity['entity']]}'>{entity['word']}<p class='tag_in {tag_in_styles[entity['entity']]}'>{entity_tag[entity['entity']]}</p></span>" for entity in entities])}
</div>
"""

if submit:
    st.markdown(f"<p style='color:#f3d8ba'>Given Input: {input_text}</p>",unsafe_allow_html=True)
    st.markdown(f"<p style='color:#f3d8ba'>Output:</p>",unsafe_allow_html=True)
    st.markdown(custom_style_tag, unsafe_allow_html=True)
