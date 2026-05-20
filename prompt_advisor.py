import streamlit as st
from groq import Groq
import re

# ── Reads key securely from .streamlit/secrets.toml ──────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""
# ─────────────────────────────────────────────────────────────────────────────

def parse_output(output: str):
    """Robustly extract PROMPT and NEGATIVE from any format."""
    prompt_line   = ""
    negative_line = ""

    # Try line by line first
    for line in output.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("PROMPT:"):
            prompt_line = stripped[7:].strip().strip('"')
        elif stripped.upper().startswith("NEGATIVE:"):
            negative_line = stripped[9:].strip().strip('"')

    # Fallback: regex search anywhere in text
    if not prompt_line:
        m = re.search(r'PROMPT:\s*(.+?)(?=NEGATIVE:|$)', output, re.IGNORECASE | re.DOTALL)
        if m:
            prompt_line = m.group(1).strip().strip('"')

    if not negative_line:
        m = re.search(r'NEGATIVE:\s*(.+?)$', output, re.IGNORECASE | re.DOTALL)
        if m:
            negative_line = m.group(1).strip().strip('"')

    return prompt_line, negative_line

def render_prompt_advisor():

    if not GROQ_API_KEY:
        st.warning("⚠️ Add GROQ_API_KEY in .streamlit/secrets.toml")
        return None

    col1, col2 = st.columns([3, 1])
    with col1:
        user_idea = st.text_area(
            "Describe your idea",
            placeholder="e.g. A young female doctor standing in a hospital, friendly smile",
            height=80, key="user_idea", label_visibility="collapsed"
        )
    with col2:
        style = st.selectbox("Style", [
            "Photorealistic", "Cartoon / Pixar", "Anime",
            "Fantasy Art", "Cyberpunk", "Watercolour", "3D Render",
        ], key="advisor_style")

    if st.button("✨ Generate Prompt", key="gen_prompt_btn"):
        if not user_idea.strip():
            st.error("Please describe your idea first.")
            return None

        with st.spinner("🤖 Generating your prompt..."):
            try:
                client = Groq(api_key=GROQ_API_KEY)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert Stable Diffusion / Juggernaut XL prompt engineer.
Convert the user's simple idea into a highly optimized image generation prompt.
Rules:
- Start with quality boosters: RAW photo, highly detailed, sharp focus, 8k uhd
- Include lighting: natural lighting / golden hour / studio lighting
- Include style keywords based on requested style
- Include camera info if photorealistic: canon eos r5, 85mm lens, bokeh
- Keep under 100 words
- Output MUST be exactly 2 lines, nothing else

Line 1: PROMPT: [your prompt here]
Line 2: NEGATIVE: [negative prompt here]"""
                        },
                        {"role": "user", "content": f"Style: {style}\nIdea: {user_idea}"}
                    ],
                    max_tokens=300, temperature=0.7,
                )

                output = response.choices[0].message.content.strip()
                prompt_line, negative_line = parse_output(output)

                if prompt_line:
                    st.success("✅ Prompt generated!")

                    st.markdown("**Generated Prompt:**")
                    st.code(prompt_line, language=None)

                    st.markdown("**Generated Negative Prompt:**")
                    st.code(negative_line, language=None)

                    st.info("👆 Copy these into the Prompt and Negative Prompt boxes and click Generate!")

                    st.session_state["generated_prompt"]   = prompt_line
                    st.session_state["generated_negative"] = negative_line
                else:
                    st.warning("Could not parse. Raw output:")
                    st.write(output)

            except Exception as e:
                st.error(f"Groq error: {str(e)}")
    return None